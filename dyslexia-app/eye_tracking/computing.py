"""Core eye-tracking computations: head-pose estimation, gaze angle extraction,
and the main OpenCV/MediaPipe capture loop."""
import math
import time
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
from utils import NOSE_INDICES


def compute_scale(pts):
    """Mean pairwise distance of a set of 3-D points (used as a distance proxy)."""
    n = len(pts)
    total, count = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(pts[i] - pts[j])
            count += 1
    return total / count if count else 1.0


def compute_head_frame(lms, indices, ref, w, h):
    """Estimate a stable head-pose rotation matrix from a cluster of landmarks.

    Returns ``(centroid, R, pts)``.
    """
    pts = np.array([[lms[i].x * w, lms[i].y * h, lms[i].z * w] for i in indices])
    c = np.mean(pts, axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov((pts - c).T))
    eigvecs = eigvecs[:, np.argsort(-eigvals)]
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1
    ro, pi, ya = Rscipy.from_matrix(eigvecs).as_euler('zyx', degrees=False)
    R = Rscipy.from_euler('zyx', [ro, pi, ya]).as_matrix()
    if ref[0] is None:
        ref[0] = R.copy()
    else:
        for i in range(3):
            if np.dot(R[:, i], ref[0][:, i]) < 0:
                R[:, i] *= -1
    return c, R, pts


def gaze_to_angles(d):
    """Convert a 3-D gaze vector to ``(yaw_deg, pitch_deg)``.

    Positive yaw = right, positive pitch = down.
    """
    d = d / np.linalg.norm(d)
    xz = np.array([d[0], 0, d[2]])
    xz /= np.linalg.norm(xz)
    yaw = math.acos(np.clip(np.dot([0, 0, -1], xz), -1, 1))
    if d[0] < 0:
        yaw = -yaw
    yz = np.array([0, d[1], d[2]])
    yz /= np.linalg.norm(yz)
    pitch = math.acos(np.clip(np.dot([0, 0, -1], yz), -1, 1))
    if d[1] > 0:
        pitch = -pitch
    return -math.degrees(yaw), math.degrees(pitch)


def eye_tracking_loop(shared):
    """Main capture loop (runs in a background thread).

    On the first frame where a face is detected the eye offsets are
    initialised automatically — no calibration step required.  Every
    subsequent frame computes per-eye yaw/pitch angles and, when
    ``shared["recording"]`` is ``True``, appends them to ``shared["gaze_log"]``.

    Args:
        shared: Thread-safe state dict with keys:
            ``lock``, ``running``, ``recording``, ``gaze_log``,
            ``tracking``, ``frame_jpeg``.
    """
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref = [None]
    initialised = False
    loff = roff = lcal = rcal = None
    base_r = 20
    frame_times: deque = deque(maxlen=30)

    while True:
        with shared["lock"]:
            if not shared["running"]:
                break
            recording = shared["recording"]

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_times.append(time.perf_counter())
        if len(frame_times) == 30:
            fps = 29 / (frame_times[-1] - frame_times[0])
            with shared["lock"]:
                shared["fps"] = round(fps, 1)

        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            hc, R, npts = compute_head_frame(lms, NOSE_INDICES, ref, w, h)
            li = lms[468]
            ri = lms[473]
            il = np.array([li.x * w, li.y * h, li.z * w])
            ir = np.array([ri.x * w, ri.y * h, ri.z * w])

            # Auto-initialise eye offsets on the first frame with a face
            if not initialised:
                cur = compute_scale(npts)
                cl = R.T @ np.array([0, 0, 1])
                loff = R.T @ (il - hc) + base_r * cl
                roff = R.T @ (ir - hc) + base_r * cl
                lcal = rcal = cur
                initialised = True
                with shared["lock"]:
                    shared["tracking"] = True

            cur = compute_scale(npts)
            sl = hc + R @ (loff * (cur / lcal))
            sr = hc + R @ (roff * (cur / rcal))

            dl = il - sl
            dl /= np.linalg.norm(dl)
            dr = ir - sr
            dr /= np.linalg.norm(dr)

            yaw_l, pitch_l = gaze_to_angles(dl)
            yaw_r, pitch_r = gaze_to_angles(dr)

            # Draw debug overlays on the camera thumbnail
            rl2 = max(1, int(base_r * (cur / lcal)))
            rr2 = max(1, int(base_r * (cur / rcal)))
            cv2.circle(frame, (int(sl[0]), int(sl[1])), rl2, (255, 255, 25), 2)
            cv2.circle(frame, (int(sr[0]), int(sr[1])), rr2, (25, 255, 255), 2)
            avg_d = (dl + dr) / 2
            avg_d /= np.linalg.norm(avg_d)
            orig = (sl + sr) / 2
            cv2.line(frame,
                     tuple(int(v) for v in orig[:2]),
                     tuple(int(v) for v in (orig + avg_d * 180)[:2]),
                     (255, 255, 10), 2)
            cv2.putText(frame, "TRACKING", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

            if recording:
                with shared["lock"]:
                    shared["gaze_log"].append({
                        "time":    time.time(),
                        "angle1_l":   yaw_l,
                        "angle2_l": pitch_l,
                        "angle1_r":   yaw_r,
                        "angle2_r": pitch_r,
                    })

            for lm in lms:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (180, 180, 180), -1)

        _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        with shared["lock"]:
            shared["frame_jpeg"] = enc.tobytes()
        time.sleep(0.01)

    cap.release()
    face_mesh.close()
    with shared["lock"]:
        shared["running"] = False
        shared["frame_jpeg"] = None
