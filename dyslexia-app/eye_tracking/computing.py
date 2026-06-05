"""Core eye-tracking computations: head-pose estimation, gaze mapping, and the
main OpenCV/MediaPipe capture loop."""
from collections import deque
import math
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
from utils import MONITOR_HEIGHT, MONITOR_WIDTH, NOSE_INDICES, CALIB_POINTS_PCT
import time

# ══════════════════════════════════════════════════════════════════════════════
# ── Fonctions core ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def compute_scale(pts):
    """Compute the mean pairwise Euclidean distance between a set of 3-D points.

    Used to estimate the apparent size of the nose landmark cluster so that
    gaze offsets can be corrected for varying subject-to-camera distances.

    Args:
        pts: Array of shape ``(n, 3)``.

    Returns:
        Mean pairwise distance as a float, or ``1.0`` if fewer than two points
        are provided (prevents division-by-zero downstream).
    """
    n = len(pts)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total += np.linalg.norm(pts[i] - pts[j])
            count += 1
    return total / count if count else 1.0


def compute_head_frame(lms, indices, ref, w, h):
    """Estimate a stable head-pose rotation matrix from a cluster of landmarks.

    Performs PCA on the selected nose landmarks to derive three principal axes,
    enforces a right-handed coordinate system, and sign-aligns the result with
    the first frame's reference rotation so axes don't flip between frames.

    Args:
        lms: Sequence of MediaPipe ``NormalizedLandmark`` objects.
        indices: Indices into ``lms`` that form the nose cluster.
        ref: Single-element list ``[R | None]`` used to store the reference
            rotation across calls (mutated in-place on first call).
        w: Camera frame width in pixels.
        h: Camera frame height in pixels.

    Returns:
        A tuple ``(centroid, R, pts)`` where ``centroid`` is the 3-D mean of
        the cluster, ``R`` is the ``(3, 3)`` rotation matrix, and ``pts`` is
        the ``(n, 3)`` array of landmark coordinates.
    """
    pts = np.array([[lms[i].x * w, lms[i].y * h, lms[i].z * w] for i in indices])
    c = np.mean(pts, axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov((pts-c).T))
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
    """Convert a 3-D unit gaze vector to yaw and pitch angles in degrees.

    The convention is camera-space where ``-Z`` is straight ahead: positive yaw
    is right, positive pitch is down (before the final sign flip applied here).

    Args:
        d: 3-D gaze direction vector (need not be normalised).

    Returns:
        A tuple ``(yaw_deg, pitch_deg)`` where yaw is positive to the right and
        pitch is positive downward, both in degrees.
    """
    d = d/np.linalg.norm(d)
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


def fit_map(samples):
    """Fit a linear mapping from gaze angles to screen pixel coordinates.

    Solves two independent least-squares problems (one per screen axis) using
    the 9 calibration samples collected during the calibration phase.

    Args:
        samples: List of ``(yaw, pitch, screen_x, screen_y)`` tuples collected
            at known calibration point positions.

    Returns:
        A tuple ``(cx, cy)`` of 3-element coefficient vectors for the affine
        mapping ``screen_x = cx[0]*yaw + cx[1]*pitch + cx[2]`` (and similarly
        for ``cy``).
    """
    X = np.column_stack([[s[0] for s in samples], [s[1] for s in samples], np.ones(len(samples))])
    cx, *_ = np.linalg.lstsq(X, [s[2] for s in samples], rcond=None)
    cy, *_ = np.linalg.lstsq(X, [s[3] for s in samples], rcond=None)
    return cx, cy


def apply_map(yaw, pitch, cx, cy):
    """Apply a calibrated linear mapping to obtain screen coordinates.

    Args:
        yaw: Horizontal gaze angle in degrees.
        pitch: Vertical gaze angle in degrees.
        cx: X-axis coefficient vector from :func:`fit_map`.
        cy: Y-axis coefficient vector from :func:`fit_map`.

    Returns:
        A tuple ``(x, y)`` of integer pixel coordinates clipped to
        ``[10, MONITOR_WIDTH-10]`` × ``[10, MONITOR_HEIGHT-10]`` to avoid
        off-screen values.
    """
    return (int(np.clip(cx[0] * yaw + cx[1] * pitch + cx[2], 10, MONITOR_WIDTH-10)),
            int(np.clip(cy[0] * yaw + cy[1] * pitch + cy[2], 10, MONITOR_HEIGHT-10)))

# ══════════════════════════════════════════════════════════════════════════════
# ── Thread OpenCV ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def eye_tracking_loop(shared):
    """Main eye-tracking capture loop, intended to run in a background thread.

    Opens the default webcam, processes each frame with MediaPipe FaceMesh, and
    writes gaze coordinates and JPEG thumbnails to the ``shared`` state dict.
    The loop also handles calibration point capture and reset signals from the
    Streamlit UI.

    Args:
        shared: Thread-safe state dictionary with the following keys:

            - ``lock`` (threading.Lock): guards all reads/writes.
            - ``running`` (bool): set to ``False`` to stop the loop.
            - ``do_capture`` (bool): pulse to ``True`` to capture a calibration
              sample for the current calibration point.
            - ``do_reset`` (bool): pulse to ``True`` to restart calibration.
            - ``calib_step`` (int): current calibration step (0–8), or ``-1``
              when idle, or ``10`` when done.
            - ``calib_samples`` (list): accumulated ``(yaw, pitch, x, y)`` tuples.
            - ``calibrated`` (bool): set to ``True`` once 9 samples are fitted.
            - ``recording`` (bool): when ``True``, gaze entries are appended to
              ``gaze_log``.
            - ``gaze_log`` (list): list of gaze entry dicts.
            - ``screen_xy`` (tuple): most recent ``(x, y)`` gaze position.
            - ``frame_jpeg`` (bytes | None): latest JPEG-encoded camera frame.
    """
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref = [None]
    ll = rl = False
    loff = roff = lcal = rcal = None
    buf = deque(maxlen=10)
    cxm = cym = None
    base_r = 20

    while True:
        with shared["lock"]:
            if not shared["running"]:
                break
            do_cap = shared["do_capture"]
            do_rst = shared["do_reset"]
            recording = shared["recording"]
            if do_cap:
                shared["do_capture"] = False
            if do_rst:
                shared["do_reset"] = False
                shared["calib_samples"] = []
                shared["calib_step"] = -1
                shared["calibrated"] = False
                ll = rl = False
                ref[0] = None
                cxm = cym = None

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        sxy = (0, 0)

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            hc, R, npts = compute_head_frame(lms, NOSE_INDICES, ref, w, h)
            li = lms[468]
            ri = lms[473]
            il = np.array([li.x * w, li.y * h, li.z * w])
            ir = np.array([ri.x * w, ri.y * h, ri.z * w])

            if do_cap and not ll:
                cur = compute_scale(npts)
                cl = R.T@np.array([0, 0, 1])
                loff = R.T@(il - hc) + base_r * cl
                roff = R.T@(ir - hc) + base_r * cl
                lcal = rcal = cur
                ll = rl = True

            if do_cap and ll:
                cur = compute_scale(npts)
                sl = hc + R@(loff * (cur / lcal))
                sr = hc + R@(roff * (cur / rcal))
                dl = il - sl
                dl /= np.linalg.norm(dl)
                dr = ir - sr
                dr /= np.linalg.norm(dr)
                raw = (dl + dr) / 2
                raw /= np.linalg.norm(raw)
                ya, pi = gaze_to_angles(raw)
                with shared["lock"]:
                    step = shared["calib_step"]
                    if 0 <= step <= 8:
                        px, py = CALIB_POINTS_PCT[step]
                        shared["calib_samples"].append(
                            (ya, pi, int(px * MONITOR_WIDTH),
                             int(py * MONITOR_HEIGHT))
                            )
                        if len(shared["calib_samples"]) >= 9:
                            cxm, cym = fit_map(shared["calib_samples"])
                            shared["calibrated"] = True
                            shared["calib_step"] = 10
                        else:
                            shared["calib_step"] = step + 1

            if ll and rl and cxm is not None:
                cur = compute_scale(npts)
                sl = hc + R@(loff * (cur / lcal))
                sr = hc + R@(roff * (cur / rcal))
                rl2 = int(base_r * (cur / lcal))
                rr2 = int(base_r * (cur / rcal))
                dl = il - sl
                dl /= np.linalg.norm(dl)
                dr = ir - sr
                dr /= np.linalg.norm(dr)

                yaw_l, pitch_l = gaze_to_angles(dl)
                yaw_r, pitch_r = gaze_to_angles(dr)

                lx, ly = apply_map(yaw_l, pitch_l, cxm, cym)
                rx, ry = apply_map(yaw_r, pitch_r, cxm, cym)

                raw = (dl + dr) / 2
                raw /= np.linalg.norm(raw)
                buf.append(raw)
                avg = np.mean(buf, axis=0)
                avg /= np.linalg.norm(avg)
                ya, pi = gaze_to_angles(avg)
                sxy = apply_map(ya, pi, cxm, cym)
                cv2.circle(frame, (int(sl[0]), int(sl[1])), rl2, (255, 255, 25), 2)
                cv2.circle(frame, (int(sr[0]), int(sr[1])), rr2, (25, 255, 255), 2)
                orig = (sl + sr) / 2
                cv2.line(frame, tuple(int(v) for v in orig[:2]),
                         tuple(int(v) for v in (orig + avg * 180)[:2]), (255, 255, 10), 2)
                cv2.putText(
                        frame,
                        "CALIBRATED",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 100),
                        1
                    )

                if recording:
                    entry = {
                        "time": time.time(),
                        "fix_x": sxy[0],
                        "fix_y": sxy[1],
                        "angle1_l": yaw_l,
                        "angle2_l": pitch_l,
                        "angle1_r": yaw_r,
                        "angle2_r": pitch_r,
                        "gaze_x_left": lx,
                        "gaze_y_left": ly,
                        "gaze_x_right": rx,
                        "gaze_y_right": ry
                    }
                    with shared["lock"]:
                        shared["gaze_log"].append(entry)
            elif ll:
                cv2.putText(
                        frame,
                        "Calibration...",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 200, 0),
                        1
                    )
            else:
                cv2.circle(frame, (int(li.x * w), int(li.y * h)), 7, (255, 50, 50), 2)
                cv2.circle(frame, (int(ri.x * w), int(ri.y * h)), 7, (50, 255, 50), 2)

            for lm in lms:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (180, 180, 180), -1)

            with shared["lock"]:
                shared["screen_xy"] = sxy

        _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        with shared["lock"]:
            shared["frame_jpeg"] = enc.tobytes()
        time.sleep(0.01)

    cap.release()
    face_mesh.close()
    with shared["lock"]:
        shared["running"] = False
        shared["frame_jpeg"] = None
