import streamlit as st
import cv2
import numpy as np
import threading
import time
import mediapipe as mp
import math
from collections import deque
from scipy.spatial.transform import Rotation as Rscipy
import pyautogui
import streamlit.components.v1 as components

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eye Tracking",
    page_icon="👁️",
    layout="wide",
)

# ── Shared state (thread-safe) ────────────────────────────────────────────────
if "state" not in st.session_state:
    st.session_state.state = {
        "lock": threading.Lock(),
        "running": False,
        "frame_jpeg": None,
        "screen_xy": (0, 0),
        "calibrated": False,
        "do_calibrate": False,
        "do_reset": False,
        "thread": None,
        # état calibration UI : None | "waiting" | "done"
        "calib_ui_state": None,
    }

S = st.session_state.state

# ══════════════════════════════════════════════════════════════════════════════
# ── Fonctions core eye tracking ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()

nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391, 3, 248]

def compute_scale(points_3d):
    n = len(points_3d)
    total, count = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(points_3d[i] - points_3d[j])
            count += 1
    return total / count if count > 0 else 1.0

def compute_and_draw_coordinate_box(frame, face_landmarks, indices, ref_matrix_container, w, h):
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])
    center = np.mean(points_3d, axis=0)
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()
    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1
    return center, R_final, points_3d

def convert_gaze_to_screen(combined_dir, offset_yaw=0, offset_pitch=0):
    reference_forward = np.array([0, 0, -1])
    avg = combined_dir / np.linalg.norm(combined_dir)
    xz = np.array([avg[0], 0, avg[2]]); xz /= np.linalg.norm(xz)
    yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz), -1, 1))
    if avg[0] < 0: yaw_rad = -yaw_rad
    yz = np.array([0, avg[1], avg[2]]); yz /= np.linalg.norm(yz)
    pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz), -1, 1))
    if avg[1] > 0: pitch_rad = -pitch_rad
    yaw_deg = -math.degrees(yaw_rad) + offset_yaw
    pitch_deg = math.degrees(pitch_rad) + offset_pitch
    yawDeg, pitchDeg = 15, 5
    sx = int(((yaw_deg + yawDeg) / (2 * yawDeg)) * MONITOR_WIDTH)
    sy = int(((pitchDeg - pitch_deg) / (2 * pitchDeg)) * MONITOR_HEIGHT)
    return max(10, min(sx, MONITOR_WIDTH - 10)), max(10, min(sy, MONITOR_HEIGHT - 10))

# ══════════════════════════════════════════════════════════════════════════════
# ── Thread OpenCV ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def eye_tracking_loop(shared):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    R_ref_nose = [None]
    left_sphere_locked = right_sphere_locked = False
    left_sphere_local_offset = right_sphere_local_offset = None
    left_cal_scale = right_cal_scale = None
    gaze_dirs = deque(maxlen=10)
    calib_offset_yaw = calib_offset_pitch = 0
    base_radius = 20

    while True:
        with shared["lock"]:
            if not shared["running"]:
                break
            do_calib = shared["do_calibrate"]
            do_reset = shared["do_reset"]
            if do_calib: shared["do_calibrate"] = False
            if do_reset:
                shared["do_reset"] = False
                left_sphere_locked = right_sphere_locked = False
                R_ref_nose[0] = None

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            head_center, R_final, nose_pts = compute_and_draw_coordinate_box(
                frame, lms, nose_indices, R_ref_nose, w, h)

            left_iris  = lms[468]
            right_iris = lms[473]
            iris3d_l = np.array([left_iris.x*w,  left_iris.y*h,  left_iris.z*w])
            iris3d_r = np.array([right_iris.x*w, right_iris.y*h, right_iris.z*w])

            if do_calib and not (left_sphere_locked and right_sphere_locked):
                cur_scale = compute_scale(nose_pts)
                cam_dir_local = R_final.T @ np.array([0, 0, 1])
                left_sphere_local_offset  = R_final.T @ (iris3d_l - head_center) + base_radius * cam_dir_local
                right_sphere_local_offset = R_final.T @ (iris3d_r - head_center) + base_radius * cam_dir_local
                left_cal_scale = right_cal_scale = cur_scale
                left_sphere_locked = right_sphere_locked = True
                sphere_l = head_center + R_final @ left_sphere_local_offset
                sphere_r = head_center + R_final @ right_sphere_local_offset
                ld = iris3d_l - sphere_l; ld /= np.linalg.norm(ld)
                rd = iris3d_r - sphere_r; rd /= np.linalg.norm(rd)
                fwd = (ld + rd) / 2; fwd /= np.linalg.norm(fwd)
                raw_sx, raw_sy = convert_gaze_to_screen(fwd, 0, 0)
                calib_offset_yaw   = (MONITOR_WIDTH  / 2 - raw_sx) / (MONITOR_WIDTH  / 30)
                calib_offset_pitch = (MONITOR_HEIGHT / 2 - raw_sy) / (MONITOR_HEIGHT / 10)
                with shared["lock"]:
                    shared["calibrated"] = True
                    shared["calib_ui_state"] = "done"

            screen_xy = (0, 0)
            if left_sphere_locked and right_sphere_locked:
                cur = compute_scale(nose_pts)
                sr_l = cur / left_cal_scale  if left_cal_scale  else 1.0
                sr_r = cur / right_cal_scale if right_cal_scale else 1.0
                sphere_l = head_center + R_final @ (left_sphere_local_offset * sr_l)
                sphere_r = head_center + R_final @ (right_sphere_local_offset * sr_r)
                rad_l = int(base_radius * sr_l)
                rad_r = int(base_radius * sr_r)

                ld = iris3d_l - sphere_l; ld /= np.linalg.norm(ld)
                rd = iris3d_r - sphere_r; rd /= np.linalg.norm(rd)
                raw_dir = (ld + rd) / 2; raw_dir /= np.linalg.norm(raw_dir)
                gaze_dirs.append(raw_dir)
                avg_dir = np.mean(gaze_dirs, axis=0); avg_dir /= np.linalg.norm(avg_dir)

                sx, sy = convert_gaze_to_screen(avg_dir, calib_offset_yaw, calib_offset_pitch)
                screen_xy = (sx, sy)

                cv2.circle(frame, (int(sphere_l[0]), int(sphere_l[1])), rad_l, (255, 255, 25), 2)
                cv2.circle(frame, (int(sphere_r[0]), int(sphere_r[1])), rad_r, (25, 255, 255), 2)
                origin = (sphere_l + sphere_r) / 2
                end = origin + avg_dir * 200
                cv2.line(frame, tuple(int(v) for v in origin[:2]),
                         tuple(int(v) for v in end[:2]), (255, 255, 10), 2)
                cv2.putText(frame, f"Gaze: ({sx}, {sy})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "CALIBRATED", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                cv2.circle(frame, (int(left_iris.x*w), int(left_iris.y*h)), 8, (255, 50, 50), 2)
                cv2.circle(frame, (int(right_iris.x*w), int(right_iris.y*h)), 8, (50, 255, 50), 2)
                cv2.putText(frame, "Appuyer sur 'Calibrer'", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

            for lm in lms:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 1, (200, 200, 200), -1)

            with shared["lock"]:
                shared["screen_xy"] = screen_xy

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with shared["lock"]:
            shared["frame_jpeg"] = buf.tobytes()

        time.sleep(0.01)

    cap.release()
    face_mesh.close()
    with shared["lock"]:
        shared["running"] = False
        shared["frame_jpeg"] = None

# ══════════════════════════════════════════════════════════════════════════════
# ── Injection JS unique : point vert + point de calibration ──────────────────
# ══════════════════════════════════════════════════════════════════════════════

components.html("""
<script>
(function() {
  const WIN = window.parent;
  const DOC = WIN.document;

  // ── Point vert (regard) ──
  if (!DOC.getElementById('gaze-dot')) {
    const dot = DOC.createElement('div');
    dot.id = 'gaze-dot';
    dot.style.cssText = `
      position:fixed; width:22px; height:22px; border-radius:50%;
      background:#00e676; border:3px solid #fff;
      box-shadow:0 0 14px #00e676, 0 0 4px rgba(0,0,0,0.4);
      pointer-events:none; z-index:999999;
      transform:translate(-50%,-50%);
      transition:left 0.08s ease-out, top 0.08s ease-out;
      left:50%; top:50%; display:none;
    `;
    DOC.body.appendChild(dot);
  }

  // ── Point rouge de calibration (centre écran) ──
  if (!DOC.getElementById('calib-dot')) {
    const cdot = DOC.createElement('div');
    cdot.id = 'calib-dot';
    cdot.style.cssText = `
      position:fixed; left:50%; top:50%;
      width:28px; height:28px; border-radius:50%;
      transform:translate(-50%,-50%);
      background:radial-gradient(circle, #ff1744 30%, #ff6d00 100%);
      border:3px solid #fff;
      box-shadow:0 0 0 0 rgba(255,23,68,0.7);
      pointer-events:none; z-index:999998;
      display:none;
    `;
    // label
    const label = DOC.createElement('div');
    label.id = 'calib-label';
    label.style.cssText = `
      position:fixed; left:50%; top:calc(50% + 30px);
      transform:translateX(-50%);
      color:#fff; font-size:14px; font-family:sans-serif;
      background:rgba(0,0,0,0.55); padding:4px 12px; border-radius:20px;
      pointer-events:none; z-index:999998; display:none; white-space:nowrap;
    `;
    label.textContent = 'Fixez ce point, puis cliquez sur Calibrer';
    DOC.body.appendChild(cdot);
    DOC.body.appendChild(label);

    // animation pulse
    const style = DOC.createElement('style');
    style.textContent = `
      @keyframes pulse-calib {
        0%   { box-shadow: 0 0 0 0   rgba(255,23,68,0.7); }
        70%  { box-shadow: 0 0 0 18px rgba(255,23,68,0);   }
        100% { box-shadow: 0 0 0 0   rgba(255,23,68,0);    }
      }
      #calib-dot.active { animation: pulse-calib 1.2s ease-out infinite; }
    `;
    DOC.head.appendChild(style);
  }

  WIN.addEventListener('gazeUpdate', function(e) {
    const dot = DOC.getElementById('gaze-dot');
    dot.style.display = 'block';
    dot.style.left = (e.detail.pct_x * 100).toFixed(2) + '%';
    dot.style.top  = (e.detail.pct_y * 100).toFixed(2) + '%';
  });

  WIN.addEventListener('gazeHide', function() {
    const dot = DOC.getElementById('gaze-dot');
    if (dot) dot.style.display = 'none';
  });

  WIN.addEventListener('calibShow', function() {
    const cdot  = DOC.getElementById('calib-dot');
    const label = DOC.getElementById('calib-label');
    if (cdot)  { cdot.style.display = 'block'; cdot.classList.add('active'); }
    if (label) label.style.display = 'block';
  });

  WIN.addEventListener('calibHide', function() {
    const cdot  = DOC.getElementById('calib-dot');
    const label = DOC.getElementById('calib-label');
    if (cdot)  { cdot.style.display = 'none'; cdot.classList.remove('active'); }
    if (label) label.style.display = 'none';
  });
})();
</script>
""", height=0)

# ══════════════════════════════════════════════════════════════════════════════
# ── UI ────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

st.title("👁️ Eye Tracking")

col_ctrl, col_video = st.columns([1, 2])

with col_ctrl:
    st.subheader("Contrôles")

    running = S["running"]
    if not running:
        if st.button("▶ Démarrer la caméra", use_container_width=True):
            with S["lock"]:
                S["running"] = True
                S["calib_ui_state"] = None
            t = threading.Thread(target=eye_tracking_loop, args=(S,), daemon=True)
            t.start()
            S["thread"] = t
            st.rerun()
    else:
        if st.button("⏹ Arrêter", use_container_width=True):
            with S["lock"]:
                S["running"] = False
                S["calib_ui_state"] = None
            st.rerun()

    st.divider()

    calibrated = S["calibrated"]
    calib_ui   = S["calib_ui_state"]

    # Bouton "Préparer calibration" → affiche le point rouge et attend le clic suivant
    if calib_ui is None:
        if st.button("🎯 Préparer la calibration", disabled=not running, use_container_width=True):
            with S["lock"]:
                S["calib_ui_state"] = "waiting"
            st.rerun()

    elif calib_ui == "waiting":
        st.info("Fixez le point rouge au centre de l'écran")
        if st.button("✅ Calibrer maintenant", use_container_width=True):
            with S["lock"]:
                S["do_calibrate"] = True
                # calib_ui_state passera à "done" dans le thread
            st.rerun()
        if st.button("✖ Annuler", use_container_width=True):
            with S["lock"]:
                S["calib_ui_state"] = None
            st.rerun()

    elif calib_ui == "done":
        st.success("✅ Calibration effectuée !")
        if st.button("🔄 Recalibrer", use_container_width=True):
            with S["lock"]:
                S["do_reset"] = True
                S["calibrated"] = False
                S["calib_ui_state"] = None
            st.rerun()

    st.divider()
    st.subheader("Position du regard")
    with S["lock"]:
        sx, sy = S["screen_xy"]
    cx, cy = st.columns(2)
    cx.metric("X", sx)
    cy.metric("Y", sy)

    if MONITOR_WIDTH > 0 and MONITOR_HEIGHT > 0:
        st.caption("Horizontal")
        st.progress(float(max(0, min(1, sx / MONITOR_WIDTH))))
        st.caption("Vertical")
        st.progress(float(max(0, min(1, sy / MONITOR_HEIGHT))))

with col_video:
    st.subheader("Flux caméra")
    placeholder = st.empty()
    with S["lock"]:
        jpeg = S["frame_jpeg"]
    if jpeg:
        placeholder.image(jpeg, channels="BGR", use_container_width=True)
    else:
        placeholder.info("Caméra arrêtée. Appuyer sur 'Démarrer'.")

# ── Dispatch JS à chaque rerun ────────────────────────────────────────────────
with S["lock"]:
    sx, sy       = S["screen_xy"]
    is_running   = S["running"]
    is_calibrated = S["calibrated"]
    calib_ui     = S["calib_ui_state"]

# Point vert du regard
if is_running and is_calibrated and MONITOR_WIDTH > 0:
    pct_x = sx / MONITOR_WIDTH
    pct_y = sy / MONITOR_HEIGHT
    components.html(f"""
<script>
window.parent.dispatchEvent(new CustomEvent('gazeUpdate',{{
  detail:{{pct_x:{pct_x:.4f},pct_y:{pct_y:.4f}}}
}}));
</script>
""", height=0)
else:
    components.html("<script>window.parent.dispatchEvent(new CustomEvent('gazeHide'));</script>", height=0)

# Point rouge de calibration
if is_running and calib_ui == "waiting":
    components.html("<script>window.parent.dispatchEvent(new CustomEvent('calibShow'));</script>", height=0)
else:
    components.html("<script>window.parent.dispatchEvent(new CustomEvent('calibHide'));</script>", height=0)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if S["running"]:
    time.sleep(0.1)
    st.rerun()
