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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eye Tracking",
    page_icon="👁️",
    layout="wide",
)

# ── Shared state (thread-safe) ────────────────────────────────────────────────
# Toutes les données échangées entre le thread OpenCV et Streamlit passent ici.
if "state" not in st.session_state:
    st.session_state.state = {
        "lock": threading.Lock(),
        "running": False,
        "frame_jpeg": None,       # bytes JPEG pour st.image()
        "debug_jpeg": None,       # vue debug (optionnelle)
        "screen_xy": (0, 0),
        "calibrated": False,
        "mouse_enabled": False,
        "do_calibrate": False,    # signal: appuyer sur 'c'
        "do_reset": False,        # signal: recalibrer
        "thread": None,
    }

S = st.session_state.state  # raccourci local

# ══════════════════════════════════════════════════════════════════════════════
# ── Toutes les fonctions du script original, légèrement adaptées ─────────────
# ══════════════════════════════════════════════════════════════════════════════

MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()

nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

def compute_scale(points_3d):
    n = len(points_3d)
    total, count = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(points_3d[i] - points_3d[j])
            count += 1
    return total / count if count > 0 else 1.0

def compute_and_draw_coordinate_box(frame, face_landmarks, indices, ref_matrix_container, w, h, color=(0, 255, 0), size=80):
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
    yaw_deg = -math.degrees(yaw_rad)
    pitch_deg = math.degrees(pitch_rad)
    yaw_deg += offset_yaw
    pitch_deg += offset_pitch
    yawDeg, pitchDeg = 15, 5
    sx = int(((yaw_deg + yawDeg) / (2 * yawDeg)) * MONITOR_WIDTH)
    sy = int(((pitchDeg - pitch_deg) / (2 * pitchDeg)) * MONITOR_HEIGHT)
    sx = max(10, min(sx, MONITOR_WIDTH - 10))
    sy = max(10, min(sy, MONITOR_HEIGHT - 10))
    return sx, sy

# ══════════════════════════════════════════════════════════════════════════════
# ── Thread principal d'eye tracking ──────────────────────────────────────────
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
            do_calib  = shared["do_calibrate"]
            do_reset  = shared["do_reset"]
            mouse_on  = shared["mouse_enabled"]
            if do_calib:  shared["do_calibrate"] = False
            if do_reset:  shared["do_reset"] = False; left_sphere_locked = False; right_sphere_locked = False; R_ref_nose[0] = None

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

            # ── Calibration (touche C / bouton Streamlit) ──
            if do_calib and not (left_sphere_locked and right_sphere_locked):
                cur_scale = compute_scale(nose_pts)
                cam_dir_local = R_final.T @ np.array([0, 0, 1])
                left_sphere_local_offset = R_final.T @ (iris3d_l - head_center) + base_radius * cam_dir_local
                right_sphere_local_offset = R_final.T @ (iris3d_r - head_center) + base_radius * cam_dir_local
                left_cal_scale = right_cal_scale = cur_scale
                left_sphere_locked = right_sphere_locked = True

                # Calibrage centre-écran
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

            screen_xy = (0, 0)
            if left_sphere_locked and right_sphere_locked:
                cur = compute_scale(nose_pts)
                sr_l = cur / left_cal_scale  if left_cal_scale  else 1.0
                sr_r = cur / right_cal_scale if right_cal_scale else 1.0
                sphere_l = head_center + R_final @ (left_sphere_local_offset  * sr_l)
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

                if mouse_on:
                    try: pyautogui.moveTo(sx, sy)
                    except: pass

                # Dessiner les sphères et le rayon combiné
                cv2.circle(frame, (int(sphere_l[0]), int(sphere_l[1])), rad_l, (255, 255, 25), 2)
                cv2.circle(frame, (int(sphere_r[0]), int(sphere_r[1])), rad_r, (25, 255, 255), 2)
                origin = (sphere_l + sphere_r) / 2
                end = origin + avg_dir * 200
                cv2.line(frame, tuple(int(v) for v in origin[:2]),
                         tuple(int(v) for v in end[:2]), (255, 255, 10), 2)

                # HUD
                cv2.putText(frame, f"Gaze: ({sx}, {sy})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "CALIBRATED" if left_sphere_locked else "",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                # Avant calibration : afficher les iris
                cv2.circle(frame, (int(left_iris.x*w), int(left_iris.y*h)), 8, (255, 50, 50), 2)
                cv2.circle(frame, (int(right_iris.x*w), int(right_iris.y*h)), 8, (50, 255, 50), 2)
                cv2.putText(frame, "Appuyer sur 'Calibrer'", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

            # Landmarks
            for lm in lms:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 1, (200, 200, 200), -1)

            with shared["lock"]:
                shared["screen_xy"] = screen_xy

        # Encoder le frame en JPEG pour Streamlit
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
# ── Interface Streamlit ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

st.title("👁️ Eye Tracking – Contrôle")

# ── Point vert flottant sur toute la page ─────────────────────────────────────
# Injecté une seule fois via un composant HTML placé dans le <head> de la page.
# Le JS écoute un événement custom "gazeUpdate" pour déplacer le point.
components.html("""
<script>
(function() {
  // Crée le point vert s'il n'existe pas déjà dans la page parente
  const WIN = window.parent;
  const DOC = WIN.document;
  if (DOC.getElementById('gaze-dot')) return;

  const dot = DOC.createElement('div');
  dot.id = 'gaze-dot';
  dot.style.cssText = `
    position: fixed;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #00e676;
    border: 3px solid #fff;
    box-shadow: 0 0 12px #00e676, 0 0 4px rgba(0,0,0,0.4);
    pointer-events: none;
    z-index: 999999;
    transform: translate(-50%, -50%);
    transition: left 0.08s ease-out, top 0.08s ease-out;
    left: 50%;
    top: 50%;
    display: none;
  `;
  DOC.body.appendChild(dot);

  WIN.addEventListener('gazeUpdate', function(e) {
    const pct_x = e.detail.pct_x;
    const pct_y = e.detail.pct_y;
    dot.style.display = 'block';
    dot.style.left = (pct_x * 100).toFixed(2) + '%';
    dot.style.top  = (pct_y * 100).toFixed(2) + '%';
  });

  WIN.addEventListener('gazeHide', function() {
    dot.style.display = 'none';
  });
})();
</script>
""", height=0)

# ── Colonne de contrôle (gauche) + flux vidéo (droite) ───────────────────────
col_ctrl, col_video = st.columns([1, 2])

with col_ctrl:
    st.subheader("Contrôles")

    # Démarrer / Arrêter
    running = S["running"]
    if not running:
        if st.button("▶ Démarrer la caméra", use_container_width=True):
            with S["lock"]:
                S["running"] = True
            t = threading.Thread(target=eye_tracking_loop, args=(S,), daemon=True)
            t.start()
            S["thread"] = t
            st.rerun()
    else:
        if st.button("⏹ Arrêter", use_container_width=True):
            with S["lock"]:
                S["running"] = False
            st.rerun()

    st.divider()

    # Calibration
    calibrated = S["calibrated"]
    if st.button("🎯 Calibrer (regarder le centre de l'écran)",
                 disabled=not running, use_container_width=True):
        with S["lock"]:
            S["do_calibrate"] = True

    if st.button("🔄 Recalibrer (reset sphères)", disabled=not running, use_container_width=True):
        with S["lock"]:
            S["do_reset"] = True
            S["calibrated"] = False

    st.caption("✅ Calibré" if calibrated else "⚠️ Non calibré")
    st.divider()

    # Contrôle souris
    mouse_on = st.toggle("🖱️ Contrôle souris", value=S["mouse_enabled"], disabled=not calibrated)
    with S["lock"]:
        S["mouse_enabled"] = mouse_on

    st.divider()
    st.subheader("Position du regard")
    with S["lock"]:
        sx, sy = S["screen_xy"]
    col_x, col_y = st.columns(2)
    col_x.metric("X", sx)
    col_y.metric("Y", sy)

    if MONITOR_WIDTH > 0 and MONITOR_HEIGHT > 0:
        prog_x = sx / MONITOR_WIDTH
        prog_y = sy / MONITOR_HEIGHT
        st.caption("Horizontal")
        st.progress(float(prog_x))
        st.caption("Vertical")
        st.progress(float(prog_y))

# ── Flux vidéo ────────────────────────────────────────────────────────────────
with col_video:
    st.subheader("Flux caméra")
    video_placeholder = st.empty()

    with S["lock"]:
        jpeg = S["frame_jpeg"]

    if jpeg:
        video_placeholder.image(jpeg, channels="BGR", use_container_width=True)
    else:
        video_placeholder.info("Caméra arrêtée. Appuyer sur 'Démarrer'.")

# ── Mise à jour du point vert via un événement JS ─────────────────────────────
# À chaque rerun on injecte les coordonnées courantes dans la page parente.
with S["lock"]:
    sx, sy = S["screen_xy"]
    is_running = S["running"]
    is_calibrated = S["calibrated"]

if is_running and is_calibrated and MONITOR_WIDTH > 0 and MONITOR_HEIGHT > 0:
    pct_x = sx / MONITOR_WIDTH
    pct_y = sy / MONITOR_HEIGHT
    components.html(f"""
<script>
window.parent.dispatchEvent(new CustomEvent('gazeUpdate', {{
  detail: {{ pct_x: {pct_x:.4f}, pct_y: {pct_y:.4f} }}
}}));
</script>
""", height=0)
else:
    components.html("""
<script>
window.parent.dispatchEvent(new CustomEvent('gazeHide'));
</script>
""", height=0)

# ── Auto-refresh toutes les 100 ms si la caméra tourne ───────────────────────
if S["running"]:
    time.sleep(0.1)
    st.rerun()
