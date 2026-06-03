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

st.set_page_config(page_title="Eye Tracking", page_icon="👁️", layout="wide")

# ── Shared state ──────────────────────────────────────────────────────────────
if "state" not in st.session_state:
    st.session_state.state = {
        "lock": threading.Lock(),
        "running": False,
        "frame_jpeg": None,
        "screen_xy": (0, 0),
        "calibrated": False,
        # calibration 9 points
        "calib_step": -1,          # -1 = idle, 0-8 = point actif, 9 = fitting, 10 = done
        "do_capture": False,       # signal: capturer ce point
        "do_reset": False,
        "calib_samples": [],       # liste de (raw_yaw_deg, raw_pitch_deg, target_sx, target_sy)
        "thread": None,
    }

S = st.session_state.state

MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()

# 9 points en grille 3x3 (fractions 0..1 de l'écran)
CALIB_POINTS_PCT = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]

nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391, 3, 248]

# ══════════════════════════════════════════════════════════════════════════════
# ── Fonctions core ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def compute_scale(points_3d):
    n = len(points_3d)
    total, count = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(points_3d[i] - points_3d[j])
            count += 1
    return total / count if count > 0 else 1.0

def compute_head_frame(face_landmarks, indices, ref_matrix_container, w, h):
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])
    center = np.mean(points_3d, axis=0)
    centered = points_3d - center
    cov = np.cov(centered.T)
    _, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-_)]
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()
    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        for i in range(3):
            if np.dot(R_final[:, i], ref_matrix_container[0][:, i]) < 0:
                R_final[:, i] *= -1
    return center, R_final, points_3d

def gaze_dir_to_angles(gaze_dir):
    """Retourne (yaw_deg, pitch_deg) bruts à partir d'un vecteur de regard normalisé."""
    d = gaze_dir / np.linalg.norm(gaze_dir)
    xz = np.array([d[0], 0, d[2]]); xz /= np.linalg.norm(xz)
    yaw = math.acos(np.clip(np.dot([0,0,-1], xz), -1, 1))
    if d[0] < 0: yaw = -yaw
    yz = np.array([0, d[1], d[2]]); yz /= np.linalg.norm(yz)
    pitch = math.acos(np.clip(np.dot([0,0,-1], yz), -1, 1))
    if d[1] > 0: pitch = -pitch
    return -math.degrees(yaw), math.degrees(pitch)

def fit_linear_map(samples):
    """
    Régression linéaire 2D : (yaw, pitch) → (screen_x, screen_y)
    Retourne (A_x, b_x, A_y, b_y) tels que :
        screen_x = A_x[0]*yaw + A_x[1]*pitch + b_x
        screen_y = A_y[0]*yaw + A_y[1]*pitch + b_y
    """
    yaws   = np.array([s[0] for s in samples])
    pitchs = np.array([s[1] for s in samples])
    sxs    = np.array([s[2] for s in samples], dtype=float)
    sys_   = np.array([s[3] for s in samples], dtype=float)

    X = np.column_stack([yaws, pitchs, np.ones(len(samples))])
    cx, _, _, _ = np.linalg.lstsq(X, sxs, rcond=None)
    cy, _, _, _ = np.linalg.lstsq(X, sys_, rcond=None)
    return cx, cy   # chacun = [coef_yaw, coef_pitch, intercept]

def apply_linear_map(yaw, pitch, cx, cy):
    sx = cx[0]*yaw + cx[1]*pitch + cx[2]
    sy = cy[0]*yaw + cy[1]*pitch + cy[2]
    return (int(np.clip(sx, 10, MONITOR_WIDTH-10)),
            int(np.clip(sy, 10, MONITOR_HEIGHT-10)))

# ══════════════════════════════════════════════════════════════════════════════
# ── Thread OpenCV ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def eye_tracking_loop(shared):
    mp_fm = mp.solutions.face_mesh
    face_mesh = mp_fm.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    R_ref = [None]
    left_locked = right_locked = False
    loff = roff = None
    lcal = rcal = None
    gaze_buf = deque(maxlen=10)
    cx_map = cy_map = None
    base_r = 20

    while True:
        with shared["lock"]:
            if not shared["running"]: break
            do_capture  = shared["do_capture"]
            do_reset    = shared["do_reset"]
            calib_step  = shared["calib_step"]
            if do_capture: shared["do_capture"] = False
            if do_reset:
                shared["do_reset"] = False
                shared["calib_samples"] = []
                shared["calib_step"] = -1
                shared["calibrated"] = False
                left_locked = right_locked = False
                R_ref[0] = None; cx_map = cy_map = None

        ret, frame = cap.read()
        if not ret: time.sleep(0.05); continue

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        screen_xy = (0, 0)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            head_center, R_final, nose_pts = compute_head_frame(frame, lms, nose_indices, R_ref, w, h)
            # Oups: compute_head_frame prend frame en 1er arg mais n'en a pas besoin
            # → on passe w, h séparément (correction ci-dessous)

            li = lms[468]; ri = lms[473]
            il = np.array([li.x*w, li.y*h, li.z*w])
            ir = np.array([ri.x*w, ri.y*h, ri.z*w])

            # ── Premier point capturé : locker les sphères ──
            if do_capture and not (left_locked and right_locked):
                cur = compute_scale(nose_pts)
                cam_l = R_final.T @ np.array([0,0,1])
                loff = R_final.T @ (il - head_center) + base_r * cam_l
                roff = R_final.T @ (ir - head_center) + base_r * cam_l
                lcal = rcal = cur
                left_locked = right_locked = True

            # ── Capturer un sample de calibration ──
            if do_capture and left_locked and right_locked:
                cur = compute_scale(nose_pts)
                sl = head_center + R_final @ (loff * (cur / lcal))
                sr = head_center + R_final @ (roff * (cur / rcal))
                dl = il - sl; dl /= np.linalg.norm(dl)
                dr = ir - sr; dr /= np.linalg.norm(dr)
                raw_dir = (dl + dr) / 2; raw_dir /= np.linalg.norm(raw_dir)
                yaw, pitch = gaze_dir_to_angles(raw_dir)

                with shared["lock"]:
                    step = shared["calib_step"]
                    if 0 <= step <= 8:
                        px, py = CALIB_POINTS_PCT[step]
                        tx = int(px * MONITOR_WIDTH)
                        ty = int(py * MONITOR_HEIGHT)
                        shared["calib_samples"].append((yaw, pitch, tx, ty))
                        # Si dernier point → fitter
                        if len(shared["calib_samples"]) >= 9:
                            shared["calib_step"] = 9   # fitting
                            samples = shared["calib_samples"]
                            cx_map, cy_map = fit_linear_map(samples)
                            shared["calibrated"] = True
                            shared["calib_step"] = 10  # done
                        else:
                            shared["calib_step"] = step + 1

            # ── Calculer position regard si calibré ──
            if left_locked and right_locked and cx_map is not None:
                cur = compute_scale(nose_pts)
                sl = head_center + R_final @ (loff * (cur/lcal))
                sr = head_center + R_final @ (roff * (cur/rcal))
                rl = int(base_r*(cur/lcal)); rr = int(base_r*(cur/rcal))
                dl = il - sl; dl /= np.linalg.norm(dl)
                dr = ir - sr; dr /= np.linalg.norm(dr)
                raw = (dl+dr)/2; raw /= np.linalg.norm(raw)
                gaze_buf.append(raw)
                avg = np.mean(gaze_buf, axis=0); avg /= np.linalg.norm(avg)
                yaw, pitch = gaze_dir_to_angles(avg)
                screen_xy = apply_linear_map(yaw, pitch, cx_map, cy_map)

                cv2.circle(frame, (int(sl[0]),int(sl[1])), rl, (255,255,25), 2)
                cv2.circle(frame, (int(sr[0]),int(sr[1])), rr, (25,255,255), 2)
                origin = (sl+sr)/2
                cv2.line(frame, tuple(int(v) for v in origin[:2]),
                         tuple(int(v) for v in (origin+avg*200)[:2]), (255,255,10), 2)
                cv2.putText(frame, f"Gaze ({screen_xy[0]},{screen_xy[1]})",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            elif left_locked and right_locked:
                cv2.putText(frame, "Calibration en cours...", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
            else:
                cv2.circle(frame, (int(li.x*w),int(li.y*h)), 8, (255,50,50), 2)
                cv2.circle(frame, (int(ri.x*w),int(ri.y*h)), 8, (50,255,50), 2)
                cv2.putText(frame, "Lancer la calibration", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,255), 2)

            for lm in lms:
                cv2.circle(frame, (int(lm.x*w),int(lm.y*h)), 1, (200,200,200), -1)

            with shared["lock"]:
                shared["screen_xy"] = screen_xy

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with shared["lock"]:
            shared["frame_jpeg"] = buf.tobytes()
        time.sleep(0.01)

    cap.release(); face_mesh.close()
    with shared["lock"]:
        shared["running"] = False; shared["frame_jpeg"] = None

# Correction : compute_head_frame ne prend pas frame, retirer ce param
def compute_head_frame(face_landmarks, indices, ref_matrix_container, w, h):
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
        for i in range(3):
            if np.dot(R_final[:, i], ref_matrix_container[0][:, i]) < 0:
                R_final[:, i] *= -1
    return center, R_final, points_3d

# Redéfinir eye_tracking_loop avec la bonne signature
def eye_tracking_loop(shared):
    mp_fm = mp.solutions.face_mesh
    face_mesh = mp_fm.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    R_ref = [None]
    left_locked = right_locked = False
    loff = roff = None
    lcal = rcal = None
    gaze_buf = deque(maxlen=10)
    cx_map = cy_map = None
    base_r = 20

    while True:
        with shared["lock"]:
            if not shared["running"]: break
            do_capture = shared["do_capture"]
            do_reset   = shared["do_reset"]
            if do_capture: shared["do_capture"] = False
            if do_reset:
                shared["do_reset"] = False
                shared["calib_samples"] = []
                shared["calib_step"] = -1
                shared["calibrated"] = False
                left_locked = right_locked = False
                R_ref[0] = None; cx_map = cy_map = None

        ret, frame = cap.read()
        if not ret: time.sleep(0.05); continue

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        screen_xy = (0, 0)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            head_center, R_final, nose_pts = compute_head_frame(lms, nose_indices, R_ref, w, h)

            li = lms[468]; ri = lms[473]
            il = np.array([li.x*w, li.y*h, li.z*w])
            ir = np.array([ri.x*w, ri.y*h, ri.z*w])

            # ── Premier point : locker les sphères ──
            if do_capture and not (left_locked and right_locked):
                cur = compute_scale(nose_pts)
                cam_l = R_final.T @ np.array([0,0,1])
                loff = R_final.T @ (il - head_center) + base_r * cam_l
                roff = R_final.T @ (ir - head_center) + base_r * cam_l
                lcal = rcal = cur
                left_locked = right_locked = True

            # ── Capturer sample ──
            if do_capture and left_locked:
                cur = compute_scale(nose_pts)
                sl = head_center + R_final @ (loff * (cur/lcal))
                sr = head_center + R_final @ (roff * (cur/rcal))
                dl = il - sl; dl /= np.linalg.norm(dl)
                dr = ir - sr; dr /= np.linalg.norm(dr)
                raw_dir = (dl+dr)/2; raw_dir /= np.linalg.norm(raw_dir)
                yaw_d, pitch_d = gaze_dir_to_angles(raw_dir)

                with shared["lock"]:
                    step = shared["calib_step"]
                    if 0 <= step <= 8:
                        px, py = CALIB_POINTS_PCT[step]
                        tx = int(px * MONITOR_WIDTH)
                        ty = int(py * MONITOR_HEIGHT)
                        shared["calib_samples"].append((yaw_d, pitch_d, tx, ty))
                        n_done = len(shared["calib_samples"])
                        if n_done >= 9:
                            samples = list(shared["calib_samples"])
                            cx_map, cy_map = fit_linear_map(samples)
                            shared["calibrated"] = True
                            shared["calib_step"] = 10
                        else:
                            shared["calib_step"] = step + 1

            # ── Afficher regard calibré ──
            if left_locked and right_locked and cx_map is not None:
                cur = compute_scale(nose_pts)
                sl = head_center + R_final @ (loff*(cur/lcal))
                sr = head_center + R_final @ (roff*(cur/rcal))
                rl = int(base_r*(cur/lcal)); rr = int(base_r*(cur/rcal))
                dl = il-sl; dl /= np.linalg.norm(dl)
                dr = ir-sr; dr /= np.linalg.norm(dr)
                raw = (dl+dr)/2; raw /= np.linalg.norm(raw)
                gaze_buf.append(raw)
                avg = np.mean(gaze_buf, axis=0); avg /= np.linalg.norm(avg)
                yaw_d, pitch_d = gaze_dir_to_angles(avg)
                screen_xy = apply_linear_map(yaw_d, pitch_d, cx_map, cy_map)
                cv2.circle(frame, (int(sl[0]),int(sl[1])), rl, (255,255,25), 2)
                cv2.circle(frame, (int(sr[0]),int(sr[1])), rr, (25,255,255), 2)
                origin = (sl+sr)/2
                cv2.line(frame, tuple(int(v) for v in origin[:2]),
                         tuple(int(v) for v in (origin+avg*200)[:2]), (255,255,10), 2)
                cv2.putText(frame, f"Gaze ({screen_xy[0]},{screen_xy[1]})",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, "CALIBRATED 9pts", (10,55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
            elif left_locked:
                cv2.putText(frame, "Calibration en cours...",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
            else:
                cv2.circle(frame, (int(li.x*w),int(li.y*h)), 8, (255,50,50), 2)
                cv2.circle(frame, (int(ri.x*w),int(ri.y*h)), 8, (50,255,50), 2)
                cv2.putText(frame, "Lancer la calibration",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,255), 2)

            for lm in lms:
                cv2.circle(frame, (int(lm.x*w),int(lm.y*h)), 1, (200,200,200), -1)

            with shared["lock"]:
                shared["screen_xy"] = screen_xy

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with shared["lock"]:
            shared["frame_jpeg"] = buf.tobytes()
        time.sleep(0.01)

    cap.release(); face_mesh.close()
    with shared["lock"]:
        shared["running"] = False; shared["frame_jpeg"] = None

# ══════════════════════════════════════════════════════════════════════════════
# ── Injection JS : point vert + point rouge de calibration positionnable ──────
# ══════════════════════════════════════════════════════════════════════════════

components.html("""
<script>
(function(){
  const W = window.parent, D = W.document;

  // Point vert (regard)
  if (!D.getElementById('gaze-dot')) {
    const d = D.createElement('div');
    d.id = 'gaze-dot';
    d.style.cssText = `position:fixed;width:22px;height:22px;border-radius:50%;
      background:#00e676;border:3px solid #fff;
      box-shadow:0 0 14px #00e676,0 0 4px rgba(0,0,0,.4);
      pointer-events:none;z-index:999999;transform:translate(-50%,-50%);
      transition:left .08s ease-out,top .08s ease-out;display:none;`;
    D.body.appendChild(d);
  }

  // Point rouge calibration (position variable)
  if (!D.getElementById('calib-dot')) {
    const c = D.createElement('div');
    c.id = 'calib-dot';
    c.style.cssText = `position:fixed;width:26px;height:26px;border-radius:50%;
      background:#ff1744;border:3px solid #fff;
      pointer-events:none;z-index:999998;transform:translate(-50%,-50%);
      display:none;`;

    const lbl = D.createElement('div');
    lbl.id = 'calib-label';
    lbl.style.cssText = `position:fixed;transform:translateX(-50%);
      color:#fff;font-size:13px;font-family:sans-serif;
      background:rgba(0,0,0,.6);padding:3px 10px;border-radius:16px;
      pointer-events:none;z-index:999998;display:none;white-space:nowrap;`;

    const num = D.createElement('div');
    num.id = 'calib-num';
    num.style.cssText = `position:fixed;color:#fff;font-size:11px;font-family:sans-serif;
      font-weight:bold;pointer-events:none;z-index:1000000;
      transform:translate(-50%,-50%);display:none;`;

    const sty = D.createElement('style');
    sty.textContent = `
      @keyframes pc{0%{box-shadow:0 0 0 0 rgba(255,23,68,.7)}
        70%{box-shadow:0 0 0 16px rgba(255,23,68,0)}
        100%{box-shadow:0 0 0 0 rgba(255,23,68,0)}}
      #calib-dot.active{animation:pc 1s ease-out infinite}`;
    D.head.appendChild(sty);
    D.body.appendChild(c);
    D.body.appendChild(lbl);
    D.body.appendChild(num);
  }

  W.addEventListener('gazeUpdate', e => {
    const d = D.getElementById('gaze-dot');
    d.style.display = 'block';
    d.style.left = (e.detail.px * 100).toFixed(2) + '%';
    d.style.top  = (e.detail.py * 100).toFixed(2) + '%';
  });
  W.addEventListener('gazeHide', () => {
    const d = D.getElementById('gaze-dot');
    if(d) d.style.display='none';
  });

  W.addEventListener('calibPoint', e => {
    const c   = D.getElementById('calib-dot');
    const lbl = D.getElementById('calib-label');
    const num = D.getElementById('calib-num');
    const vw  = W.innerWidth, vh = W.innerHeight;
    const px  = e.detail.px * 100, py = e.detail.py * 100;
    const step = e.detail.step, total = e.detail.total;

    c.style.left = px + '%'; c.style.top = py + '%';
    c.style.display = 'block'; c.classList.add('active');

    // label en dessous sauf si en bas (>85%) → au-dessus
    const lblOffset = py > 85 ? '-42px' : '24px';
    lbl.style.left = px + '%';
    lbl.style.top  = `calc(${py}% + ${lblOffset})`;
    lbl.textContent = `Point ${step}/${total} — Fixez et cliquez`;
    lbl.style.display = 'block';

    num.style.left = px + '%'; num.style.top = py + '%';
    num.textContent = step;
    num.style.display = 'block';
  });

  W.addEventListener('calibHide', () => {
    ['calib-dot','calib-label','calib-num'].forEach(id => {
      const el = D.getElementById(id);
      if(el){ el.style.display='none'; if(id==='calib-dot') el.classList.remove('active'); }
    });
  });
})();
</script>
""", height=0)

# ══════════════════════════════════════════════════════════════════════════════
# ── UI ────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

st.title("👁️ Eye Tracking — Calibration 9 points")

col_ctrl, col_video = st.columns([1, 2])

with col_ctrl:
    st.subheader("Contrôles")

    running = S["running"]
    if not running:
        if st.button("▶ Démarrer la caméra", use_container_width=True):
            with S["lock"]:
                S["running"] = True
                S["calib_step"] = -1
                S["calib_samples"] = []
                S["calibrated"] = False
            t = threading.Thread(target=eye_tracking_loop, args=(S,), daemon=True)
            t.start(); S["thread"] = t
            st.rerun()
    else:
        if st.button("⏹ Arrêter", use_container_width=True):
            with S["lock"]:
                S["running"] = False
            st.rerun()

    st.divider()

    with S["lock"]:
        calib_step = S["calib_step"]
        calibrated = S["calibrated"]
        n_samples  = len(S["calib_samples"])

    # ── Machine à états calibration ──
    if calib_step == -1:
        # Pas encore démarrée
        if st.button("🎯 Lancer la calibration 9 pts", disabled=not running, use_container_width=True):
            with S["lock"]:
                S["calib_step"] = 0
                S["calib_samples"] = []
                S["calibrated"] = False
            st.rerun()

    elif 0 <= calib_step <= 8:
        # En cours
        px, py = CALIB_POINTS_PCT[calib_step]
        st.info(f"**Point {calib_step + 1} / 9**  \nFixez le point rouge sur l'écran")
        st.progress((calib_step) / 9, text=f"{calib_step}/9 points capturés")

        if st.button(f"✅ Capturer le point {calib_step + 1}", use_container_width=True, type="primary"):
            with S["lock"]:
                S["do_capture"] = True
            time.sleep(0.15)   # laisser le thread traiter
            st.rerun()

        if st.button("✖ Annuler la calibration", use_container_width=True):
            with S["lock"]:
                S["calib_step"] = -1
                S["calib_samples"] = []
            st.rerun()

    elif calib_step == 10:
        # Terminée
        st.success("✅ Calibration 9 points terminée !")
        st.progress(1.0, text="9/9 points capturés")
        if st.button("🔄 Recalibrer", use_container_width=True):
            with S["lock"]:
                S["do_reset"] = True
            st.rerun()

    st.divider()
    st.subheader("Position du regard")
    with S["lock"]:
        sx, sy = S["screen_xy"]
    cx2, cy2 = st.columns(2)
    cx2.metric("X", sx)
    cy2.metric("Y", sy)
    if MONITOR_WIDTH > 0:
        st.caption("Horizontal")
        st.progress(float(max(0, min(1, sx / MONITOR_WIDTH))))
        st.caption("Vertical")
        st.progress(float(max(0, min(1, sy / MONITOR_HEIGHT))))

with col_video:
    st.subheader("Flux caméra")
    ph = st.empty()
    with S["lock"]:
        jpeg = S["frame_jpeg"]
    if jpeg:
        ph.image(jpeg, channels="BGR", use_container_width=True)
    else:
        ph.info("Caméra arrêtée.")

# ── Dispatch JS ───────────────────────────────────────────────────────────────
with S["lock"]:
    sx, sy      = S["screen_xy"]
    is_running  = S["running"]
    is_cal      = S["calibrated"]
    step        = S["calib_step"]

# Point vert
if is_running and is_cal and MONITOR_WIDTH > 0:
    px = sx / MONITOR_WIDTH; py = sy / MONITOR_HEIGHT
    components.html(f"""<script>
window.parent.dispatchEvent(new CustomEvent('gazeUpdate',{{
  detail:{{px:{px:.4f},py:{py:.4f}}}
}}));</script>""", height=0)
else:
    components.html("<script>window.parent.dispatchEvent(new CustomEvent('gazeHide'));</script>", height=0)

# Point rouge calibration
if is_running and 0 <= step <= 8:
    ppx, ppy = CALIB_POINTS_PCT[step]
    components.html(f"""<script>
window.parent.dispatchEvent(new CustomEvent('calibPoint',{{
  detail:{{px:{ppx},py:{ppy},step:{step+1},total:9}}
}}));</script>""", height=0)
else:
    components.html("<script>window.parent.dispatchEvent(new CustomEvent('calibHide'));</script>", height=0)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if S["running"]:
    time.sleep(0.1)
    st.rerun()
