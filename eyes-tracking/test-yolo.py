"""
╔══════════════════════════════════════════════════════════════════╗
║           EYE GAZE TRACKING — YOLO + MediaPipe                  ║
║                                                                  ║
║  Dépendances :                                                   ║
║    pip install ultralytics opencv-python mediapipe numpy         ║
║                                                                  ║
║  Modèle YOLO visage :                                            ║
║    Télécharger yolo26n-face.pt depuis :                         ║
║    https://github.com/akanametov/yolo-face/releases             ║
║    → placer dans le même dossier que ce script                  ║
║                                                                  ║
║  Contrôles :                                                     ║
║    Q / Échap  → Quitter                                         ║
║    C          → Lancer la calibration (9 points)                ║
║    R          → Réinitialiser la calibration                    ║
║    S          → Sauvegarder une capture d'écran                 ║
║    H          → Afficher/masquer le HUD                         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────
#  Tentative d'import YOLO (optionnel)
# ─────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics non installé — YOLO désactivé, MediaPipe seul utilisé.")

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

class Config:
    # Modèle YOLO
    YOLO_MODEL_PATH   = "yolo26n-face.pt"
    YOLO_CONF         = 0.5           # seuil de confiance détection visage
    USE_YOLO          = True          # False → MediaPipe seul

    # Webcam
    CAMERA_INDEX      = 0
    FRAME_WIDTH       = 1280
    FRAME_HEIGHT      = 720
    FPS_TARGET        = 30

    # Gaze
    SMOOTH_WINDOW     = 7            # lissage temporel (frames)
    GAZE_THRESHOLD_L  = 0.38         # seuil gauche
    GAZE_THRESHOLD_R  = 0.62         # seuil droite
    GAZE_THRESHOLD_U  = 0.35         # seuil haut
    GAZE_THRESHOLD_D  = 0.65         # seuil bas

    # Calibration
    CALIB_POINTS      = 9            # nb points de calibration
    CALIB_DWELL_MS    = 1500         # durée de fixation par point (ms)

    # Affichage
    SHOW_LANDMARKS    = True
    SHOW_BOUNDING_BOX = True
    SHOW_GAZE_VECTOR  = True
    SHOW_HEATMAP      = False        # expérimental


# ═══════════════════════════════════════════════════════════════════
#  INDICES MEDIAPIPE
# ═══════════════════════════════════════════════════════════════════

LEFT_IRIS         = [474, 475, 476, 477]
RIGHT_IRIS        = [469, 470, 471, 472]
LEFT_EYE_CORNERS  = [33,  133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_EYE_TOP_BOT  = [159, 145]
RIGHT_EYE_TOP_BOT = [386, 374]

# Contour complet des yeux pour dessin
LEFT_EYE_CONTOUR  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE_CONTOUR = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]


# ═══════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ═══════════════════════════════════════════════════════════════════

def lm_to_px(landmark, w, h):
    """Convertit un landmark normalisé en pixels."""
    return int(landmark.x * w), int(landmark.y * h)


def iris_center(landmarks, indices, w, h):
    """Calcule le centre d'un iris à partir des landmarks."""
    pts = [lm_to_px(landmarks[i], w, h) for i in indices]
    cx  = sum(p[0] for p in pts) // len(pts)
    cy  = sum(p[1] for p in pts) // len(pts)
    return cx, cy


def eye_aspect_ratio(landmarks, top_bot, corners, w, h):
    """EAR — ratio d'ouverture de l'œil (1 = grand ouvert, ~0.2 = fermé)."""
    top    = lm_to_px(landmarks[top_bot[0]], w, h)
    bot    = lm_to_px(landmarks[top_bot[1]], w, h)
    left   = lm_to_px(landmarks[corners[0]], w, h)
    right  = lm_to_px(landmarks[corners[1]], w, h)
    height = np.linalg.norm(np.array(top) - np.array(bot))
    width  = np.linalg.norm(np.array(left) - np.array(right))
    return height / (width + 1e-6)


def gaze_ratio_h(landmarks, corners, iris_cx, w, h):
    """Ratio horizontal du regard : 0 = extrême gauche, 1 = extrême droite."""
    left_pt  = lm_to_px(landmarks[corners[0]], w, h)
    right_pt = lm_to_px(landmarks[corners[1]], w, h)
    eye_w    = right_pt[0] - left_pt[0]
    if eye_w < 1:
        return 0.5
    return np.clip((iris_cx - left_pt[0]) / eye_w, 0, 1)


def gaze_ratio_v(landmarks, top_bot, iris_cy, w, h):
    """Ratio vertical du regard : 0 = extrême haut, 1 = extrême bas."""
    top_pt = lm_to_px(landmarks[top_bot[0]], w, h)
    bot_pt = lm_to_px(landmarks[top_bot[1]], w, h)
    eye_h  = bot_pt[1] - top_pt[1]
    if eye_h < 1:
        return 0.5
    return np.clip((iris_cy - top_pt[1]) / eye_h, 0, 1)


# ═══════════════════════════════════════════════════════════════════
#  CALIBRATION
# ═══════════════════════════════════════════════════════════════════

class Calibrator:
    """
    Calibration en 9 points.
    Collecte les ratios gaze pendant que l'utilisateur fixe chaque point.
    Calcule ensuite une homographie (ratio_h, ratio_v) → (px_x, px_y).
    """

    def __init__(self, screen_w, screen_h, n_points=9):
        self.screen_w  = screen_w
        self.screen_h  = screen_h
        self.n_points  = n_points
        self.calibrated = False
        self.homography = None
        self._build_target_points()
        self.reset()

    def _build_target_points(self):
        margin = 0.1
        cols   = int(np.sqrt(self.n_points))
        rows   = self.n_points // cols
        xs     = np.linspace(margin, 1 - margin, cols)
        ys     = np.linspace(margin, 1 - margin, rows)
        self.targets = [(x, y) for y in ys for x in xs][:self.n_points]

    def reset(self):
        self.current_idx   = 0
        self.current_data  = []
        self.all_src       = []
        self.all_dst       = []
        self.dwell_start   = None
        self.calibrated    = False
        self.homography    = None

    @property
    def active(self):
        return self.current_idx < self.n_points and not self.calibrated

    def current_target_px(self):
        tx, ty = self.targets[self.current_idx]
        return int(tx * self.screen_w), int(ty * self.screen_h)

    def feed(self, ratio_h, ratio_v):
        """Appeler à chaque frame pendant la calibration. Retourne True si terminée."""
        now = time.time()
        if self.dwell_start is None:
            self.dwell_start   = now
            self.current_data  = []

        elapsed = (now - self.dwell_start) * 1000
        self.current_data.append((ratio_h, ratio_v))

        if elapsed >= Config.CALIB_DWELL_MS:
            # Moyenne des mesures collectées
            avg_h = float(np.median([d[0] for d in self.current_data]))
            avg_v = float(np.median([d[1] for d in self.current_data]))
            tx, ty = self.targets[self.current_idx]
            self.all_src.append([avg_h, avg_v])
            self.all_dst.append([tx,    ty   ])
            self.current_idx += 1
            self.dwell_start  = None
            self.current_data = []

            if self.current_idx >= self.n_points:
                self._compute_homography()
                return True
        return False

    def _compute_homography(self):
        src = np.array(self.all_src, dtype=np.float32)
        dst = np.array(self.all_dst, dtype=np.float32)
        self.homography, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        self.calibrated = True

    def map_gaze(self, ratio_h, ratio_v):
        """Convertit un ratio de regard en coordonnées normalisées écran."""
        if self.homography is None:
            return ratio_h, ratio_v
        pt  = np.array([[[ratio_h, ratio_v]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.homography)[0][0]
        return float(np.clip(out[0], 0, 1)), float(np.clip(out[1], 0, 1))

    def progress(self):
        elapsed = 0
        if self.dwell_start is not None:
            elapsed = min((time.time() - self.dwell_start) * 1000, Config.CALIB_DWELL_MS)
        return elapsed / Config.CALIB_DWELL_MS


# ═══════════════════════════════════════════════════════════════════
#  AFFICHAGE / OVERLAY
# ═══════════════════════════════════════════════════════════════════

# Palette de couleurs
C_GREEN   = (0,   220,  80)
C_BLUE    = (80,  180, 255)
C_RED     = (0,   60,  220)
C_YELLOW  = (0,   220, 220)
C_WHITE   = (240, 240, 240)
C_BLACK   = (10,   10,  10)
C_ORANGE  = (0,   150, 255)
C_PURPLE  = (200,  80, 200)
C_GRAY    = (120, 120, 120)


def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.55):
    """Rectangle arrondi semi-transparent."""
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_eye_contour(frame, landmarks, indices, w, h, color, thickness=1):
    pts = np.array([lm_to_px(landmarks[i], w, h) for i in indices], np.int32)
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)


def draw_iris(frame, center, radius=6, color=C_RED):
    cv2.circle(frame,  center, radius,     color,  -1,  cv2.LINE_AA)
    cv2.circle(frame,  center, radius + 2, C_WHITE, 1,  cv2.LINE_AA)


def draw_gaze_arrow(frame, iris_center, ratio_h, ratio_v, length=50):
    dx = int((ratio_h - 0.5) * 2 * length)
    dy = int((ratio_v - 0.5) * 2 * length)
    end = (iris_center[0] + dx, iris_center[1] + dy)
    cv2.arrowedLine(frame, iris_center, end, C_YELLOW, 2, cv2.LINE_AA, tipLength=0.35)


def draw_hud(frame, fps, direction_h, direction_v, ear_l, ear_r,
             ratio_h, ratio_v, calibrated, blink_count, show):
    if not show:
        return
    h_frame, w_frame = frame.shape[:2]
    pad = 16
    box_w, box_h = 280, 230
    x1, y1 = pad, pad
    x2, y2 = x1 + box_w, y1 + box_h
    draw_rounded_rect(frame, (x1, y1), (x2, y2), C_BLACK, radius=14, alpha=0.6)

    def txt(text, row, color=C_WHITE, scale=0.52, bold=False):
        thickness = 2 if bold else 1
        cv2.putText(frame, text, (x1 + 12, y1 + 24 + row * 26),
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

    txt("EYE GAZE TRACKER", 0, C_GREEN, scale=0.58, bold=True)
    cv2.line(frame, (x1+10, y1+34), (x2-10, y1+34), C_GRAY, 1)

    txt(f"FPS        : {fps:5.1f}", 2)
    txt(f"Direction  : {direction_h} / {direction_v}", 3,
        color=C_YELLOW if direction_h != "CENTER" else C_GREEN)
    txt(f"Ratio H/V  : {ratio_h:.3f} / {ratio_v:.3f}", 4)
    txt(f"EAR  L/R   : {ear_l:.3f} / {ear_r:.3f}", 5)
    txt(f"Clignements: {blink_count}", 6, C_BLUE)
    calib_txt = "OUI ✓" if calibrated else "NON"
    calib_col = C_GREEN if calibrated else C_ORANGE
    txt(f"Calibré    : {calib_txt}", 7, calib_col)

    cv2.line(frame, (x1+10, y1+34+7*26+8), (x2-10, y1+34+7*26+8), C_GRAY, 1)
    txt("[C] Calibrer  [R] Reset  [S] Save  [H] HUD", 8, C_GRAY, scale=0.42)


def draw_direction_indicator(frame, direction_h, direction_v):
    """Mini boussole de direction en bas à droite."""
    h_frame, w_frame = frame.shape[:2]
    cx, cy = w_frame - 60, h_frame - 60
    r = 40
    draw_rounded_rect(frame,
                      (cx - r - 8, cy - r - 8),
                      (cx + r + 8, cy + r + 8),
                      C_BLACK, radius=10, alpha=0.55)
    cv2.circle(frame, (cx, cy), r, C_GRAY, 1, cv2.LINE_AA)

    arrows = {
        "GAUCHE": (cx - r + 8, cy),
        "DROITE": (cx + r - 8, cy),
        "HAUT":   (cx, cy - r + 8),
        "BAS":    (cx, cy + r - 8),
    }
    for label, pt in arrows.items():
        active = (label == direction_h) or (label == direction_v)
        color  = C_GREEN if active else C_GRAY
        cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)

    # Point central
    cv2.circle(frame, (cx, cy), 4, C_WHITE, -1, cv2.LINE_AA)

    # Vecteur actif
    dx = (1 if direction_h == "DROITE" else (-1 if direction_h == "GAUCHE" else 0))
    dy = (1 if direction_v == "BAS"    else (-1 if direction_v == "HAUT"   else 0))
    if dx != 0 or dy != 0:
        norm = np.sqrt(dx**2 + dy**2)
        ex   = int(cx + dx / norm * (r - 10))
        ey   = int(cy + dy / norm * (r - 10))
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), C_GREEN, 2,
                        cv2.LINE_AA, tipLength=0.4)


def draw_calibration_overlay(frame, calibrator):
    """Affiche le point de calibration courant et la progression."""
    h_frame, w_frame = frame.shape[:2]

    # Assombrir l'image
    overlay = np.zeros_like(frame)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Titre
    txt  = f"CALIBRATION — Point {calibrator.current_idx + 1} / {calibrator.n_points}"
    size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
    cv2.putText(frame, txt,
                ((w_frame - size[0]) // 2, 50),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, C_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "Fixez le point rouge",
                ((w_frame - 200) // 2, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GRAY, 1, cv2.LINE_AA)

    # Point cible
    tx, ty = calibrator.current_target_px()
    progress = calibrator.progress()

    # Cercle de progression
    angle = int(360 * progress)
    cv2.ellipse(frame, (tx, ty), (22, 22), -90, 0, angle, C_GREEN, 3, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), 10, C_RED,  -1, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), 10, C_WHITE, 1, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty),  3, C_WHITE, -1, cv2.LINE_AA)

    # Points déjà calibrés
    for i in range(calibrator.current_idx):
        ttx = int(calibrator.targets[i][0] * w_frame)
        tty = int(calibrator.targets[i][1] * h_frame)
        cv2.circle(frame, (ttx, tty), 6, C_GREEN, -1, cv2.LINE_AA)
        cv2.circle(frame, (ttx, tty), 6, C_WHITE,  1, cv2.LINE_AA)


def draw_gaze_cursor(frame, gaze_x_norm, gaze_y_norm):
    """Curseur de regard sur l'image."""
    h_frame, w_frame = frame.shape[:2]
    cx = int(gaze_x_norm * w_frame)
    cy = int(gaze_y_norm * h_frame)
    cx = np.clip(cx, 10, w_frame - 10)
    cy = np.clip(cy, 10, h_frame - 10)

    cv2.circle(frame, (cx, cy), 18, (*C_GREEN, 0), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy),  4, C_GREEN,      -1, cv2.LINE_AA)
    cv2.line(frame, (cx - 24, cy), (cx - 10, cy), C_GREEN, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + 10, cy), (cx + 24, cy), C_GREEN, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 24), (cx, cy - 10), C_GREEN, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + 10), (cx, cy + 24), C_GREEN, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
#  DÉTECTEUR DE CLIGNEMENTS
# ═══════════════════════════════════════════════════════════════════

class BlinkDetector:
    EAR_THRESHOLD = 0.20
    MIN_FRAMES    = 2

    def __init__(self):
        self.blink_count  = 0
        self.closed_frames = 0
        self.last_blink_t  = 0

    def update(self, ear_l, ear_r):
        ear_avg = (ear_l + ear_r) / 2
        if ear_avg < self.EAR_THRESHOLD:
            self.closed_frames += 1
        else:
            if self.closed_frames >= self.MIN_FRAMES:
                now = time.time()
                if now - self.last_blink_t > 0.2:
                    self.blink_count  += 1
                    self.last_blink_t  = now
            self.closed_frames = 0
        return self.closed_frames >= self.MIN_FRAMES  # True si yeux fermés


# ═══════════════════════════════════════════════════════════════════
#  LISSEUR DE GAZE
# ═══════════════════════════════════════════════════════════════════

class GazeSmoother:
    def __init__(self, window=7):
        self.buf_h = deque(maxlen=window)
        self.buf_v = deque(maxlen=window)

    def update(self, h, v):
        self.buf_h.append(h)
        self.buf_v.append(v)

    def get(self):
        if not self.buf_h:
            return 0.5, 0.5
        return float(np.median(self.buf_h)), float(np.median(self.buf_v))


# ═══════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

def build_face_detector():
    """Initialise MediaPipe Face Mesh."""
    mp_fm = mp.solutions.face_mesh
    return mp_fm.FaceMesh(
        max_num_faces       = 1,
        refine_landmarks    = True,   # active la détection des iris
        min_detection_confidence = 0.6,
        min_tracking_confidence  = 0.6,
    )


def build_yolo():
    """Charge le modèle YOLO si disponible."""
    if not YOLO_AVAILABLE or not Config.USE_YOLO:
        return None
    if not os.path.exists(Config.YOLO_MODEL_PATH):
        print(f"[WARN] {Config.YOLO_MODEL_PATH} introuvable — YOLO désactivé.")
        print("       Télécharger : https://github.com/akanametov/yolo-face/releases")
        return None
    print(f"[INFO] Chargement YOLO : {Config.YOLO_MODEL_PATH}")
    return YOLO(Config.YOLO_MODEL_PATH)


def process_frame_yolo(yolo_model, frame):
    """Retourne la liste des bounding boxes visages détectées par YOLO."""
    results = yolo_model(frame, conf=Config.YOLO_CONF, verbose=False)
    boxes   = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, conf))
    return boxes


def run():
    # ── Initialisation ──────────────────────────────────────────────
    yolo       = build_yolo()
    face_mesh  = build_face_detector()
    cap        = cv2.VideoCapture(Config.CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la caméra.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          Config.FPS_TARGET)

    w_cam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_cam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Résolution caméra : {w_cam}x{h_cam}")

    smoother   = GazeSmoother(window=Config.SMOOTH_WINDOW)
    blink_det  = BlinkDetector()
    calibrator = Calibrator(w_cam, h_cam, n_points=Config.CALIB_POINTS)

    # État UI
    show_hud   = True
    fps_buf    = deque(maxlen=30)
    t_prev     = time.time()
    screenshots_dir = "screenshots"

    cv2.namedWindow("Eye Gaze Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Eye Gaze Tracking", w_cam, h_cam)
    print("[INFO] Démarrage. Appuie sur H pour le HUD, C pour calibrer, Q pour quitter.")

    # ── Boucle principale ───────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERREUR] Frame manquante.")
            break

        frame = cv2.flip(frame, 1)  # effet miroir
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── FPS ──────────────────────────────────────────────────────
        t_now   = time.time()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
        fps     = np.mean(fps_buf)
        t_prev  = t_now

        # ── YOLO : détection visages ──────────────────────────────────
        yolo_boxes = []
        if yolo is not None:
            yolo_boxes = process_frame_yolo(yolo, frame)
            if Config.SHOW_BOUNDING_BOX:
                for (x1, y1, x2, y2, conf) in yolo_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), C_BLUE, 2, cv2.LINE_AA)
                    label = f"face {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_BLUE, 1, cv2.LINE_AA)

        # ── MediaPipe : landmarks & iris ──────────────────────────────
        mesh_res = face_mesh.process(rgb)

        ratio_h, ratio_v = 0.5, 0.5
        ear_l, ear_r     = 0.3, 0.3
        blinking         = False
        gaze_x, gaze_y   = 0.5, 0.5

        if mesh_res.multi_face_landmarks:
            lm = mesh_res.multi_face_landmarks[0].landmark

            # Centres iris
            l_iris = iris_center(lm, LEFT_IRIS,  w, h)
            r_iris = iris_center(lm, RIGHT_IRIS, w, h)

            # Ratios du regard
            rh_l = gaze_ratio_h(lm, LEFT_EYE_CORNERS,  l_iris[0], w, h)
            rh_r = gaze_ratio_h(lm, RIGHT_EYE_CORNERS, r_iris[0], w, h)
            rv_l = gaze_ratio_v(lm, LEFT_EYE_TOP_BOT,  l_iris[1], w, h)
            rv_r = gaze_ratio_v(lm, RIGHT_EYE_TOP_BOT, r_iris[1], w, h)

            raw_h = (rh_l + rh_r) / 2
            raw_v = (rv_l + rv_r) / 2

            # EAR
            ear_l = eye_aspect_ratio(lm, LEFT_EYE_TOP_BOT,  LEFT_EYE_CORNERS,  w, h)
            ear_r = eye_aspect_ratio(lm, RIGHT_EYE_TOP_BOT, RIGHT_EYE_CORNERS, w, h)
            blinking = blink_det.update(ear_l, ear_r)

            # Lissage
            if not blinking:
                smoother.update(raw_h, raw_v)
            ratio_h, ratio_v = smoother.get()

            # Calibration
            if calibrator.active:
                calibrator.feed(ratio_h, ratio_v)
            elif calibrator.calibrated:
                gaze_x, gaze_y = calibrator.map_gaze(ratio_h, ratio_v)
            else:
                gaze_x, gaze_y = ratio_h, ratio_v

            # Affichage landmarks
            if Config.SHOW_LANDMARKS and not calibrator.active:
                draw_eye_contour(frame, lm, LEFT_EYE_CONTOUR,  w, h, C_BLUE)
                draw_eye_contour(frame, lm, RIGHT_EYE_CONTOUR, w, h, C_BLUE)
                draw_iris(frame, l_iris, color=C_RED)
                draw_iris(frame, r_iris, color=C_RED)

            if Config.SHOW_GAZE_VECTOR and not calibrator.active:
                draw_gaze_arrow(frame, l_iris, ratio_h, ratio_v)
                draw_gaze_arrow(frame, r_iris, ratio_h, ratio_v)

        # ── Direction ─────────────────────────────────────────────────
        if   ratio_h < Config.GAZE_THRESHOLD_L: dir_h = "GAUCHE"
        elif ratio_h > Config.GAZE_THRESHOLD_R: dir_h = "DROITE"
        else:                                    dir_h = "CENTER"

        if   ratio_v < Config.GAZE_THRESHOLD_U: dir_v = "HAUT"
        elif ratio_v > Config.GAZE_THRESHOLD_D: dir_v = "BAS"
        else:                                    dir_v = "CENTER"

        # ── Curseur de regard ─────────────────────────────────────────
        if not calibrator.active and mesh_res.multi_face_landmarks:
            draw_gaze_cursor(frame, gaze_x, gaze_y)

        # ── Indicateur clignement ─────────────────────────────────────
        if blinking:
            cv2.putText(frame, "* CLIGNEMENT *", (w // 2 - 90, h - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, C_ORANGE, 2, cv2.LINE_AA)

        # ── Overlays ──────────────────────────────────────────────────
        if calibrator.active:
            draw_calibration_overlay(frame, calibrator)
        else:
            draw_direction_indicator(frame, dir_h, dir_v)
            draw_hud(frame, fps, dir_h, dir_v, ear_l, ear_r,
                     ratio_h, ratio_v, calibrator.calibrated,
                     blink_det.blink_count, show_hud)

        # ── Affichage ─────────────────────────────────────────────────
        cv2.imshow("Eye Gaze Tracking", frame)

        # ── Touches ───────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):             # Q / Échap
            break
        elif key == ord('h'):                  # H — toggle HUD
            show_hud = not show_hud
        elif key == ord('c'):                  # C — calibration
            calibrator.reset()
            print("[INFO] Calibration démarrée — fixez chaque point rouge.")
        elif key == ord('r'):                  # R — reset calibration
            calibrator.reset()
            print("[INFO] Calibration réinitialisée.")
        elif key == ord('s'):                  # S — screenshot
            os.makedirs(screenshots_dir, exist_ok=True)
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(screenshots_dir, f"gaze_{ts}.png")
            cv2.imwrite(path, frame)
            print(f"[INFO] Screenshot sauvegardé : {path}")

    # ── Nettoyage ────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print(f"[INFO] Session terminée. Clignements détectés : {blink_det.blink_count}")


# ═══════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run()
