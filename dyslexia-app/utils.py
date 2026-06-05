"""Shared constants for the dyslexia eye-tracking Streamlit application.

Provides:
    - ``READING_TEXT``: the French passage displayed to subjects.
    - ``MONITOR_WIDTH`` / ``MONITOR_HEIGHT``: screen dimensions in pixels,
      queried once at import time via :func:`pyautogui.size`.
    - ``CALIB_POINTS_PCT``: 9 calibration point positions as ``(x, y)``
      fractions of the monitor dimensions (3 × 3 grid).
    - ``NOSE_INDICES``: MediaPipe FaceMesh landmark indices used to build the
      head-pose reference frame.
"""
import pyautogui

# ── Texte à lire ─────────────────────────────────────────────────────────────
READING_TEXT = """J’ai ainsi vécu seul, sans personne avec qui parler véritablement,
jusqu’à une panne dans le désert du Sahara, il y a six ans.
Quelque chose s’était cassé dans mon moteur.
Et comme je n’avais avec moi ni mécanicien, ni passagers,
je me préparai à essayer de réussir, tout seul, une réparation difficile.
C’était pour moi une question de vie ou de mort.
J’avais à peine de l’eau à boire pour huit jours.
Le premier soir je me suis donc endormi sur le sable à mille milles de toute terre habitée.
J’étais bien plus isolé qu’un naufragé sur un radeau au milieu de l’Océan.
Alors vous imaginez ma surprise, au lever du jour, quand une drôle de petite voix m’a réveillé.
"""

MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()

CALIB_POINTS_PCT = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]

NOSE_INDICES = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391, 3, 248]
