"""Streamlit front-end for the dyslexia eye-tracking experiment.

Manages the full user session: camera start/stop, recording of gaze angles
during reading, CSV export, and result display after sending the recording
to the prediction API.

The application is single-threaded from Streamlit's perspective; the OpenCV /
MediaPipe capture runs in a daemon ``threading.Thread`` and communicates through
a shared ``dict`` protected by a ``threading.Lock``.
"""
import streamlit as st
import threading
import time
import csv
import io
import requests
from utils import READING_TEXT
from eye_tracking.computing import eye_tracking_loop

st.set_page_config(page_title="Eye Tracking", page_icon="👁️", layout="wide")

try:
    text_request = requests.get('http://localhost:8000/passage')

    if text_request.status_code == 200:
        reading_text = text_request.json()['text']
    else:
        reading_text = READING_TEXT
except requests.exceptions.ConnectionError:
    reading_text = READING_TEXT



# ── Shared state ──────────────────────────────────────────────────────────────
if "state" not in st.session_state:
    st.session_state.state = {
        "lock":             threading.Lock(),
        "running":          False,
        "tracking":         False,   # True once first face is detected
        "frame_jpeg":       None,
        "recording":        False,
        "gaze_log":         [],
        "thread":           None,
        "pending_download": False,
        "download_data":    None,
        "api_result":       None,
        "fps":              None,
        "text":             reading_text,
    }

S = st.session_state.state

# ══════════════════════════════════════════════════════════════════════════════
# ── UI ────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

col_ctrl, col_main = st.columns([1, 4])

# ── Colonne gauche : contrôles ────────────────────────────────────────────────
with col_ctrl:
    st.subheader("Contrôles")

    running = S["running"]
    if not running:
        if st.button("▶ Démarrer la caméra", use_container_width=True):
            with S["lock"]:
                S["running"]  = True
                S["tracking"] = False
            t = threading.Thread(target=eye_tracking_loop, args=(S,), daemon=True)
            t.start()
            S["thread"] = t
            st.rerun()
    else:
        if st.button("⏹ Arrêter la caméra", use_container_width=True):
            with S["lock"]:
                S["running"]   = False
                S["recording"] = False
                S["tracking"]  = False
            st.rerun()

    st.divider()

    # ── Statut du tracking ────────────────────────────────────────────────────
    with S["lock"]:
        tracking = S["tracking"]

    if running and not tracking:
        st.info("👁️ En attente du visage…")
    elif running and tracking:
        with S["lock"]:
            fps = S["fps"]
        fps_str = f" — {fps} fps" if fps else ""
        st.success(f"✅ Visage détecté{fps_str}")

    st.divider()

    # ── Enregistrement ────────────────────────────────────────────────────────
    with S["lock"]:
        recording = S["recording"]
        n_log     = len(S["gaze_log"])

    st.subheader("Enregistrement")

    if not recording:
        if st.button(
            "⏺ Démarrer l'enregistrement",
            disabled=not tracking,
            use_container_width=True,
            type="primary",
        ):
            with S["lock"]:
                S["recording"] = True
                S["gaze_log"]  = []
            st.rerun()
    else:
        st.error(f"🔴 Enregistrement en cours… {n_log} pts")

        if st.button("⏹ Arrêter l'enregistrement", use_container_width=True):
            with S["lock"]:
                S["recording"]        = False
                S["pending_download"] = True
                S["download_data"]    = list(S["gaze_log"]) if S["gaze_log"] else []

    if S.get("pending_download") and S.get("download_data"):
        st.success("✅ Enregistrement terminé")

        out = io.StringIO()
        writer = csv.DictWriter(
            out,
            fieldnames=["time", "angle1_l", "angle2_l", "angle1_r", "angle2_r"],
        )
        writer.writeheader()
        writer.writerows(S["download_data"])
        csv_data = out.getvalue()

        csv_bytes = io.BytesIO(csv_data.encode("utf-8"))
        csv_bytes.name = "gaze_log.csv"

        if csv_data:
            if st.button("Envoyer les résultats"):
                response = requests.post(
                    "http://localhost:8000/predict",
                    files={"csv_file": csv_bytes},
                )
                if response.status_code == 200:
                    S["api_result"] = response.json()
                else:
                    S["api_result"] = f"Error {response.status_code}: {response.text}"
                st.rerun()

        st.download_button(
            label="💾 Télécharger CSV",
            data=csv_data,
            file_name="gaze_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Miniature caméra ──────────────────────────────────────────────────────
    st.divider()
    with S["lock"]:
        jpeg = S["frame_jpeg"]
    if jpeg:
        st.image(jpeg, use_container_width=True, caption="Caméra")
    else:
        st.caption("Caméra inactive")


# ── Colonne principale : texte à lire ─────────────────────────────────────────
with col_main:
    st.markdown("#### Texte à lire")

    st.markdown(
        f"<div style='"
        f"font-size:20px;line-height:2.0;font-family:Georgia,serif;"
        f"max-width:720px;margin:0 auto;"
        f"padding:32px 32px;"
        f"background:var(--background-color);"
        f"border-radius:12px;"
        f"border:1px solid rgba(128,128,128,0.2);"
        f"'>{S['text'].replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True,
    )

    with S["lock"]:
        api_result = S["api_result"]
    if api_result is not None:
        st.divider()
        st.markdown("#### Résultats")
        st.write(api_result)


# ── Boucle de rafraîchissement ────────────────────────────────────────────────
if S["running"]:
    time.sleep(0.1)
    st.rerun()
