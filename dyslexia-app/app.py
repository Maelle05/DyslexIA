"""Streamlit front-end for the dyslexia eye-tracking experiment.

Manages the full user session: camera start/stop, a 9-point gaze calibration
workflow, recording of gaze data during reading, CSV export, and result display
after sending the recording to the prediction API.

The application is single-threaded from Streamlit's perspective; the OpenCV /
MediaPipe capture runs in a daemon ``threading.Thread`` and communicates through
a shared ``dict`` protected by a ``threading.Lock``.  JavaScript injected via
``st.components.v1.components.html`` handles the animated calibration dot and
the real-time gaze cursor overlay in the parent browser window.
"""
import streamlit as st
import threading
import time
import csv
import io
import streamlit.components.v1 as components
from utils import MONITOR_HEIGHT, MONITOR_WIDTH, READING_TEXT, CALIB_POINTS_PCT
from eye_tracking.computing import eye_tracking_loop
import requests

st.set_page_config(page_title="Eye Tracking", page_icon="👁️", layout="wide")

# ── Shared state ──────────────────────────────────────────────────────────────
if "state" not in st.session_state:
    st.session_state.state = {
        "lock": threading.Lock(),
        "running": False,
        "frame_jpeg": None,
        "screen_xy": (0, 0),
        "calibrated": False,
        "calib_step": -1,
        "do_capture": False,
        "do_reset": False,
        "calib_samples": [],
        "recording": False,
        "gaze_log": [],        # liste de {"t": timestamp, "x": px, "y": py}
        "thread": None,
        "pending_download": False,
        "download_data": None,
        "api_result": None,
    }

S = st.session_state.state

# ══════════════════════════════════════════════════════════════════════════════
# ── Injection JS (point vert + point calibration) ─────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

components.html("""
<script>
(function(){
  const W=window.parent,D=W.document;
  if(!D.getElementById('calib-dot')){
    const c=D.createElement('div'); c.id='calib-dot';
    c.style.cssText='position:fixed;width:24px;height:24px;border-radius:50%;background:#ff1744;border:3px solid #fff;pointer-events:none;z-index:999998;transform:translate(-50%,-50%);display:none;';
    const lbl=D.createElement('div'); lbl.id='calib-lbl';
    lbl.style.cssText='position:fixed;transform:translateX(-50%);color:#fff;font-size:12px;font-family:sans-serif;background:rgba(0,0,0,.65);padding:2px 10px;border-radius:14px;pointer-events:none;z-index:999998;display:none;white-space:nowrap;';
    const num=D.createElement('div'); num.id='calib-num';
    num.style.cssText='position:fixed;color:#fff;font-size:10px;font-weight:bold;font-family:sans-serif;pointer-events:none;z-index:1000000;transform:translate(-50%,-50%);display:none;';
    const sty=D.createElement('style');
    sty.textContent='@keyframes pc{0%{box-shadow:0 0 0 0 rgba(255,23,68,.7)}70%{box-shadow:0 0 0 14px rgba(255,23,68,0)}100%{box-shadow:0 0 0 0 rgba(255,23,68,0)}}#calib-dot.on{animation:pc 1s ease-out infinite}';
    D.head.appendChild(sty);
    D.body.appendChild(c); D.body.appendChild(lbl); D.body.appendChild(num);
  }
  W.addEventListener('gazeUpdate',e=>{
    const d=D.getElementById('gaze-dot');
    d.style.display='block';
    d.style.left=(e.detail.px*100).toFixed(2)+'%';
    d.style.top=(e.detail.py*100).toFixed(2)+'%';
  });
  W.addEventListener('gazeHide',()=>{const d=D.getElementById('gaze-dot');if(d)d.style.display='none';});
  W.addEventListener('calibPoint',e=>{
    const c=D.getElementById('calib-dot'),lbl=D.getElementById('calib-lbl'),num=D.getElementById('calib-num');
    const px=e.detail.px*100,py=e.detail.py*100;
    c.style.left=px+'%'; c.style.top=py+'%'; c.style.display='block'; c.classList.add('on');
    lbl.style.left=px+'%'; lbl.style.top=`calc(${py}% + ${py>85?'-38px':'22px'})`;
    lbl.textContent=`Point ${e.detail.step}/9 — Fixez et cliquez`; lbl.style.display='block';
    num.style.left=px+'%'; num.style.top=py+'%'; num.textContent=e.detail.step; num.style.display='block';
  });
  W.addEventListener('calibHide',()=>{
    ['calib-dot','calib-lbl','calib-num'].forEach(id=>{
      const el=D.getElementById(id);
      if(el){el.style.display='none';if(id==='calib-dot')el.classList.remove('on');}
    });
  });
})();
</script>
""", height=0)

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
                S["running"]=True; S["calib_step"]=-1
                S["calib_samples"]=[]; S["calibrated"]=False
            t = threading.Thread(target=eye_tracking_loop, args=(S,), daemon=True)
            t.start(); S["thread"]=t
            st.rerun()
    else:
        if st.button("⏹ Arrêter la caméra", use_container_width=True):
            with S["lock"]:
                S["running"]=False; S["recording"]=False
            st.rerun()

    st.divider()

    with S["lock"]:
        calib_step = S["calib_step"]
        calibrated = S["calibrated"]

    if calib_step == -1:
        if st.button("🎯 Lancer la calibration", disabled=not running, use_container_width=True):
            with S["lock"]:
                S["calib_step"]=0; S["calib_samples"]=[]; S["calibrated"]=False
            st.rerun()

    elif 0 <= calib_step <= 8:
        st.info(f"**Point {calib_step+1} / 9**\nFixez le point rouge")
        st.progress(calib_step/9, text=f"{calib_step}/9")
        if st.button(f"✅ Capturer point {calib_step+1}", use_container_width=True, type="primary"):
            with S["lock"]: S["do_capture"]=True
            time.sleep(0.15); st.rerun()
        if st.button("✖ Annuler", use_container_width=True):
            with S["lock"]: S["calib_step"]=-1; S["calib_samples"]=[]
            st.rerun()

    elif calib_step == 10:
        st.success("✅ Calibration terminée !")
        if st.button("🔄 Recalibrer", use_container_width=True):
            with S["lock"]: S["do_reset"]=True
            st.rerun()

    st.divider()

    # ── Enregistrement ────────────────────────────────────────────────────────
    with S["lock"]:
        recording = S["recording"]
        n_log     = len(S["gaze_log"])

    st.subheader("Enregistrement")

    if not recording:
        btn_label = "⏺ Démarrer l'enregistrement"
        btn_disabled = not calibrated
        if st.button(btn_label, disabled=btn_disabled, use_container_width=True, type="primary"):
            with S["lock"]:
                S["recording"]=True; S["gaze_log"]=[]
            st.rerun()
    else:
        st.error(f"🔴 Enregistrement en cours… {n_log} pts")

        if st.button("⏹ Arrêter l'enregistrement", use_container_width=True):
            with S["lock"]:
                S["recording"] = False
                S["pending_download"] = True
                S["download_data"] = list(S["gaze_log"]) if S["gaze_log"] else []

    if S.get("pending_download") and S.get("download_data"):
        st.success("✅ Enregistrement terminé")

        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=["time", "fix_x", "fix_y", "gaze_x_left", "gaze_y_left", "gaze_x_right", "gaze_y_right"])
        writer.writeheader()
        writer.writerows(S["download_data"])

        csv_data = out.getvalue()

        csv_bytes = io.BytesIO(csv_data.encode("utf-8"))
        csv_bytes.name = "gaze_log.csv"

        if csv_data:
            if st.button('Envoyer les résultats'):
                response = requests.post('http://localhost:8000/predict', files={"csv_file": csv_bytes})
                if response.status_code == 200:
                    S["api_result"] = response.json()
                else:
                    S["api_result"] = f"Error {response.status_code}: {response.text}"
                st.rerun()

    # Miniature caméra en bas de la colonne
    st.divider()
    with S["lock"]:
        jpeg = S["frame_jpeg"]
    if jpeg:
        st.image(jpeg, use_container_width=True,
                 caption="Caméra")
    else:
        st.caption("Caméra inactive")


# ── Colonne principale : texte à lire ─────────────────────────────────────────
with col_main:
    with S["lock"]:
        calibrated = S["calibrated"]
        recording  = S["recording"]

    st.markdown("#### Texte à lire")
    st.markdown(
        f"<div style='"
        f"font-size:20px;line-height:2.0;font-family:Georgia,serif;"
        f"max-width:720px;margin:0 auto;"
        f"padding:32px 32px;"
        f"background:var(--background-color);"
        f"border-radius:12px;"
        f"border:1px solid rgba(128,128,128,0.2);"
        f"'>{READING_TEXT.replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True,
    )

    if S["api_result"] is not None:
        st.divider()
        st.markdown("#### Résultats")
        st.write(S["api_result"])

# ══════════════════════════════════════════════════════════════════════════════
# ── Dispatch JS ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

with S["lock"]:
    sx, sy     = S["screen_xy"]
    is_running = S["running"]
    is_cal     = S["calibrated"]
    step       = S["calib_step"]

if is_running and is_cal and MONITOR_WIDTH > 0:
    px=sx/MONITOR_WIDTH; py=sy/MONITOR_HEIGHT
    components.html(f"<script>window.parent.dispatchEvent(new CustomEvent('gazeUpdate',{{detail:{{px:{px:.4f},py:{py:.4f}}}}}));</script>", height=0)
else:
    components.html("<script>window.parent.dispatchEvent(new CustomEvent('gazeHide'));</script>", height=0)

if is_running and 0 <= step <= 8:
    ppx,ppy = CALIB_POINTS_PCT[step]
    components.html(f"<script>window.parent.dispatchEvent(new CustomEvent('calibPoint',{{detail:{{px:{ppx},py:{ppy},step:{step+1}}}}}));</script>", height=0)
else:
    components.html("<script>window.parent.dispatchEvent(new CustomEvent('calibHide'));</script>", height=0)

if S["running"]:
    time.sleep(0.1)
    st.rerun()
