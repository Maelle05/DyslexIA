import streamlit as st
import cv2
import numpy as np
import threading
import time
import mediapipe as mp
import math
import json
import csv
import io
from collections import deque
from scipy.spatial.transform import Rotation as Rscipy
import pyautogui
import streamlit.components.v1 as components

st.set_page_config(page_title="Eye Tracking", page_icon="👁️", layout="wide")

# ── Texte à lire ──────────────────────────────────────────────────────────────
READING_TEXT = """J’ai ainsi vécu seul, sans personne avec qui parler véritablement, jusqu’à une panne dans le désert du Sahara, il y a six ans.
Quelque chose s’était cassé dans mon moteur.
Et comme je n’avais avec moi ni mécanicien, ni passagers, je me préparai à essayer de réussir, tout seul, une réparation difficile.
C’était pour moi une question de vie ou de mort.
J’avais à peine de l’eau à boire pour huit jours.
Le premier soir je me suis donc endormi sur le sable à mille milles de toute terre habitée.
J’étais bien plus isolé qu’un naufragé sur un radeau au milieu de l’Océan.
Alors vous imaginez ma surprise, au lever du jour, quand une drôle de petite voix m’a réveillé.
"""

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
    }

S = st.session_state.state
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()

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

def compute_scale(pts):
    n = len(pts); total = 0; count = 0
    for i in range(n):
        for j in range(i+1, n):
            total += np.linalg.norm(pts[i]-pts[j]); count += 1
    return total/count if count else 1.0

def compute_head_frame(lms, indices, ref, w, h):
    pts = np.array([[lms[i].x*w, lms[i].y*h, lms[i].z*w] for i in indices])
    c = np.mean(pts, axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov((pts-c).T))
    eigvecs = eigvecs[:, np.argsort(-eigvals)]
    if np.linalg.det(eigvecs) < 0: eigvecs[:, 2] *= -1
    ro, pi, ya = Rscipy.from_matrix(eigvecs).as_euler('zyx', degrees=False)
    R = Rscipy.from_euler('zyx', [ro, pi, ya]).as_matrix()
    if ref[0] is None: ref[0] = R.copy()
    else:
        for i in range(3):
            if np.dot(R[:, i], ref[0][:, i]) < 0: R[:, i] *= -1
    return c, R, pts

def gaze_to_angles(d):
    d = d/np.linalg.norm(d)
    xz = np.array([d[0],0,d[2]]); xz /= np.linalg.norm(xz)
    yaw = math.acos(np.clip(np.dot([0,0,-1], xz), -1, 1))
    if d[0] < 0: yaw = -yaw
    yz = np.array([0,d[1],d[2]]); yz /= np.linalg.norm(yz)
    pitch = math.acos(np.clip(np.dot([0,0,-1], yz), -1, 1))
    if d[1] > 0: pitch = -pitch
    return -math.degrees(yaw), math.degrees(pitch)

def fit_map(samples):
    X = np.column_stack([[s[0] for s in samples], [s[1] for s in samples], np.ones(len(samples))])
    cx, *_ = np.linalg.lstsq(X, [s[2] for s in samples], rcond=None)
    cy, *_ = np.linalg.lstsq(X, [s[3] for s in samples], rcond=None)
    return cx, cy

def apply_map(yaw, pitch, cx, cy):
    return (int(np.clip(cx[0]*yaw+cx[1]*pitch+cx[2], 10, MONITOR_WIDTH-10)),
            int(np.clip(cy[0]*yaw+cy[1]*pitch+cy[2], 10, MONITOR_HEIGHT-10)))

# ══════════════════════════════════════════════════════════════════════════════
# ── Thread OpenCV ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def eye_tracking_loop(shared):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref=[None]; ll=rl=False; loff=roff=lcal=rcal=None
    buf = deque(maxlen=10); cxm=cym=None; base_r=20

    while True:
        with shared["lock"]:
            if not shared["running"]: break
            do_cap = shared["do_capture"]
            do_rst = shared["do_reset"]
            recording = shared["recording"]
            if do_cap: shared["do_capture"] = False
            if do_rst:
                shared["do_reset"]=False; shared["calib_samples"]=[]; shared["calib_step"]=-1
                shared["calibrated"]=False; ll=rl=False; ref[0]=None; cxm=cym=None

        ret, frame = cap.read()
        if not ret: time.sleep(0.05); continue

        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        sxy = (0,0)

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            hc, R, npts = compute_head_frame(lms, nose_indices, ref, w, h)
            li=lms[468]; ri=lms[473]
            il=np.array([li.x*w,li.y*h,li.z*w]); ir=np.array([ri.x*w,ri.y*h,ri.z*w])

            if do_cap and not ll:
                cur=compute_scale(npts); cl=R.T@np.array([0,0,1])
                loff=R.T@(il-hc)+base_r*cl; roff=R.T@(ir-hc)+base_r*cl
                lcal=rcal=cur; ll=rl=True

            if do_cap and ll:
                cur=compute_scale(npts)
                sl=hc+R@(loff*(cur/lcal)); sr=hc+R@(roff*(cur/rcal))
                dl=il-sl; dl/=np.linalg.norm(dl); dr=ir-sr; dr/=np.linalg.norm(dr)
                raw=(dl+dr)/2; raw/=np.linalg.norm(raw)
                ya,pi=gaze_to_angles(raw)
                with shared["lock"]:
                    step=shared["calib_step"]
                    if 0<=step<=8:
                        px,py=CALIB_POINTS_PCT[step]
                        shared["calib_samples"].append((ya,pi,int(px*MONITOR_WIDTH),int(py*MONITOR_HEIGHT)))
                        if len(shared["calib_samples"])>=9:
                            cxm,cym=fit_map(shared["calib_samples"])
                            shared["calibrated"]=True; shared["calib_step"]=10
                        else:
                            shared["calib_step"]=step+1

            if ll and rl and cxm is not None:
                cur=compute_scale(npts)
                sl=hc+R@(loff*(cur/lcal)); sr=hc+R@(roff*(cur/rcal))
                rl2=int(base_r*(cur/lcal)); rr2=int(base_r*(cur/rcal))
                dl=il-sl
                dl/=np.linalg.norm(dl)
                dr=ir-sr
                dr/=np.linalg.norm(dr)

                yaw_l, pitch_l = gaze_to_angles(dl)
                yaw_r, pitch_r = gaze_to_angles(dr)

                lx, ly = apply_map(yaw_l, pitch_l, cxm, cym)
                rx, ry = apply_map(yaw_r, pitch_r, cxm, cym)

                raw=(dl+dr)/2; raw/=np.linalg.norm(raw)
                buf.append(raw)
                avg=np.mean(buf,axis=0); avg/=np.linalg.norm(avg)
                ya,pi=gaze_to_angles(avg); sxy=apply_map(ya,pi,cxm,cym)
                cv2.circle(frame,(int(sl[0]),int(sl[1])),rl2,(255,255,25),2)
                cv2.circle(frame,(int(sr[0]),int(sr[1])),rr2,(25,255,255),2)
                orig=(sl+sr)/2
                cv2.line(frame,tuple(int(v) for v in orig[:2]),
                         tuple(int(v) for v in (orig+avg*180)[:2]),(255,255,10),2)
                cv2.putText(frame,"CALIBRATED",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,100),1)

                if recording:
                    entry={
                        "time":time.time(),
                        "fix_x":sxy[0],
                        "fix_y":sxy[1],
                        "gaze_x_left": lx,
                        "gaze_y_left": ly,
                        "gaze_x_right": rx,
                        "gaze_y_right": ry
                    }
                    with shared["lock"]:
                        shared["gaze_log"].append(entry)
            elif ll:
                cv2.putText(frame,"Calibration...",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,200,0),1)
            else:
                cv2.circle(frame,(int(li.x*w),int(li.y*h)),7,(255,50,50),2)
                cv2.circle(frame,(int(ri.x*w),int(ri.y*h)),7,(50,255,50),2)

            for lm in lms:
                cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),1,(180,180,180),-1)

            with shared["lock"]:
                shared["screen_xy"]=sxy

        _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        with shared["lock"]:
            shared["frame_jpeg"]=enc.tobytes()
        time.sleep(0.01)

    cap.release(); face_mesh.close()
    with shared["lock"]:
        shared["running"]=False; shared["frame_jpeg"]=None

# ══════════════════════════════════════════════════════════════════════════════
# ── Injection JS (point vert + point calibration) ─────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

components.html("""
<script>
(function(){
  const W=window.parent,D=W.document;
  if(!D.getElementById('gaze-dot')){
    const d=D.createElement('div'); d.id='gaze-dot';
    d.style.cssText='position:fixed;width:20px;height:20px;border-radius:50%;background:#00e676;border:3px solid #fff;box-shadow:0 0 12px #00e676;pointer-events:none;z-index:999999;transform:translate(-50%,-50%);transition:left .07s ease-out,top .07s ease-out;display:none;';
    D.body.appendChild(d);
  }
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

        st.download_button(
            label="💾 Télécharger CSV",
            data=csv_data,
            file_name="gaze_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Miniature caméra en bas de la colonne
    st.divider()
    with S["lock"]:
        jpeg = S["frame_jpeg"]
    if jpeg:
        st.image(jpeg, channels="BGR", use_container_width=True,
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
