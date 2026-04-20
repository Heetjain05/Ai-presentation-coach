"""
AI Presentation Coach — Advanced v3.0
Silver Oak University | Heet Jain | 2201031000030
Run: streamlit run app.py
"""
import streamlit as st
import subprocess
import cv2, numpy as np, pickle, time, threading
import re, wave, os, tempfile, datetime
import mediapipe as mp
import sounddevice as sd
import speech_recognition as sr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque, Counter
from threading import Lock
from step1_collect_data import extract_features

st.set_page_config(layout="wide", page_title="AI Presentation Coach", page_icon="🎤")

# ================================================================ CSS
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Share+Tech+Mono&display=swap');
html,body,[class*="css"]{background:#080d18!important;color:#c8d8e8!important;font-family:'Share Tech Mono',monospace!important}
.stApp{background:#080d18!important}
.card{background:#0d1b2a;border:1px solid #1e3a5a;border-radius:12px;padding:14px 16px;margin-bottom:10px;position:relative;overflow:hidden}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00e5ff,#00ff88)}
.sh{font-size:10px;letter-spacing:4px;color:#445566;margin-bottom:8px;display:block}
.snum{font-family:'Orbitron',monospace;font-size:56px;font-weight:900;text-align:center;line-height:1;margin:4px 0 2px}
.ssub{font-size:10px;letter-spacing:3px;color:#667788;text-align:center}
.pbg{background:#1a2a3a;border-radius:4px;height:8px;margin:8px 0;overflow:hidden}
.pfl{height:100%;border-radius:4px}
.bdg{display:block;text-align:center;border:2px solid;border-radius:20px;padding:6px 0;font-family:'Orbitron',monospace;font-size:11px;font-weight:700;letter-spacing:3px;margin-top:8px}
.vbar{border-radius:8px;padding:8px 12px;font-size:12px;margin-top:8px;display:flex;align-items:center;gap:8px}
.mg{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px}
.mb{background:#091522;border:1px solid #1a3050;border-radius:8px;padding:12px;text-align:center}
.mv{font-family:'Orbitron',monospace;font-size:22px;font-weight:700;margin-bottom:2px}
.ml{font-size:9px;letter-spacing:2px;color:#445566}
.fi{background:#091522;border-left:3px solid;border-radius:0 8px 8px 0;padding:8px 12px;margin-bottom:5px;font-size:12px;display:flex;align-items:center;gap:8px}
.ci{display:flex;align-items:center;gap:10px;padding:6px 2px;font-size:12px;border-bottom:1px solid #112233}
.dot{width:7px;height:7px;border-radius:50%;background:#00e5ff;flex-shrink:0}
.sg{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:6px}
.sg2{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:6px}
.sb{background:#0d1b2a;border:1px solid #1e3a5a;border-radius:10px;padding:16px;text-align:center;position:relative;overflow:hidden}
.sb::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00e5ff55,#00ff8855)}
.sl{font-size:9px;letter-spacing:3px;color:#667788;margin-bottom:6px}
.sv{font-family:'Orbitron',monospace;font-size:26px;font-weight:700}
.tag{display:inline-block;background:#ff445522;border:1px solid #ff4455;color:#ff8899;border-radius:4px;padding:2px 8px;font-size:11px;margin:2px}
.tag-g{display:inline-block;background:#00ff8822;border:1px solid #00ff88;color:#00ff88;border-radius:4px;padding:2px 8px;font-size:11px;margin:2px}
.stButton>button{background:#091522!important;border:1px solid #1e3a5a!important;color:#c8d8e8!important;border-radius:8px!important;font-family:'Share Tech Mono',monospace!important;letter-spacing:1px!important;width:100%!important}
.stButton>button:hover{border-color:#00e5ff!important;color:#00e5ff!important}
.stTabs [data-baseweb="tab-list"]{background:#0d1b2a!important;border-bottom:2px solid #00e5ff33!important}
.stTabs [data-baseweb="tab"]{color:#556677!important;font-family:'Share Tech Mono',monospace!important;font-size:12px!important;padding:10px 18px!important}
.stTabs [aria-selected="true"]{color:#00e5ff!important;background:#091522!important;border-bottom:2px solid #00e5ff!important}
.stSelectbox>div>div{background:#0d1b2a!important;border:1px solid #1e3a5a!important;color:#c8d8e8!important;border-radius:8px!important}
.stExpander{border:1px solid #1e3a5a!important;border-radius:8px!important;background:#0d1b2a!important}
.stTextArea textarea{background:#0d1b2a!important;border:1px solid #1e3a5a!important;color:#c8d8e8!important;font-family:'Share Tech Mono',monospace!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:.6rem 1.2rem!important}
div[data-testid="column"]{padding:4px!important}
</style>""", unsafe_allow_html=True)

# ================================================================ CONSTANTS
SAMPLE_RATE = 16000
CHUNK_SEC   = 5
SPK_THR     = 0.003
WPM_MIN, WPM_MAX = 120, 160
FILLERS = {
    "um","uh","like","you know","basically","literally","actually",
    "right","okay","so","well","hmm","ah","er","kind of","sort of",
    "i mean","anyway","alright","yeah"
}

# ================================================================ STATE
@st.cache_resource
def get_G():
    return {
        "lock": Lock(), "running": False,
        "frame_jpg": None,
        # posture
        "status": "waiting", "confidence": 0.0, "score": 0,
        "eye_contact": 0.0, "head_tilt": 0.0,
        "shoulder_ok": True, "face_detected": False,
        "emotion": "😐 neutral", "mouth_open": 0.0,
        # audio
        "speaking": False, "volume": 0, "pace": "normal",
        "energy_ring": deque(maxlen=30),
        "audio_chunks": [], "v_hist": [],
        "full_audio": [],
        # speech
        "transcript": "", "sr_status": "idle",
        # history
        "score_hist": [], "eye_hist": [],
        # recording
        "recorder": None, "rec_path": "",
    }

G = get_G()

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        d = pickle.load(f)
    return d['model'], d['label_encoder'], d['feature_names']

model, le, feature_names = load_model()

for k, v in [("t_start", None), ("last_append", 0.0), ("last_chart", 0.0), ("history", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ================================================================ HELPERS
def score_col(s):
    return "#00ff88" if s>=80 else "#00e5ff" if s>=65 else "#ffcc00" if s>=50 else "#ff4455"

def score_lbl(s):
    if s>=85: return "EXCELLENT","#00ff88"
    if s>=70: return "GOOD","#00e5ff"
    if s>=55: return "AVERAGE","#ffcc00"
    return "NEEDS WORK","#ff4455"

def pace_detect(ring):
    if len(ring)<10: return "normal"
    r = sum(1 for e in list(ring)[-10:] if e>SPK_THR)/10
    return "fast" if r>.8 else "slow" if r<.3 else "normal"

def detect_emotion(mouth_open, eye_contact, tilt):
    if mouth_open>.03 and eye_contact>.6: return "😄 confident"
    if tilt>15:                           return "😕 uncertain"
    if eye_contact<.3:                    return "😟 nervous"
    if mouth_open>.02:                    return "🙂 engaged"
    return "😐 neutral"

def compute_metrics(frame, fm, pose):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    eye=0.0; tilt=0.0; sh=True; face=False; mo=0.0
    fr = fm.process(rgb)
    if fr.multi_face_landmarks:
        lm = fr.multi_face_landmarks[0].landmark
        xs=[p.x for p in lm]; ys=[p.y for p in lm]
        cx,cy=float(np.mean(xs)),float(np.mean(ys))
        face=True
        le_,re_=lm[33],lm[263]
        tilt=float(abs(np.degrees(np.arctan2(re_.y-le_.y,re_.x-le_.x))))
        eye=float(max(0,min(1,1-(abs(cx-.5)*2+abs(cy-.4)*2)/2)))
        mo=float(abs(lm[14].y-lm[13].y))
    pr = pose.process(rgb)
    if pr.pose_landmarks:
        pl=pr.pose_landmarks.landmark
        sh=abs(pl[11].y-pl[12].y)<0.05
    return eye, tilt, sh, face, mo

def calc_score(face, eye, tilt, sh, spk, fct=0, wpm=0):
    s=10
    if face:   s+=20
    s+=min(eye*25,25)
    if tilt<10: s+=15
    if sh:     s+=15
    if spk:    s+=10
    if fct==0 and wpm>0: s+=3
    if WPM_MIN<=wpm<=WPM_MAX: s+=2
    return min(100,int(s))

# ================================================================ SPEECH
def analyze_speech(text):
    if not text: return 0,[],0.0,0
    words=re.findall(r"\b\w+\b",text.lower())
    if not words: return 0,[],0.0,0
    fw=[w for w in words if w in FILLERS]
    vr=round(len(set(words))/len(words),2)
    return len(fw),fw,vr,len(words)

def get_english_grade(fr,vr,asl):
    if fr==0  and vr>=.7 and asl>=8: return "A+","#00ff88","Perfect fluency!"
    if fr<=3  and vr>=.6 and asl>=6: return "A", "#00e5ff","Excellent English"
    if fr<=8  and vr>=.5:            return "B", "#ffcc00","Good — minor issues"
    if fr<=15:                        return "C", "#ff8800","Needs improvement"
    return "D","#ff4455","Practice more"

def analyze_english(text):
    if not text or len(text.split())<5: return None
    words=re.findall(r"\b\w+\b",text.lower())
    sents=[s.strip() for s in re.split(r'[.!?]+',text) if len(s.strip())>5]
    fw=[w for w in words if w in FILLERS]
    fct=len(fw); vr=round(len(set(words))/len(words),2) if words else 0
    fr=round(fct/len(words)*100,1) if words else 0
    asl=round(len(words)/max(len(sents),1),1)
    gr,gc,gm=get_english_grade(fr,vr,asl)
    issues=[]
    for s in sents[:5]:
        w=s.lower().split()
        for i in range(len(w)-1):
            if w[i]==w[i+1] and w[i] not in {"very","really"}:
                issues.append(f"Repeated: '{w[i]}'")
        if w and w[0] in {"and","but","because"} and len(sents)>1:
            issues.append("Avoid starting with conjunction")
    return {"fct":fct,"fw":Counter(fw).most_common(6),
            "vr":vr,"fr":fr,"nw":len(words),"nsent":len(sents),
            "asl":asl,"gr":gr,"gc":gc,"gm":gm,
            "issues":list(dict.fromkeys(issues))[:2],"unique":len(set(words))}

# ================================================================ FEEDBACK
def get_feedback(face,eye,tilt,sh,spk,vol,pace,fct,vr,wpm,emotion):
    out=[]
    out.append(("✅","#00ff88","Face clearly detected") if face else ("🚫","#ff4455","Face not in frame"))
    out.append(("👁","#00ff88" if eye>=.6 else "#ffcc00",
        f"Strong eye contact ({eye:.0%})" if eye>=.6 else f"Eye contact {eye:.0%} — look at camera"))
    out.append(("📐","#00ff88" if tilt<10 else "#ffcc00",
        "Head level — great!" if tilt<10 else f"Head tilted {tilt:.0f}°"))
    out.append(("💪","#00ff88" if sh else "#ffcc00",
        "Shoulders balanced" if sh else "Balance your shoulders"))
    if not spk:
        out.append(("🔇","#ff6633","No voice — start speaking"))
    else:
        out.append(("🎤" if vol>=15 else "🔊","#00ff88" if vol>=15 else "#ffcc00",
            f"Good volume ({vol})" if vol>=15 else "Voice quiet — project more"))
        out.append({"fast":("⚡","#ffcc00","Too fast — slow down"),
                    "slow":("🐢","#ffcc00","Too slow — pick up pace"),
                    "normal":("✅","#00ff88","Perfect speaking pace")}[pace])
    if fct==0:   out.append(("🗣","#00ff88","No filler words — excellent!"))
    elif fct<=3: out.append(("🗣","#ffcc00",f"{fct} fillers detected"))
    else:        out.append(("🗣","#ff4455",f"{fct} fillers — avoid um/uh"))
    if vr>=.7:   out.append(("📚","#00ff88",f"Rich vocabulary ({vr:.0%})"))
    elif vr>0:   out.append(("📚","#ffcc00",f"Vocabulary {vr:.0%} — vary words"))
    if WPM_MIN<=wpm<=WPM_MAX: out.append(("⏱","#00ff88",f"Ideal pace: {wpm} wpm"))
    elif wpm>WPM_MAX:          out.append(("⏱","#ffcc00",f"Fast: {wpm} wpm"))
    elif wpm>0:                out.append(("⏱","#ffcc00",f"Slow: {wpm} wpm"))
    out.append(("🎭","#00aaff",f"Emotion: {emotion}"))
    return out

# ================================================================ CHARTS
def make_timeline(hist):
    fig,ax=plt.subplots(figsize=(12,2.6))
    fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#080d18')
    xs=list(range(len(hist)))
    ax.fill_between(xs,hist,alpha=0.2,color='#00e5ff')
    ax.plot(xs,hist,color='#00e5ff',lw=2)
    avg=sum(hist)/len(hist)
    ax.axhline(avg,color='#ffcc00',lw=1,ls='--')
    ax.axhline(80,color='#00ff8844',lw=1,ls=':')
    ax.text(xs[-1],avg+2,f"Avg:{avg:.0f}",color='#ffcc00',fontsize=8,ha='right')
    ax.set_ylim(0,100); ax.set_xlim(0,max(1,max(xs)))
    ax.tick_params(colors='#334455',labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#1e3a5a')
    plt.tight_layout(pad=0.3); return fig

def make_radar(eye,tilt_sc,sh_sc,fluency,vol_sc,wpm_sc):
    labels=['Eye\nContact','Head\nPosture','Shoulders','Fluency','Voice\nVol','WPM\nPace']
    vals=[max(0,min(100,v)) for v in [eye*100,tilt_sc,sh_sc,fluency,vol_sc,wpm_sc]]
    N=len(labels)
    angles=[n/float(N)*2*np.pi for n in range(N)]+[0]
    vals_p=vals+vals[:1]
    fig,ax=plt.subplots(figsize=(4,4),subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#091522')
    ax.plot(angles,vals_p,color='#00e5ff',lw=2)
    ax.fill(angles,vals_p,color='#00e5ff',alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels,color='#aabbcc',fontsize=8)
    ax.set_ylim(0,100); ax.yaxis.set_ticklabels([])
    ax.spines['polar'].set_color('#1e3a5a'); ax.grid(color='#1e3a5a',linewidth=0.5)
    plt.tight_layout(); return fig

def make_vol_bar(v_hist):
    fig,ax=plt.subplots(figsize=(12,1.5))
    fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#080d18')
    xs=list(range(len(v_hist)))
    cols=['#00ff88' if v>30 else '#ffcc00' if v>10 else '#ff4455' for v in v_hist]
    ax.bar(xs,[v/100 for v in v_hist],color=cols,alpha=0.8,width=1.0)
    ax.set_ylim(0,1); ax.set_xlim(0,max(1,len(xs)))
    ax.tick_params(colors='#334455',labelsize=7)
    ax.set_ylabel("Vol",color='#556677',fontsize=8)
    for sp in ax.spines.values(): sp.set_color('#1e3a5a')
    plt.tight_layout(pad=0.2); return fig

# ================================================================ AUDIO CB
def audio_cb(indata,frames,t,status):
    data=indata[:,0] if indata.ndim>1 else indata.flatten()
    rms=float(np.sqrt(np.mean(data**2)))
    G["energy_ring"].append(rms)
    with G["lock"]:
        G["speaking"]=rms>SPK_THR
        G["volume"]=int(min(rms*1000,99))
        G["pace"]=pace_detect(G["energy_ring"])
        G["v_hist"].append(G["volume"])
        if len(G["v_hist"])>400: G["v_hist"]=G["v_hist"][-300:]
        G["audio_chunks"].append(data.astype(np.float32).copy())
        if len(G["audio_chunks"])>600: G["audio_chunks"]=G["audio_chunks"][-400:]
        G["full_audio"].append(data.astype(np.float32).copy())

# ================================================================ CAMERA THREAD
def camera_loop():
    fm=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,
        min_detection_confidence=0.5,min_tracking_confidence=0.5)
    pose=mp.solutions.pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if not cap.isOpened(): cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FPS,15)

    rec_path=os.path.join(tempfile.gettempdir(),
        f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    recorder=cv2.VideoWriter(rec_path,fourcc,10,(640,480))
    with G["lock"]:
        G["rec_path"]=rec_path
        G["recorder"]=recorder

    n=0
    while G["running"]:
        ret,frame=cap.read()
        if not ret: time.sleep(0.05); continue
        frame=cv2.flip(frame,1); n+=1

        if n%4==0:
            try:
                sm=cv2.resize(frame,(320,240))
                ft=extract_features(sm,fm,pose)
                X=np.array([[ft[f] for f in feature_names]])
                pe=model.predict(X)[0]; pr=model.predict_proba(X)[0]
                eye,tilt,sh,face,mo=compute_metrics(sm,fm,pose)
                with G["lock"]: sp=G["speaking"]
                sc=calc_score(face,eye,tilt,sh,sp)
                emo=detect_emotion(mo,eye,tilt)
                with G["lock"]:
                    G["status"]=le.inverse_transform([pe])[0]
                    G["confidence"]=float(max(pr)); G["score"]=sc
                    G["eye_contact"]=eye; G["head_tilt"]=tilt
                    G["shoulder_ok"]=sh; G["face_detected"]=face
                    G["emotion"]=emo; G["mouth_open"]=mo
                    G["score_hist"].append(sc); G["eye_hist"].append(eye)
                    if len(G["score_hist"])>600: G["score_hist"]=G["score_hist"][-400:]
                    if len(G["eye_hist"])>600:   G["eye_hist"]=G["eye_hist"][-400:]
            except: pass

        with G["lock"]: sc_=G["score"]; sp=G["speaking"]
        col=(40,255,120) if sc_>=80 else (0,220,255) if sc_>=60 else (60,80,255)
        cv2.rectangle(frame,(0,0),(640,46),(8,14,24),-1)
        cv2.putText(frame,f"Score: {sc_}/100",(12,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2)
        cv2.putText(frame,"SPEAKING" if sp else "SILENT",(460,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0) if sp else (80,80,200),2)
        recorder.write(frame)   # ← CORRECT: cv2.VideoWriter uses .write() not .write_frame()
        _,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,75])
        with G["lock"]: G["frame_jpg"]=buf.tobytes()
        time.sleep(0.06)

    cap.release(); fm.close(); pose.close()
    with G["lock"]:
        if G["recorder"]: G["recorder"].release(); G["recorder"]=None

# ================================================================ SPEECH THREAD
def speech_recognition_loop():
    recognizer=sr.Recognizer()
    recognizer.energy_threshold=300; recognizer.dynamic_energy_threshold=True
    with G["lock"]: G["sr_status"]="listening"

    while G["running"]:
        time.sleep(CHUNK_SEC)
        if not G["running"]: break
        with G["lock"]:
            chunks=list(G["audio_chunks"]); G["audio_chunks"]=[]
        if not chunks: continue
        try:
            audio=np.concatenate(chunks).flatten().astype(np.float32)
            if len(audio)<SAMPLE_RATE: continue
            audio=np.clip(audio*5.0,-1.0,1.0)
            with G["lock"]: G["sr_status"]="processing"
            fname=os.path.join(tempfile.gettempdir(),"sr_in.wav")
            i16=(audio*32767).astype(np.int16)
            with wave.open(fname,"w") as wf:
                wf.setnchannels(1); wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE); wf.writeframes(i16.tobytes())
            with sr.AudioFile(fname) as src: aud=recognizer.record(src)
            try:
                txt=recognizer.recognize_google(aud,language="en-IN")
                if txt and len(txt)>2:
                    with G["lock"]:
                        prev=G["transcript"]; G["transcript"]=(prev+" "+txt).strip() if prev else txt
                    G["sr_status"]="done"
            except sr.UnknownValueError: G["sr_status"]="listening"
            except sr.RequestError:      G["sr_status"]="offline"
            try: os.unlink(fname)
            except: pass
        except: G["sr_status"]="listening"

# ================================================================ START / STOP
def start_session():
    with G["lock"]:
        G["running"]=True; G["transcript"]=""; G["audio_chunks"]=[]
        G["sr_status"]="listening"; G["score_hist"]=[]; G["eye_hist"]=[]
        G["v_hist"]=[]; G["frame_jpg"]=None; G["score"]=0
        G["full_audio"]=[]
    threading.Thread(target=camera_loop,daemon=True).start()
    threading.Thread(target=speech_recognition_loop,daemon=True).start()
    try:
        s=sd.InputStream(callback=audio_cb,channels=1,samplerate=SAMPLE_RATE,blocksize=4096)
        s.start(); G["_s"]=s
    except Exception as e:
        st.warning(f"Mic error: {e}")

def stop_session():
    G["running"] = False

    # Stop mic stream
    try:
        if G.get("_s"): G["_s"].stop()
    except: pass

    # Save full audio to WAV
    audio_path = None
    try:
        if G["full_audio"]:
            audio = np.concatenate(G["full_audio"]).flatten()
            audio = np.clip(audio, -1.0, 1.0)
            audio_path = os.path.join(tempfile.gettempdir(), "final_audio.wav")
            i16 = (audio * 32767).astype(np.int16)
            with wave.open(audio_path, "w") as wf:
                wf.setnchannels(1); wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE); wf.writeframes(i16.tobytes())
    except Exception as e:
        print("Audio save error:", e)

    # Merge video + audio using ffmpeg
    try:
        video_path = G.get("rec_path", "")
        if video_path and os.path.exists(video_path) and audio_path and os.path.exists(audio_path):
            output_path = video_path.replace(".avi", "_final.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(output_path):
                G["rec_path"] = output_path  # point to merged MP4
            # else keep original .avi (no audio)
    except Exception as e:
        print("Merge error:", e)

# ================================================================ HEADER
st.markdown("""
<p style='font-family:Orbitron,monospace;font-size:11px;letter-spacing:5px;
color:#00e5ff;padding:6px 0;border-bottom:1px solid #1e3a5a;margin-bottom:10px'>
🎤 AI PRESENTATION COACH &nbsp;·&nbsp; SILVER OAK UNIVERSITY &nbsp;·&nbsp; ADVANCED v3.0
</p>""", unsafe_allow_html=True)

tab1,tab2,tab3=st.tabs(["📹  Live Coaching","📊  Session Report","📁  Upload & Analyze"])

# ================================================================ TAB 1 — LIVE
with tab1:
    b1,b2,b3,_=st.columns([1.4,1,1.6,3])
    with b1: start_btn=st.button("▶ START PRESENTATION")
    with b2: stop_btn=st.button("⬛ STOP")
    with b3: rep_btn=st.button("📊 SESSION REPORT")

    if start_btn and not G["running"]:
        st.session_state.t_start=time.time()
        st.session_state.last_append=0.0
        st.session_state.history=[]
        start_session(); st.rerun()

    if stop_btn and G["running"]:
        stop_session(); st.rerun()   # ← no freeze

    st.markdown("🎙 **Microphone**")
    try:
        devs=sd.query_devices()
        mics=["Auto (recommended)"]+[f"[{i}] {d['name'][:45]}"
              for i,d in enumerate(devs) if d['max_input_channels']>0]
    except: mics=["Auto (recommended)"]
    mic=st.selectbox("",mics,label_visibility="collapsed")
    run=G["running"]
    st.markdown(f"<p style='font-size:12px;color:{'#00ff88' if run else '#445566'};margin:2px 0 6px'>"
                f"{'🟢 Session active — all systems running' if run else f'⚪ Ready · {mic}'}</p>",
                unsafe_allow_html=True)

    with st.expander("🔧 Mic Test + Features"):
        ca,cb=st.columns(2)
        with ca:
            if st.button("🔍 Test Mic (2s)"):
                try:
                    a=sd.rec(int(2*SAMPLE_RATE),samplerate=SAMPLE_RATE,channels=1); sd.wait()
                    rms=float(np.sqrt(np.mean(a**2)))
                    if rms>.003:    st.success(f"✅ Working — RMS:{rms:.5f}")
                    elif rms>.0005: st.warning(f"⚠️ Quiet — RMS:{rms:.5f} · Increase mic vol in Windows")
                    else:           st.error(f"❌ No audio — RMS:{rms:.5f} · Check Sound Settings")
                except Exception as ex: st.error(f"❌ {ex}")
        with cb:
            st.markdown("""**v3.0 Features:**
- ✅ Face, eye contact, head tilt
- ✅ Shoulder balance
- ✅ 😊 Emotion detection
- ✅ Voice vol + pace
- ✅ Speech transcription (Google)
- ✅ Filler word detection
- ✅ Vocabulary richness
- ✅ WPM calculation
- ✅ English Grade A+ to D
- ✅ Grammar checks
- ✅ Score timeline chart
- ✅ Voice volume chart
- ✅ Radar performance chart
- ✅ Session recording + download
- ✅ Upload video analysis""")

    st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)

    col_v,col_m,col_f=st.columns([1.3,1.15,1.15])
    with col_v:  cam_ph=st.empty()
    with col_m:  sc_ph=st.empty(); met_ph=st.empty(); sp_ph=st.empty()
    with col_f:  fb_ph=st.empty(); en_ph=st.empty()

    st.markdown("<hr style='border-color:#1e3a5a;margin:8px 0'>",unsafe_allow_html=True)
    sum_ph=st.empty(); tl_ph=st.empty(); vl_ph=st.empty(); tr_ph=st.empty()

    en_ph.markdown("""
<div class="card">
  <span class="sh">🧍 &nbsp;POSTURE CHECKLIST</span>
  <div class="ci"><span class="dot"></span> 👁 Look directly at camera</div>
  <div class="ci"><span class="dot"></span> 📐 Head vertical (&lt;10°)</div>
  <div class="ci"><span class="dot"></span> ⚖️ Level shoulders</div>
  <div class="ci"><span class="dot"></span> 🧍 Body visible in frame</div>
  <div class="ci"><span class="dot"></span> 🎵 Vary speaking pace</div>
  <div class="ci" style="border:none"><span class="dot"></span> 👐 Use hand gestures</div>
</div>""",unsafe_allow_html=True)

    # ── RENDER FUNCTION ──────────────────────────────────────────
    def render(sc,eye,tilt,sh,face,spk,vol,pac,jpg,trans,sr_s,s_hist,v_hist,elapsed,emotion):
        fct,fw,vr,nw=analyze_speech(trans)
        t0=st.session_state.get("t_start") or time.time()
        wpm=min(int(nw/max((time.time()-t0)/60,0.1)),300) if nw>0 else 0
        spk_pct=int(sum(1 for v in v_hist if v>3)/max(len(v_hist),1)*100)

        # Camera
        if jpg: cam_ph.image(jpg,channels="BGR",use_container_width=True)
        else:
            cam_ph.markdown("""
<div style="background:#0a1220;border:1px dashed #1e3a5a;border-radius:12px;
height:300px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px">
  <div style="font-size:44px;opacity:.3">📷</div>
  <div style="font-size:11px;letter-spacing:4px;color:#2a4a6a">CAMERA STANDBY</div>
  <div style="font-size:11px;color:#445566">Click ▶ START PRESENTATION</div>
</div>""",unsafe_allow_html=True)

        lbl,c=score_lbl(sc)
        vc="#00ff88" if spk else "#ff4455"
        vi="🎤" if spk else "🔇"
        vt=f"SPEAKING · vol {vol} · pace:{pac}" if spk else "SILENT · silence 100%"
        sc_ph.markdown(f"""
<div class="card">
  <span class="sh">🏆 &nbsp;CONFIDENCE SCORE</span>
  <div class="snum" style="color:{c}">{sc}</div>
  <p class="ssub">OUT OF 100</p>
  <div class="pbg"><div class="pfl" style="width:{sc}%;background:linear-gradient(90deg,{c},{c}aa)"></div></div>
  <div class="bdg" style="color:{c};border-color:{c}55">{lbl}</div>
  <div class="vbar" style="background:{'#0a1a0a' if spk else '#1a0a0a'};color:{vc};border:1px solid {vc}33">
    {vi} &nbsp;{vt}</div>
  <p style="font-size:11px;color:#445566;margin:6px 0 0">{emotion} &nbsp;·&nbsp; 🗣 Speaking {spk_pct}%</p>
</div>""",unsafe_allow_html=True)

        shv="✅" if sh else "❌"
        met_ph.markdown(f"""
<div class="card">
  <span class="sh">📊 &nbsp;LIVE METRICS</span>
  <div class="mg">
    <div class="mb"><div class="mv" style="color:#00aaff">{eye:.0%}</div><div class="ml">EYE CONTACT</div></div>
    <div class="mb"><div class="mv" style="color:#ffcc00">{tilt:.0f}°</div><div class="ml">HEAD TILT</div></div>
    <div class="mb"><div class="mv" style="font-size:20px">{shv}</div><div class="ml">SHOULDERS</div></div>
    <div class="mb"><div class="mv" style="color:#ff6633">{vol}</div><div class="ml">VOICE VOL</div></div>
  </div>
  <div class="mg" style="margin-top:8px">
    <div class="mb"><div class="mv" style="color:#00ff88;font-size:18px">{wpm}</div><div class="ml">WPM</div></div>
    <div class="mb"><div class="mv" style="color:#00e5ff;font-size:18px">{nw}</div><div class="ml">WORDS</div></div>
  </div>
</div>""",unsafe_allow_html=True)

        sr_map={"idle":("⏳","#445566","Waiting..."),"listening":("🎤","#00aaff","Listening — speak now"),
                "processing":("⚙️","#ffcc00","Transcribing..."),"done":("✅","#00ff88","Updated"),
                "offline":("🌐","#ff4455","No internet"),"error":("❌","#ff4455","Error")}
        si,sc2,st2=sr_map.get(sr_s,("🎤","#445566","..."))
        fwh="".join(f'<span class="tag">{w}</span>' for w in set(fw[:6])) if fw \
            else '<span class="tag-g">None detected ✓</span>'
        sp_ph.markdown(f"""
<div class="card">
  <span class="sh">🗣 &nbsp;SPEECH ANALYSIS</span>
  <p style="font-size:11px;color:{sc2};margin:0 0 8px">{si} {st2}</p>
  <div class="mg">
    <div class="mb"><div class="mv" style="color:#ff6633">{fct}</div><div class="ml">FILLERS</div></div>
    <div class="mb"><div class="mv" style="color:#00aaff">{vr:.0%}</div><div class="ml">VOCAB RICH</div></div>
  </div>
  <p style="font-size:10px;color:#445566;margin:6px 0 3px;letter-spacing:2px">FILLER WORDS</p>
  <div>{fwh}</div>
</div>""",unsafe_allow_html=True)

        items=get_feedback(face,eye,tilt,sh,spk,vol,pac,fct,vr,wpm,emotion)
        ih="".join(f'<div class="fi" style="border-color:{co}"><span style="font-size:13px">{ic}</span>'
                   f'<span style="color:#ddeeff">{tx}</span></div>' for ic,co,tx in items)
        fb_ph.markdown(f'<div class="card"><span class="sh">💡 &nbsp;REAL-TIME FEEDBACK</span>{ih}</div>',
                       unsafe_allow_html=True)

        eng=analyze_english(trans)
        if eng:
            tips=[]
            if eng["fr"]==0:    tips.append(("✅","#00ff88","Perfect — zero fillers!"))
            elif eng["fr"]<=3:  tips.append(("🗣","#00e5ff",f"Good fluency ({eng['fr']:.1f}%)"))
            elif eng["fr"]<=8:  tips.append(("🗣","#ffcc00",f"Reduce fillers ({eng['fr']:.1f}%)"))
            else:                tips.append(("🗣","#ff4455",f"Too many fillers ({eng['fr']:.1f}%)"))
            if eng["vr"]>=.7:   tips.append(("📚","#00ff88",f"Rich vocab ({eng['vr']:.0%} unique)"))
            elif eng["vr"]>=.5: tips.append(("📚","#ffcc00",f"Vocab: {eng['vr']:.0%}"))
            else:                tips.append(("📚","#ff4455",f"Limited vocab ({eng['vr']:.0%})"))
            if eng["asl"]>=8:   tips.append(("📝","#00ff88",f"Good sentences ({eng['asl']:.0f}w avg)"))
            else:                tips.append(("📝","#ffcc00",f"Short sentences ({eng['asl']:.0f}w avg)"))
            for iss in eng["issues"]: tips.append(("⚠️","#ffcc00",iss))
            th="".join(f'<div class="fi" style="border-color:{co}"><span>{ic}</span>'
                       f'<span style="color:#ddeeff">{tx}</span></div>' for ic,co,tx in tips)
            en_ph.markdown(f"""
<div class="card">
  <span class="sh">🇬🇧 &nbsp;ENGLISH FEEDBACK</span>
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
    <div style="font-family:Orbitron,monospace;font-size:34px;font-weight:900;color:{eng['gc']}">{eng['gr']}</div>
    <div>
      <p style="font-size:10px;color:#445566;margin:0">ENGLISH GRADE</p>
      <p style="font-size:12px;color:#c8d8e8;margin:2px 0">{eng['gm']}</p>
      <p style="font-size:10px;color:#445566;margin:0">{eng['nsent']} sentences · {eng['unique']} unique words</p>
    </div>
  </div>
  {th}
</div>""",unsafe_allow_html=True)

        if s_hist:
            avg_s=int(sum(s_hist)/len(s_hist)); pk=max(s_hist); lw=min(s_hist)
            m,s2=divmod(elapsed,60)
            sum_ph.markdown(f"""
<p class="sh">📋 &nbsp;SESSION SUMMARY &nbsp;·&nbsp; <span style="color:#00e5ff">{m}m {s2:02d}s</span></p>
<div class="sg">
  <div class="sb"><div class="sl">AVERAGE</div><div class="sv" style="color:{score_col(avg_s)}">{avg_s}</div></div>
  <div class="sb"><div class="sl">PEAK</div><div class="sv" style="color:{score_col(pk)}">{pk}</div></div>
  <div class="sb"><div class="sl">LOW</div><div class="sv" style="color:{score_col(lw)}">{lw}</div></div>
  <div class="sb"><div class="sl">SPEAKING</div><div class="sv" style="color:#00e5ff">{spk_pct}%</div></div>
</div>""",unsafe_allow_html=True)
            now=time.time()
            if len(s_hist)>5 and (now-st.session_state.last_chart)>=5:
                fig=make_timeline(s_hist)
                with tl_ph.container():
                    st.markdown("<p class='sh'>📈 &nbsp;SCORE TIMELINE</p>",unsafe_allow_html=True)
                    st.pyplot(fig,use_container_width=True); plt.close(fig)
                if len(v_hist)>10:
                    fig2=make_vol_bar(v_hist[-200:])
                    with vl_ph.container():
                        st.markdown("<p class='sh'>🔊 &nbsp;VOICE VOLUME HISTORY</p>",unsafe_allow_html=True)
                        st.pyplot(fig2,use_container_width=True); plt.close(fig2)
                st.session_state.last_chart=now

        if trans:
            tr_ph.markdown(f"""
<div class="card">
  <span class="sh">📝 &nbsp;LIVE TRANSCRIPT</span>
  <p style="font-size:12px;color:#aabbcc;line-height:1.9;margin:0">{' '.join(trans.split()[-80:])}</p>
</div>""",unsafe_allow_html=True)

    # ── SNAP ──────────────────────────────────────────────────────
    def snap():
        with G["lock"]:
            return (G["score"],G["eye_contact"],G["head_tilt"],G["shoulder_ok"],
                    G["face_detected"],G["speaking"],G["volume"],G["pace"],
                    G["frame_jpg"],G["transcript"],G["sr_status"],
                    list(G["score_hist"]),list(G["v_hist"]),G["emotion"])

    # ── LIVE LOOP ──────────────────────────────────────────────────
    if run:
        while True:
            sc,eye,tilt,sh,face,spk,vol,pac,jpg,trans,sr_s,s_hist,v_hist,emotion=snap()
            elapsed=int(time.time()-(st.session_state.t_start or time.time()))
            render(sc,eye,tilt,sh,face,spk,vol,pac,jpg,trans,sr_s,s_hist,v_hist,elapsed,emotion)
            now=time.time()
            if sc>0 and (now-st.session_state.last_append)>=1.0:
                st.session_state.history.append(sc)
                st.session_state.last_append=now
            time.sleep(0.3)
            if not G["running"]: st.rerun()   # ← clean stop, no freeze
    else:
        sc,eye,tilt,sh,face,spk,vol,pac,jpg,trans,sr_s,s_hist,v_hist,emotion=snap()
        elapsed=int(time.time()-(st.session_state.t_start or time.time()))
        render(sc,eye,tilt,sh,face,spk,vol,pac,jpg,trans,sr_s,s_hist,v_hist,elapsed,emotion)
        with G["lock"]: rp=G.get("rec_path","")
        if rp and os.path.exists(rp):
            try:
                with open(rp,"rb") as vf: vb=vf.read()
                st.markdown("<hr style='border-color:#1e3a5a;margin:10px 0'>",unsafe_allow_html=True)
                r1,r2=st.columns([3,1])
                with r1:
                    st.markdown("<p class='sh'>▶ &nbsp;REVIEW SESSION RECORDING</p>",unsafe_allow_html=True)
                    st.video(vb)
                with r2:
                    st.markdown("<br>",unsafe_allow_html=True)
                    ext  = "mp4" if rp.endswith(".mp4") else "avi"
                    mime = "video/mp4" if ext=="mp4" else "video/x-msvideo"
                    label = "⬇️ Download Recording (with audio)" if ext=="mp4" else "⬇️ Download Recording (video only)"
                    st.download_button(label, vb,
                        file_name=f"presentation_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.{ext}",
                        mime=mime, key="dl_rec")
                    if ext=="mp4":
                        st.markdown("<p style='font-size:11px;color:#00ff88;margin:4px 0'>✅ Audio included in recording</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<p style='font-size:11px;color:#ffcc00;margin:4px 0'>⚠️ No audio — install ffmpeg to enable</p>", unsafe_allow_html=True)
            except: pass

# ================================================================ TAB 2 — REPORT
with tab2:
    st.markdown("<p style='font-size:14px;letter-spacing:3px;color:#00e5ff'>📊 FULL SESSION REPORT</p>",
                unsafe_allow_html=True)
    with G["lock"]:
        s_hist=list(G["score_hist"]); trans=G["transcript"]
        eye_hist=list(G["eye_hist"]); v_hist=list(G["v_hist"])

    if not s_hist:
        st.markdown("""<div class="card" style="text-align:center;padding:40px">
  <p style="font-size:32px">📊</p>
  <p style="color:#445566">Complete a session first to see your report</p>
</div>""",unsafe_allow_html=True)
    else:
        avg_s=int(sum(s_hist)/len(s_hist)); pk=max(s_hist); lw=min(s_hist)
        t0=st.session_state.get("t_start") or time.time()
        el=int(time.time()-t0); m,s2=divmod(el,60)
        avg_eye=sum(eye_hist)/len(eye_hist) if eye_hist else 0
        spk_pct=int(sum(1 for v in v_hist if v>3)/max(len(v_hist),1)*100)
        fct,fw,vr,nw=analyze_speech(trans)
        wpm=min(int(nw/max(el/60,0.1)),300) if nw>0 else 0

        st.markdown(f"""
<div class="sg" style="margin-bottom:12px">
  <div class="sb"><div class="sl">AVERAGE SCORE</div><div class="sv" style="color:{score_col(avg_s)}">{avg_s}/100</div></div>
  <div class="sb"><div class="sl">PEAK SCORE</div><div class="sv" style="color:{score_col(pk)}">{pk}/100</div></div>
  <div class="sb"><div class="sl">DURATION</div><div class="sv" style="color:#00e5ff">{m}m {s2:02d}s</div></div>
  <div class="sb"><div class="sl">SPEAKING TIME</div><div class="sv" style="color:#00ff88">{spk_pct}%</div></div>
</div>
<div class="sg2" style="margin-bottom:16px">
  <div class="sb"><div class="sl">AVG EYE CONTACT</div><div class="sv" style="color:#00aaff">{avg_eye:.0%}</div></div>
  <div class="sb"><div class="sl">TOTAL WORDS</div><div class="sv" style="color:#ffcc00">{nw}</div></div>
</div>""",unsafe_allow_html=True)

        c1,c2=st.columns(2)
        with c1:
            fw_c=Counter(fw).most_common(8)
            fwh="".join(f'<span class="tag">{w}({n}x)</span>' for w,n in fw_c) if fw_c \
                else '<span class="tag-g">None detected ✓</span>'
            st.markdown(f"""
<div class="card">
  <span class="sh">🗣 &nbsp;SPEECH REPORT</span>
  <div class="mg">
    <div class="mb"><div class="mv" style="color:#ff6633">{fct}</div><div class="ml">FILLERS</div></div>
    <div class="mb"><div class="mv" style="color:#00aaff">{vr:.0%}</div><div class="ml">VOCAB RICH</div></div>
    <div class="mb"><div class="mv" style="color:#00ff88">{wpm}</div><div class="ml">AVG WPM</div></div>
    <div class="mb"><div class="mv" style="color:#ffcc00">{nw}</div><div class="ml">TOTAL WORDS</div></div>
  </div>
  <p style="font-size:10px;color:#445566;margin:6px 0 3px">FILLER WORDS USED</p>
  <div>{fwh}</div>
</div>""",unsafe_allow_html=True)

            eng=analyze_english(trans)
            if eng:
                st.markdown(f"""
<div class="card">
  <span class="sh">🇬🇧 &nbsp;ENGLISH ANALYSIS</span>
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px">
    <div style="font-family:Orbitron,monospace;font-size:42px;font-weight:900;color:{eng['gc']}">{eng['gr']}</div>
    <div>
      <p style="font-size:11px;color:#445566;margin:0">ENGLISH GRADE</p>
      <p style="font-size:13px;color:#c8d8e8;margin:2px 0;font-weight:bold">{eng['gm']}</p>
      <p style="font-size:11px;color:#445566;margin:0">Filler rate: {eng['fr']}% · Avg sentence: {eng['asl']} words</p>
      <p style="font-size:11px;color:#445566;margin:0">{eng['nsent']} sentences · {eng['unique']} unique words</p>
    </div>
  </div>
</div>""",unsafe_allow_html=True)

            if trans:
                st.markdown("""<div class="card"><span class="sh">📝 &nbsp;FULL TRANSCRIPT</span>""",
                            unsafe_allow_html=True)
                st.text_area("",trans,height=130,label_visibility="collapsed")
                st.markdown("</div>",unsafe_allow_html=True)

        with c2:
            if len(s_hist)>5:
                fig=make_timeline(s_hist)
                st.markdown("<p class='sh'>📈 &nbsp;SCORE TIMELINE</p>",unsafe_allow_html=True)
                st.pyplot(fig,use_container_width=True); plt.close(fig)

            tilt_sc=100 if avg_eye>=.6 else 50
            fluency=max(0,100-fct*5)
            wpm_sc=100 if WPM_MIN<=wpm<=WPM_MAX else 60 if wpm>0 else 0
            avg_vol=sum(v_hist)/len(v_hist) if v_hist else 0
            fig_r=make_radar(avg_eye,tilt_sc,100,fluency,min(avg_vol*2,100),wpm_sc)
            st.markdown("<p class='sh'>🕸 &nbsp;PERFORMANCE RADAR</p>",unsafe_allow_html=True)
            st.pyplot(fig_r,use_container_width=True); plt.close(fig_r)

            if len(v_hist)>10:
                fig_v=make_vol_bar(v_hist[-200:])
                st.markdown("<p class='sh'>🔊 &nbsp;VOICE VOLUME HISTORY</p>",unsafe_allow_html=True)
                st.pyplot(fig_v,use_container_width=True); plt.close(fig_v)

            recs=[]
            if avg_s<60:            recs.append(("📐","Practice posture — sit straight, camera at eye level"))
            if fct>5:               recs.append(("🗣","Replace fillers with deliberate pauses"))
            if vr<.5 and nw>0:      recs.append(("📚","Expand vocabulary — use synonyms, read more"))
            if wpm>WPM_MAX:         recs.append(("⏱","Slow down — aim 120-150 wpm"))
            if wpm<WPM_MIN and wpm>0: recs.append(("⏱","Speak more confidently — 120-150 wpm"))
            if avg_eye<.6:          recs.append(("👁","Look at camera lens, not the screen"))
            if spk_pct<40:          recs.append(("🎤","Speak more — fill silence with content"))
            if not recs:            recs.append(("🏆","Excellent session — maintain this quality!"))
            rh="".join(f'<div class="fi" style="border-color:#00aaff"><span>{ic}</span>'
                       f'<span style="color:#ddeeff">{tx}</span></div>' for ic,tx in recs)
            st.markdown(f'<div class="card"><span class="sh">🎯 &nbsp;IMPROVEMENT TIPS</span>{rh}</div>',
                        unsafe_allow_html=True)

# ================================================================ TAB 3 — UPLOAD & ANALYZE
def analyze_video_file(video_path):
    fm=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,
        min_detection_confidence=0.5,min_tracking_confidence=0.5)
    pose=mp.solutions.pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
    cap=cv2.VideoCapture(video_path)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps=cap.get(cv2.CAP_PROP_FPS) or 10
    scores=[]; eyes=[]; tilts=[]; sh_list=[]; face_list=[]
    n=0
    prog=st.progress(0,text="Analyzing video...")
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret: break
        n+=1
        if n%5!=0: continue
        try:
            sm=cv2.resize(frame,(320,240))
            eye,tilt,sh,face,mo=compute_metrics(sm,fm,pose)
            sc=calc_score(face,eye,tilt,sh,False)
            scores.append(sc); eyes.append(eye)
            tilts.append(tilt); sh_list.append(sh); face_list.append(face)
        except: pass
        prog.progress(min(n/total,1.0),text=f"Analyzing frame {n}/{total}...")
    prog.empty(); cap.release(); fm.close(); pose.close()
    return scores,eyes,tilts,sh_list,face_list,total,fps

with tab3:
    st.markdown("""<div class="card" style="text-align:center;padding:24px">
  <p style="font-size:32px">📁</p>
  <p style="font-family:Orbitron,monospace;color:#00e5ff;letter-spacing:3px;font-size:13px">
    UPLOAD VIDEO & ANALYZE</p>
  <p style="color:#445566;font-size:12px;margin-top:6px">
    Upload your recorded presentation — full posture + eye contact AI analysis</p>
</div>""",unsafe_allow_html=True)

    up=st.file_uploader("Upload video (mp4, avi, mov)",type=["mp4","avi","mov"])
    if up:
        suffix="."+up.name.split(".")[-1]
        tmp=tempfile.NamedTemporaryFile(delete=False,suffix=suffix)
        tmp.write(up.read()); tmp.flush(); tmp.close()
        vid_path=tmp.name

        st.markdown("<p class='sh'>▶ &nbsp;VIDEO PLAYBACK</p>",unsafe_allow_html=True)
        with open(vid_path,"rb") as vf: st.video(vf.read())
        st.markdown("<hr style='border-color:#1e3a5a;margin:10px 0'>",unsafe_allow_html=True)

        if st.button("🔍 ANALYZE THIS VIDEO",key="analyze_vid"):

            # ── VISION ANALYSIS ─────────────────────────────────
            with st.spinner("Analyzing posture & face..."):
                scores,eyes,tilts,sh_list,face_list,total_frames,fps=analyze_video_file(vid_path)

            # ── AUDIO ANALYSIS ──────────────────────────────────
            vid_transcript = ""; vid_fct=0; vid_fw=[]; vid_vr=0.0; vid_nw=0
            vid_wpm=0; has_audio=False

            audio_wav_tmp = os.path.join(tempfile.gettempdir(),"upload_audio.wav")
            audio_status  = st.empty()

            try:
                # Try ffmpeg extract audio from video
                cmd = ["ffmpeg","-y","-i",vid_path,
                       "-vn","-ac","1","-ar","16000","-f","wav", audio_wav_tmp]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode==0 and os.path.exists(audio_wav_tmp) and os.path.getsize(audio_wav_tmp)>4096:
                    has_audio = True
                    audio_status.markdown("<p style='color:#ffcc00;font-size:12px'>🎤 Audio found — transcribing...</p>",
                                          unsafe_allow_html=True)

                    # Transcribe in 10-second chunks via Google SR
                    recognizer = sr.Recognizer()
                    recognizer.energy_threshold=300; recognizer.dynamic_energy_threshold=True
                    full_text = []

                    with sr.AudioFile(audio_wav_tmp) as source:
                        audio_len = source.DURATION
                        chunk_sec = 10
                        offset = 0
                        while offset < audio_len:
                            try:
                                aud = recognizer.record(source, duration=min(chunk_sec, audio_len-offset))
                                txt = recognizer.recognize_google(aud, language="en-IN")
                                if txt: full_text.append(txt)
                            except sr.UnknownValueError: pass
                            except sr.RequestError:
                                audio_status.markdown("<p style='color:#ff4455;font-size:12px'>🌐 No internet — Google SR needs connection</p>",unsafe_allow_html=True)
                                break
                            offset += chunk_sec

                    vid_transcript = " ".join(full_text).strip()
                    if vid_transcript:
                        vid_fct,vid_fw,vid_vr,vid_nw = analyze_speech(vid_transcript)
                        duration_sec = int(total_frames/max(fps,1))
                        vid_wpm = min(int(vid_nw/max(duration_sec/60,0.1)),300) if vid_nw>0 else 0
                        audio_status.markdown("<p style='color:#00ff88;font-size:12px'>✅ Transcription complete</p>",unsafe_allow_html=True)
                    else:
                        audio_status.markdown("<p style='color:#ffcc00;font-size:12px'>🔇 Audio found but no speech detected</p>",unsafe_allow_html=True)
                else:
                    audio_status.markdown("<p style='color:#445566;font-size:12px'>🔇 No audio stream in video</p>",unsafe_allow_html=True)
            except FileNotFoundError:
                audio_status.markdown("<p style='color:#445566;font-size:12px'>ℹ️ ffmpeg not installed — audio analysis skipped</p>",unsafe_allow_html=True)
            except Exception as ex:
                audio_status.markdown(f"<p style='color:#ff4455;font-size:12px'>Audio error: {ex}</p>",unsafe_allow_html=True)

            if not scores:
                st.error("❌ No frames analyzed — try a different video")
            else:
                duration=int(total_frames/max(fps,1))
                avg_sc=int(sum(scores)/len(scores)); pk_sc=max(scores)
                avg_eye=sum(eyes)/len(eyes); avg_tilt=sum(tilts)/len(tilts)
                sh_pct=int(sum(sh_list)/len(sh_list)*100)
                face_pct=int(sum(face_list)/len(face_list)*100)
                m,s2=divmod(duration,60)

                st.markdown(f"""
<p class="sh">📊 &nbsp;VIDEO ANALYSIS RESULTS</p>
<div class="sg" style="margin-bottom:12px">
  <div class="sb"><div class="sl">AVG SCORE</div><div class="sv" style="color:{score_col(avg_sc)}">{avg_sc}/100</div></div>
  <div class="sb"><div class="sl">PEAK SCORE</div><div class="sv" style="color:{score_col(pk_sc)}">{pk_sc}/100</div></div>
  <div class="sb"><div class="sl">DURATION</div><div class="sv" style="color:#00e5ff">{m}m {s2:02d}s</div></div>
  <div class="sb"><div class="sl">FACE DETECTED</div><div class="sv" style="color:#00ff88">{face_pct}%</div></div>
</div>
<div class="sg" style="margin-bottom:16px">
  <div class="sb"><div class="sl">EYE CONTACT</div><div class="sv" style="color:#00aaff">{avg_eye:.0%}</div></div>
  <div class="sb"><div class="sl">AVG HEAD TILT</div><div class="sv" style="color:#ffcc00">{avg_tilt:.1f}°</div></div>
  <div class="sb"><div class="sl">SHOULDERS OK</div><div class="sv" style="color:#00ff88">{sh_pct}%</div></div>
  <div class="sb"><div class="sl">WORDS SPOKEN</div><div class="sv" style="color:#445566">{vid_nw}</div></div>
</div>""",unsafe_allow_html=True)

                # Vision charts
                v1,v2=st.columns(2)
                with v1:
                    if len(scores)>5:
                        fig=make_timeline(scores)
                        st.markdown("<p class='sh'>📈 &nbsp;SCORE TIMELINE</p>",unsafe_allow_html=True)
                        st.pyplot(fig,use_container_width=True); plt.close(fig)
                with v2:
                    fluency_sc=max(0,100-vid_fct*5) if has_audio else 70
                    wpm_sc=100 if WPM_MIN<=vid_wpm<=WPM_MAX else 60 if vid_wpm>0 else 50
                    fig_r=make_radar(avg_eye,100 if avg_tilt<10 else 50,sh_pct,fluency_sc,70,wpm_sc)
                    st.markdown("<p class='sh'>🕸 &nbsp;PERFORMANCE RADAR</p>",unsafe_allow_html=True)
                    st.pyplot(fig_r,use_container_width=True); plt.close(fig_r)

                # ── SPEECH ANALYSIS SECTION ─────────────────────
                if vid_transcript:
                    fw_c=Counter(vid_fw).most_common(6)
                    fwh="".join(f'<span class="tag">{w}({n}x)</span>' for w,n in fw_c) if fw_c                         else '<span class="tag-g">None detected ✓</span>'
                    st.markdown(f"""
<div class="card">
  <span class="sh">🗣 &nbsp;SPEECH ANALYSIS FROM VIDEO</span>
  <div class="mg">
    <div class="mb"><div class="mv" style="color:#ff6633">{vid_fct}</div><div class="ml">FILLERS</div></div>
    <div class="mb"><div class="mv" style="color:#00aaff">{vid_vr:.0%}</div><div class="ml">VOCAB RICH</div></div>
    <div class="mb"><div class="mv" style="color:#00ff88">{vid_wpm}</div><div class="ml">WPM</div></div>
    <div class="mb"><div class="mv" style="color:#ffcc00">{vid_nw}</div><div class="ml">WORDS</div></div>
  </div>
  <p style="font-size:10px;color:#445566;margin:6px 0 3px">FILLER WORDS</p>
  <div>{fwh}</div>
</div>""",unsafe_allow_html=True)

                    # English grade
                    eng=analyze_english(vid_transcript)
                    if eng:
                        st.markdown(f"""
<div class="card">
  <span class="sh">🇬🇧 &nbsp;ENGLISH ANALYSIS</span>
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px">
    <div style="font-family:Orbitron,monospace;font-size:42px;font-weight:900;color:{eng['gc']}">{eng['gr']}</div>
    <div>
      <p style="font-size:11px;color:#445566;margin:0">ENGLISH GRADE</p>
      <p style="font-size:13px;color:#c8d8e8;margin:2px 0;font-weight:bold">{eng['gm']}</p>
      <p style="font-size:11px;color:#445566;margin:0">Filler rate: {eng['fr']}% · Avg sentence: {eng['asl']} words</p>
    </div>
  </div>
</div>""",unsafe_allow_html=True)

                    # Transcript
                    st.markdown("""<div class="card"><span class="sh">📝 &nbsp;TRANSCRIPT FROM VIDEO AUDIO</span>""",
                                unsafe_allow_html=True)
                    st.text_area("",vid_transcript,height=120,label_visibility="collapsed",key="vid_trans")
                    st.markdown("</div>",unsafe_allow_html=True)

                # Posture feedback
                recs=[]
                if avg_sc<60:    recs.append(("📐","Posture needs work — sit straight"))
                if avg_eye<.6:   recs.append(("👁","Improve eye contact — look at camera lens"))
                if avg_tilt>10:  recs.append(("📐",f"Head tilted avg {avg_tilt:.1f}° — keep straight"))
                if sh_pct<80:    recs.append(("💪","Balance shoulders more consistently"))
                if face_pct<80:  recs.append(("🚫","Stay in frame — face not detected some frames"))
                if vid_fct>5:    recs.append(("🗣","Reduce filler words — use deliberate pauses"))
                if vid_wpm>WPM_MAX: recs.append(("⏱",f"Speaking too fast ({vid_wpm} wpm) — slow down"))
                if vid_wpm<WPM_MIN and vid_wpm>0: recs.append(("⏱","Speak more confidently"))
                if not recs:     recs.append(("🏆","Excellent presentation!"))
                rh="".join(f'<div class="fi" style="border-color:#00aaff"><span>{ic}</span>'
                           f'<span style="color:#ddeeff">{tx}</span></div>' for ic,tx in recs)
                st.markdown(f'<div class="card"><span class="sh">🎯 &nbsp;FEEDBACK & TIPS</span>{rh}</div>',
                            unsafe_allow_html=True)

                # Audio availability note
                if not has_audio:
                    st.markdown("""
<div class="card" style="border-color:#ffcc0044">
  <span class="sh">ℹ️ &nbsp;AUDIO INFO</span>
  <p style="font-size:12px;color:#ffcc00;margin:4px 0">
    No audio stream found in this video.</p>
  <p style="font-size:12px;color:#aabbcc;margin:4px 0">
    To get speech analysis: record with your phone/external camera (has audio),
    or use Live Coaching tab for real-time speech analysis.</p>
</div>""",unsafe_allow_html=True)
                elif not vid_transcript:
                    st.markdown("""
<div class="card" style="border-color:#ffcc0044">
  <span class="sh">ℹ️ &nbsp;SPEECH NOTE</span>
  <p style="font-size:12px;color:#ffcc00;margin:4px 0">Audio detected but no speech recognized.</p>
  <p style="font-size:12px;color:#aabbcc;margin:4px 0">Make sure internet is connected (Google SR needs it).
    Speak clearly and avoid background noise.</p>
</div>""",unsafe_allow_html=True)

            try:
                os.unlink(vid_path)
                if os.path.exists(audio_wav_tmp): os.unlink(audio_wav_tmp)
            except: pass

        else:
            try: os.unlink(vid_path)
            except: pass

st.markdown("<hr style='border-color:#1e3a5a;margin-top:16px'>"
            "<p style='text-align:center;color:#2a3a4a;font-size:10px;letter-spacing:2px'>"
            "AI PRESENTATION COACH · SILVER OAK UNIVERSITY · HEET JAIN (2201031000030) · 2026</p>",
            unsafe_allow_html=True)