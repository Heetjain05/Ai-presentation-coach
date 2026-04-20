"""
engine.py — Enhanced AI Presentation Coach Engine
Features:
  • Rich posture metrics  (face position, body frame, landmark details)
  • Speech analysis       (volume, pace, silence ratio, energy trend)
  • Confidence tracking   (rolling trend, momentum, consistency score)
  • CSV export            (full session log with all metrics)
"""

import cv2
import pickle
import numpy as np
import mediapipe as mp
import sounddevice as sd
import threading
import time
import math
import csv
import os
from collections import deque
from datetime import datetime


# ══════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════

def load_model(path='model.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['label_encoder'], data['feature_names']


# ══════════════════════════════════════════════
#  SPEECH ANALYSER  (runs in background thread)
# ══════════════════════════════════════════════

class SpeechAnalyser:
    """
    Continuously captures mic audio and computes:
      - speaking  : bool   (active voice detected)
      - volume    : float  0-100 (normalised RMS energy)
      - silence_ratio : float  0-1 (fraction of recent frames that were silent)
      - speech_pace   : str   "slow" | "normal" | "fast"  (burst frequency)
    """

    SAMPLE_RATE    = 16000
    CHUNK_DURATION = 0.15   # seconds per capture
    HISTORY_SIZE   = 40     # frames kept for trend analysis
    SPEAK_THRESH   = 15     # energy threshold for "speaking"

    def __init__(self):
        self.speaking      = False
        self.volume        = 0.0
        self.silence_ratio = 1.0
        self.speech_pace   = "normal"
        self._running      = False
        self._thread       = None
        self._energy_hist  = deque(maxlen=self.HISTORY_SIZE)
        self._speak_hist   = deque(maxlen=self.HISTORY_SIZE)   # bool bursts

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _worker(self):
        while self._running:
            try:
                audio = sd.rec(
                    int(self.CHUNK_DURATION * self.SAMPLE_RATE),
                    samplerate=self.SAMPLE_RATE, channels=1
                )
                sd.wait()
                energy = float(np.linalg.norm(audio))

                self._energy_hist.append(energy)
                self._speak_hist.append(energy > self.SPEAK_THRESH)

                self.speaking      = energy > self.SPEAK_THRESH
                # Normalise volume to 0-100 using soft cap at energy=150
                self.volume        = min(100.0, energy / 1.5)
                self.silence_ratio = 1 - (sum(self._speak_hist) / max(len(self._speak_hist), 1))
                self.speech_pace   = self._estimate_pace()

            except Exception:
                self.speaking = False
            time.sleep(0.05)

    def _estimate_pace(self):
        """Count speaking→silent transitions per second as a pace proxy."""
        if len(self._speak_hist) < 4:
            return "normal"
        transitions = sum(
            1 for i in range(1, len(self._speak_hist))
            if self._speak_hist[i] != self._speak_hist[i - 1]
        )
        rate = transitions / (len(self._speak_hist) * self.CHUNK_DURATION)
        if rate < 1.5:  return "slow"
        if rate > 4.5:  return "fast"
        return "normal"

    @property
    def summary(self):
        return {
            "speaking":      self.speaking,
            "volume":        round(self.volume, 1),
            "silence_ratio": round(self.silence_ratio, 2),
            "speech_pace":   self.speech_pace,
        }


# ══════════════════════════════════════════════
#  POSTURE EXTRACTOR  (frame-level)
# ══════════════════════════════════════════════

def extract_posture(img, face_mesh, pose):
    """
    Returns a rich dict of posture / body metrics.
    All values are floats or ints — safe to feed into the ML model.
    """
    h, w = img.shape[:2]
    features = {
        # Face presence
        "face_detected":      0,
        "face_center_x":      0.5,
        "face_center_y":      0.5,
        "face_size":          0.0,
        # Head orientation
        "head_tilt":          0.0,   # degrees left/right
        "head_nod":           0.0,   # degrees forward/back (pitch proxy)
        "head_distance":      0.5,   # relative depth proxy via face_size
        # Eye contact
        "eye_contact_score":  0.5,   # 0-1
        "gaze_offset_x":      0.0,   # how far from center
        # Mouth
        "mouth_open":         0.0,   # normalised opening
        # Body / posture
        "body_visible":       0,
        "shoulder_level":     0.0,   # |left_y - right_y|
        "shoulder_width":     0.0,   # |left_x - right_x|  (frame fill)
        "body_center_x":      0.5,   # torso horizontal center
        "lean_angle":         0.0,   # forward/back lean (hip-shoulder line)
    }

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Face Mesh ──
    face_res = face_mesh.process(rgb)
    if face_res.multi_face_landmarks:
        features["face_detected"] = 1
        lm = face_res.multi_face_landmarks[0].landmark

        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        features["face_center_x"] = cx
        features["face_center_y"] = cy
        features["face_size"]     = float((max(xs) - min(xs)) * (max(ys) - min(ys)))
        features["head_distance"] = features["face_size"]  # larger = closer

        # Head tilt (roll) — chin-to-nose horizontal offset
        dx = lm[152].x - lm[1].x
        dy = lm[152].y - lm[1].y
        features["head_tilt"] = abs(math.degrees(math.atan2(dx, dy + 1e-6)))

        # Head nod (pitch proxy) — nose tip vertical position relative to face bbox
        nose_y_norm = (lm[1].y - min(ys)) / max(max(ys) - min(ys), 1e-6)
        features["head_nod"] = abs(nose_y_norm - 0.45) * 100  # 0 = neutral

        # Eye contact: penalise horizontal offset from screen centre
        features["gaze_offset_x"]   = abs(cx - 0.5)
        features["eye_contact_score"] = max(0.0, 1.0 - features["gaze_offset_x"] * 3.5)

        # Mouth openness (landmark 13 = upper lip, 14 = lower lip)
        mouth_open = abs(lm[13].y - lm[14].y) / max(max(ys) - min(ys), 1e-6)
        features["mouth_open"] = float(mouth_open)

    # ── Pose ──
    pose_res = pose.process(rgb)
    if pose_res.pose_landmarks:
        features["body_visible"] = 1
        lm = pose_res.pose_landmarks.landmark

        l_sh, r_sh = lm[11], lm[12]   # shoulders
        l_hp, r_hp = lm[23], lm[24]   # hips

        features["shoulder_level"] = abs(l_sh.y - r_sh.y)
        features["shoulder_width"] = abs(l_sh.x - r_sh.x)
        features["body_center_x"]  = (l_sh.x + r_sh.x) / 2

        # Lean angle: vector from hip midpoint to shoulder midpoint
        mid_hip_y = (l_hp.y + r_hp.y) / 2
        mid_sh_y  = (l_sh.y + r_sh.y) / 2
        mid_hip_x = (l_hp.x + r_hp.x) / 2
        mid_sh_x  = (l_sh.x + r_sh.x) / 2
        dx = mid_sh_x - mid_hip_x
        dy = mid_hip_y - mid_sh_y   # inverted y
        features["lean_angle"] = abs(math.degrees(math.atan2(dx, dy + 1e-6)))

    return features


# ══════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════

def calculate_score(features, speech):
    """
    Weighted breakdown → max 100
      Face visible    : 15
      Eye contact     : 25
      Head straight   : 15  (tilt + nod)
      Shoulders level : 15
      Body framing    : 10  (width / lean)
      Speaking        : 10
      Voice quality   : 10  (volume + pace)
    """
    score = 0.0

    # Face
    if features.get("face_detected", 0):
        score += 15

    # Eye contact
    score += features.get("eye_contact_score", 0.5) * 25

    # Head orientation
    tilt_ok = max(0, 15 - features.get("head_tilt", 0)) / 15
    nod_ok  = max(0, 10 - features.get("head_nod", 0))  / 10
    score  += tilt_ok * 10 + nod_ok * 5

    # Shoulders
    sl = features.get("shoulder_level", 0)
    score += max(0, 15 - sl * 200)

    # Body framing
    sw   = features.get("shoulder_width", 0)
    lean = features.get("lean_angle", 0)
    framing  = min(sw * 80, 6)
    lean_pen = max(0, 4 - lean / 5)
    score += framing + lean_pen

    # Speaking
    if speech.get("speaking", False):
        score += 10

    # Voice quality
    vol  = speech.get("volume", 0)
    pace = speech.get("speech_pace", "normal")
    vol_score  = min(vol / 60, 1.0) * 6
    pace_score = 4 if pace == "normal" else 2
    score += vol_score + pace_score

    return min(100, int(score))


# ══════════════════════════════════════════════
#  FEEDBACK GENERATOR
# ══════════════════════════════════════════════

def generate_feedback(features, speech, score):
    """Returns list of (kind, icon, message) tuples."""
    tips = []

    # Face
    if not features.get("face_detected", 0):
        tips.append(("bad",  "🚫", "Face not in frame — move closer"))
    else:
        tips.append(("good", "✅", "Face clearly detected"))

    # Eye contact
    ec = features.get("eye_contact_score", 0.5)
    if ec >= 0.75:
        tips.append(("good", "👁️", f"Strong eye contact ({int(ec*100)}%)"))
    elif ec >= 0.45:
        tips.append(("warn", "👁️", f"Eye contact at {int(ec*100)}% — look at the lens"))
    else:
        tips.append(("bad",  "👁️", f"Eye contact low ({int(ec*100)}%) — face the camera"))

    # Head tilt
    tilt = features.get("head_tilt", 0)
    if tilt < 8:
        tips.append(("good", "🗣️", "Head level — great posture"))
    elif tilt < 18:
        tips.append(("warn", "↗️", f"Head tilted {tilt:.0f}° — try to straighten"))
    else:
        tips.append(("bad",  "↗️", f"Head tilted {tilt:.0f}° — keep it vertical"))

    # Head nod / pitch
    nod = features.get("head_nod", 0)
    if nod > 20:
        tips.append(("warn", "🔽", f"Head nodding too much ({nod:.0f}) — stay upright"))

    # Shoulders
    sl = features.get("shoulder_level", 0)
    if sl < 0.04:
        tips.append(("good", "💪", "Shoulders balanced"))
    elif sl < 0.09:
        tips.append(("warn", "⚖️", "Shoulders slightly uneven"))
    else:
        tips.append(("bad",  "⚖️", f"Shoulders very uneven ({sl:.2f}) — fix posture"))

    # Lean
    lean = features.get("lean_angle", 0)
    if lean > 12:
        tips.append(("warn", "📐", f"Body leaning {lean:.0f}° — stand/sit straight"))

    # Body framing
    sw = features.get("shoulder_width", 0)
    if 0 < sw < 0.15:
        tips.append(("warn", "📷", "Too far from camera — move closer"))
    elif sw > 0.65:
        tips.append(("warn", "📷", "Too close to camera — step back slightly"))

    # Speech
    spk  = speech.get("speaking", False)
    vol  = speech.get("volume", 0)
    pace = speech.get("speech_pace", "normal")

    if not spk:
        tips.append(("warn", "🔇", "No voice detected — speak up!"))
    else:
        if vol < 20:
            tips.append(("warn", "🔉", f"Voice too quiet (vol {vol:.0f}) — project more"))
        elif vol > 85:
            tips.append(("warn", "🔊", f"Voice very loud (vol {vol:.0f}) — moderate a little"))
        else:
            tips.append(("good", "🎙️", f"Good voice volume ({vol:.0f})"))

        if pace == "slow":
            tips.append(("warn", "🐢", "Speaking pace is slow — pick up the energy"))
        elif pace == "fast":
            tips.append(("warn", "🐇", "Speaking too fast — slow down for clarity"))
        else:
            tips.append(("good", "✅", "Speaking pace is good"))

    # Overall
    if score >= 85:
        tips.append(("info", "🏆", "Excellent! Keep this energy!"))
    elif score >= 65:
        tips.append(("info", "📈", "Good — refine the weak areas above"))
    elif score >= 45:
        tips.append(("info", "💡", "Fair — focus on eye contact & posture"))
    else:
        tips.append(("info", "🔧", "Keep practising — you'll improve quickly"))

    return tips


# ══════════════════════════════════════════════
#  CONFIDENCE TRACKER
# ══════════════════════════════════════════════

class ConfidenceTracker:
    """
    Maintains rolling history and computes:
      - trend      : "rising" | "stable" | "falling"
      - momentum   : float  -1..+1  (rate of change)
      - consistency: float  0-1     (low variance = consistent)
      - rolling_avg: float  current N-frame average
    """

    WINDOW = 30   # frames

    def __init__(self):
        self._hist = deque(maxlen=self.WINDOW)

    def update(self, score: int):
        self._hist.append(score)

    @property
    def rolling_avg(self):
        if not self._hist: return 0
        return round(np.mean(self._hist), 1)

    @property
    def trend(self):
        if len(self._hist) < 6: return "stable"
        half = len(self._hist) // 2
        first_half  = np.mean(list(self._hist)[:half])
        second_half = np.mean(list(self._hist)[half:])
        diff = second_half - first_half
        if diff >  4: return "rising"
        if diff < -4: return "falling"
        return "stable"

    @property
    def momentum(self):
        """Normalised slope of last WINDOW scores (-1 to +1)."""
        if len(self._hist) < 3: return 0.0
        scores = np.array(self._hist, dtype=float)
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        return float(np.clip(slope / 2, -1, 1))

    @property
    def consistency(self):
        if len(self._hist) < 3: return 1.0
        std = np.std(list(self._hist))
        return float(max(0.0, 1.0 - std / 30))   # std=0 → 1.0, std=30 → 0.0

    @property
    def summary(self):
        return {
            "rolling_avg":  self.rolling_avg,
            "trend":        self.trend,
            "momentum":     round(self.momentum, 3),
            "consistency":  round(self.consistency, 2),
        }


# ══════════════════════════════════════════════
#  CSV SESSION LOGGER
# ══════════════════════════════════════════════

class SessionLogger:
    """
    Appends one row per frame to a CSV file.
    Call .start() when session begins, .log() each frame, .finish() at end.
    """

    COLUMNS = [
        "timestamp", "elapsed_s", "score",
        # posture
        "face_detected", "face_center_x", "face_center_y", "face_size",
        "head_tilt", "head_nod", "head_distance",
        "eye_contact_score", "gaze_offset_x", "mouth_open",
        "body_visible", "shoulder_level", "shoulder_width",
        "body_center_x", "lean_angle",
        # speech
        "speaking", "volume", "silence_ratio", "speech_pace",
        # confidence
        "rolling_avg", "trend", "momentum", "consistency",
        # model
        "prediction", "confidence",
    ]

    def __init__(self, output_dir="sessions"):
        self.output_dir  = output_dir
        self._file       = None
        self._writer     = None
        self._start_time = None
        self.filepath    = None

    def start(self):
        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath    = os.path.join(self.output_dir, f"session_{ts}.csv")
        self._start_time = time.time()
        self._file       = open(self.filepath, "w", newline="")
        self._writer     = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()

    def log(self, score, posture, speech, confidence, prediction, pred_confidence):
        if self._writer is None:
            return
        row = {
            "timestamp":  datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "elapsed_s":  round(time.time() - self._start_time, 2),
            "score":      score,
            # posture fields
            **{k: round(posture.get(k, 0), 4) for k in [
                "face_detected", "face_center_x", "face_center_y", "face_size",
                "head_tilt", "head_nod", "head_distance",
                "eye_contact_score", "gaze_offset_x", "mouth_open",
                "body_visible", "shoulder_level", "shoulder_width",
                "body_center_x", "lean_angle",
            ]},
            # speech fields
            "speaking":      int(speech["speaking"]),
            "volume":        speech["volume"],
            "silence_ratio": speech["silence_ratio"],
            "speech_pace":   speech["speech_pace"],
            # confidence fields
            "rolling_avg":  confidence["rolling_avg"],
            "trend":        confidence["trend"],
            "momentum":     confidence["momentum"],
            "consistency":  confidence["consistency"],
            # model
            "prediction":   prediction,
            "confidence":   round(pred_confidence, 3),
        }
        self._writer.writerow(row)

    def finish(self):
        if self._file:
            self._file.flush()
            self._file.close()
            self._file   = None
            self._writer = None
        return self.filepath

    def export_summary_csv(self, score_history: list, output_dir="sessions"):
        """Export a lightweight per-session summary CSV (all sessions)."""
        path = os.path.join(output_dir, "session_summary.csv")
        exists = os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["datetime", "avg_score", "peak_score", "low_score", "frames"])
            if score_history:
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    int(np.mean(score_history)),
                    int(np.max(score_history)),
                    int(np.min(score_history)),
                    len(score_history),
                ])
        return path


# ══════════════════════════════════════════════
#  MAIN ENGINE  (single frame pass)
# ══════════════════════════════════════════════

def run_engine(frame, model, le, feature_names, face_mesh, pose, speech_analyser, confidence_tracker):
    """
    Process one frame and return a rich result dict.

    Parameters
    ----------
    frame               : BGR numpy array
    model               : sklearn model
    le                  : LabelEncoder
    feature_names       : list[str]
    face_mesh           : MediaPipe FaceMesh context
    pose                : MediaPipe Pose context
    speech_analyser     : SpeechAnalyser instance (already running)
    confidence_tracker  : ConfidenceTracker instance

    Returns
    -------
    dict with keys:
      prediction, pred_confidence,
      score, posture, speech, confidence_summary,
      feedback (list of tips)
    """
    small   = cv2.resize(frame, (480, 360))
    posture = extract_posture(small, face_mesh, pose)
    speech  = speech_analyser.summary

    # ML prediction
    try:
        X            = np.array([[posture.get(f, 0) for f in feature_names]])
        pred_enc     = model.predict(X)[0]
        proba        = model.predict_proba(X)[0]
        pred_conf    = float(max(proba))
        prediction   = le.inverse_transform([pred_enc])[0]
    except Exception:
        prediction   = "unknown"
        pred_conf    = 0.0

    score = calculate_score(posture, speech)
    confidence_tracker.update(score)

    feedback = generate_feedback(posture, speech, score)

    return {
        "prediction":         prediction,
        "pred_confidence":    pred_conf,
        "score":              score,
        "posture":            posture,
        "speech":             speech,
        "confidence_summary": confidence_tracker.summary,
        "feedback":           feedback,
    }
