#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PART B — FINAL DROWSINESS DETECTION SYSTEM                               ║
║   Real-time | No training | MediaPipe Face Mesh | PERCLOS | MAR            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Deep Learning backbone: MediaPipe Face Mesh (468 landmarks, pre-trained)  ║
║   Eye state  : Eye Aspect Ratio (EAR) + PERCLOS rolling window              ║
║   Mouth state: Mouth Aspect Ratio (MAR) + sustained duration                ║
║   Fusion     : Temporal decision fusion (not simple if-else)                 ║
║                                                                              ║
║   Run: python drowsiness_final_system.py                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd
import collections
import threading
import os
import sys

# Optional audio alert (pygame) — graceful fallback if not installed
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[INFO] pygame not found — audio alerts disabled. Install with: pip install pygame")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    # ── EAR / Eye ─────────────────────────────────────────────────────────────
    "EAR_THRESHOLD"       : 0.20,   # below this → eye is "closed"
    "BLINK_CONSEC_FRAMES" : 3,      # frames of closure to count as a blink

    # ── PERCLOS ───────────────────────────────────────────────────────────────
    "PERCLOS_WINDOW_SEC"  : 30,    # 30s is a reasonable demo compromise
    "PERCLOS_DROWSY"      : 0.35,
    "PERCLOS_CRITICAL"    : 0.60,

    # ── MAR / Mouth ───────────────────────────────────────────────────────────
    "MAR_THRESHOLD"       : 0.60,   # above this → mouth "open" (possible yawn)
    "YAWN_DURATION_SEC"   : 1.5,    # sustained open > 1.5s → yawn event
    "YAWN_RATE_WINDOW"    : 60,     # window (sec) for yawns-per-minute

    # ── Fusion thresholds ─────────────────────────────────────────────────────
    "YAWN_RATE_WARNING"   : 2,      # ≥ 2 yawns/min → increase risk
    "YAWN_RATE_CRITICAL"  : 4,      # ≥ 4 yawns/min → critical boost

    # ── Display ───────────────────────────────────────────────────────────────
    "FRAME_WIDTH"         : 1280,
    "FRAME_HEIGHT"        : 720,
    "CAMERA_INDEX"        : 0,

    # ── Alert cooldown ────────────────────────────────────────────────────────
    "ALERT_COOLDOWN_SEC"  : 3,      # min seconds between audio alerts
}


# ══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE LANDMARK INDICES
# ══════════════════════════════════════════════════════════════════════════════

# Left eye (outer to inner, upper to lower) — MediaPipe Face Mesh indices
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# Mouth landmarks
MOUTH_IDX = {
    "top"    : 13,
    "bottom" : 14,
    "left"   : 78,
    "right"  : 308,
    "top_l"  : 82,
    "top_r"  : 312,
    "bot_l"  : 87,
    "bot_r"  : 317,
}

# Iris refinement indices (for gaze, optional)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)


def compute_ear(landmarks: np.ndarray, eye_idx: list) -> float:
    """
    Eye Aspect Ratio (EAR) — Soukupova & Čech (2016).

    EAR = (‖P2−P6‖ + ‖P3−P5‖) / (2 · ‖P1−P4‖)

    Returns:
        float: EAR value (low → closed, high → open)
    """
    p = landmarks[eye_idx]
    # Vertical distances
    v1 = euclidean(p[1], p[5])
    v2 = euclidean(p[2], p[4])
    # Horizontal distance
    h  = euclidean(p[0], p[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def compute_mar(landmarks: np.ndarray) -> float:
    """
    Mouth Aspect Ratio (MAR) — analogous to EAR.

    MAR = (‖top−bottom‖) / (‖left−right‖)
    """
    top    = landmarks[MOUTH_IDX["top"]]
    bottom = landmarks[MOUTH_IDX["bottom"]]
    left   = landmarks[MOUTH_IDX["left"]]
    right  = landmarks[MOUTH_IDX["right"]]

    vertical   = euclidean(top, bottom)
    horizontal = euclidean(left, right)
    return vertical / (horizontal + 1e-6)


def get_landmarks_array(face_landmarks, w: int, h: int) -> np.ndarray:
    return np.array([
        (lm.x * w, lm.y * h)
        for lm in face_landmarks
    ])


# ══════════════════════════════════════════════════════════════════════════════
# PERCLOS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class PerclosEngine:
    """
    Maintains a rolling time-window of per-frame eye states.
    PERCLOS = fraction of frames where EAR < threshold (eye closed).
    """

    def __init__(self, window_sec: float, fps_estimate: float = 25.0):
        self.window_sec  = window_sec
        self.fps_estimate = fps_estimate
        # maxlen auto-prunes oldest frames
        maxlen = int(window_sec * fps_estimate)
        self._closed_flags: collections.deque = collections.deque(maxlen=maxlen)
        self._timestamps:   collections.deque = collections.deque(maxlen=maxlen)

    def update(self, eye_closed: bool, timestamp: float):
        """Record whether eye was closed at this timestamp."""
        self._closed_flags.append(int(eye_closed))
        self._timestamps.append(timestamp)
        # Prune entries older than window_sec
        cutoff = timestamp - self.window_sec
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
            self._closed_flags.popleft()

    def perclos(self) -> float:
        """Return current PERCLOS value in [0, 1]."""
        if not self._closed_flags:
            return 0.0
        return sum(self._closed_flags) / len(self._closed_flags)

    def frame_count(self) -> int:
        return len(self._closed_flags)


# ══════════════════════════════════════════════════════════════════════════════
# YAWN TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class YawnTracker:
    """
    Detects yawn events from MAR using duration gating.
    Counts yawns-per-minute in a rolling window.
    """

    def __init__(self, mar_threshold: float, duration_sec: float, rate_window_sec: float):
        self.mar_threshold   = mar_threshold
        self.duration_sec    = duration_sec
        self.rate_window_sec = rate_window_sec

        self._mouth_open_since: float | None = None
        self._yawn_timestamps: collections.deque = collections.deque()
        self.current_yawning = False

    def update(self, mar: float, timestamp: float):
        """Update yawn state given current MAR value."""
        mouth_open = mar > self.mar_threshold

        if mouth_open:
            if self._mouth_open_since is None:
                self._mouth_open_since = timestamp
            duration = timestamp - self._mouth_open_since
            if duration >= self.duration_sec and not self.current_yawning:
                # Yawn event confirmed
                self.current_yawning = True
                self._yawn_timestamps.append(timestamp)
        else:
            self._mouth_open_since = None
            self.current_yawning   = False

        # Prune old yawn events
        cutoff = timestamp - self.rate_window_sec
        while self._yawn_timestamps and self._yawn_timestamps[0] < cutoff:
            self._yawn_timestamps.popleft()

    def yawn_rate(self) -> float:
        """Yawns per minute in the rolling window."""
        if not self._yawn_timestamps:
            return 0.0
        return len(self._yawn_timestamps) / (self.rate_window_sec / 60.0)

    def total_yawns(self) -> int:
        return len(self._yawn_timestamps)


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL FUSION — DROWSINESS STATE MACHINE
# ══════════════════════════════════════════════════════════════════════════════

class DrowsinessState:
    ALERT    = "ALERT"
    WARNING  = "WARNING"
    DROWSY   = "DROWSY"
    CRITICAL = "CRITICAL"


def temporal_fusion(
    perclos: float,
    yawn_rate: float,
    is_yawning: bool,
    ear: float,
    cfg: dict
) -> str:
    """
    Multi-signal temporal fusion.
    Combines PERCLOS (primary), yawn rate (secondary), and instantaneous
    yawning flag to determine drowsiness state.

    This is temporal fusion, not simple if-else: each signal contributes
    to a risk score that gates state transitions.
    """
    risk = 0.0

    # ── Primary signal: PERCLOS ───────────────────────────────────────────────
    if perclos >= cfg["PERCLOS_CRITICAL"]:
        risk += 3.0
    elif perclos >= cfg["PERCLOS_DROWSY"]:
        risk += 2.0
    elif perclos >= 0.20:
        risk += 0.5

    # ── Secondary signal: yawn rate ───────────────────────────────────────────
    if yawn_rate >= cfg["YAWN_RATE_CRITICAL"]:
        risk += 2.0
    elif yawn_rate >= cfg["YAWN_RATE_WARNING"]:
        risk += 1.0

    # ── Instantaneous yawn during high PERCLOS → amplify risk ────────────────
    if is_yawning and perclos >= cfg["PERCLOS_DROWSY"]:
        risk += 1.5

    # ── Very low EAR right now (acute closure) ────────────────────────────────
    if ear < cfg["EAR_THRESHOLD"] * 0.7:
        risk += 0.3

    # ── State mapping from risk score ────────────────────────────────────────
    if risk >= 4.5:
        return DrowsinessState.CRITICAL
    elif risk >= 2.0:
        return DrowsinessState.DROWSY
    elif risk >= 0.8:
        return DrowsinessState.WARNING
    else:
        return DrowsinessState.ALERT


# ══════════════════════════════════════════════════════════════════════════════
# ALERT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class AlertSystem:
    def __init__(self, cooldown_sec: float = 5.0, music_path: str = None):
        self.cooldown_sec  = cooldown_sec
        self._last_alert   = 0.0
        self._alert_thread = None
        self._current_state = DrowsinessState.ALERT

        # Start background music
        if AUDIO_AVAILABLE and music_path and os.path.exists(music_path):
            pygame.mixer.music.load(music_path)
            pygame.mixer.music.set_volume(0.4)
            pygame.mixer.music.play(-1)  # loop forever

    def _generate_beep(self, frequency=880, duration_ms=500, volume=0.8):
        if not AUDIO_AVAILABLE:
            return
        sample_rate = 44100
        n_samples   = int(sample_rate * duration_ms / 1000)
        t           = np.linspace(0, duration_ms / 1000, n_samples)
        wave        = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)
        wave        = np.column_stack([wave, wave])
        sound       = pygame.sndarray.make_sound(wave)
        sound.play()
        pygame.time.wait(duration_ms)

    def _generate_alarm(self, duration_ms=1000):
        if not AUDIO_AVAILABLE:
            return
        sample_rate = 44100
        n_samples   = int(sample_rate * duration_ms / 1000)
        t           = np.linspace(0, duration_ms / 1000, n_samples)
        freq_sweep  = 600 + 800 * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        wave        = (np.sin(2 * np.pi * freq_sweep * t) * 0.95 * 32767).astype(np.int16)
        wave        = np.column_stack([wave, wave])
        sound       = pygame.sndarray.make_sound(wave)
        sound.play()
        pygame.time.wait(duration_ms)

    def _set_music_volume(self, state: str):
        if not AUDIO_AVAILABLE:
            return
        volumes = {
            DrowsinessState.ALERT    : 0.4,
            DrowsinessState.WARNING  : 0.4,
            DrowsinessState.DROWSY   : 0.85,   # music gets louder
            DrowsinessState.CRITICAL : 0.0,    # mute during siren
        }
        pygame.mixer.music.set_volume(volumes.get(state, 0.4))

    def trigger(self, state: str, now: float):
        # Stop alerts immediately when back to ALERT
        if state == DrowsinessState.ALERT:
            if AUDIO_AVAILABLE:
                pygame.mixer.stop()
                pygame.mixer.music.set_volume(0.4)
            self._current_state = state
            return
        if state == self._current_state:
            # Still update volume even if no new alert
            self._set_music_volume(state)

        if state not in (DrowsinessState.WARNING, DrowsinessState.DROWSY, DrowsinessState.CRITICAL):
            self._set_music_volume(state)
            self._current_state = state
            return

        self._set_music_volume(state)
        self._current_state = state

        if now - self._last_alert < self.cooldown_sec:
            return
        if self._alert_thread and self._alert_thread.is_alive():
            return

        self._last_alert = now

        def _play():
            if state == DrowsinessState.CRITICAL:
                pygame.mixer.music.set_volume(0.0)
                for _ in range(3):
                    self._generate_alarm(1000)
                    time.sleep(0.05)
                pygame.mixer.music.set_volume(0.85)  # resume loud after siren

            elif state == DrowsinessState.DROWSY:
                # Three beeps over the loud music
                for freq in [700, 900, 1100]:
                    self._generate_beep(freq, 400, volume=0.85)
                    time.sleep(0.05)

            elif state == DrowsinessState.WARNING:
                self._generate_beep(660, 350, volume=0.4)

        self._alert_thread = threading.Thread(target=_play, daemon=True)
        self._alert_thread.start()


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY / HUD
# ══════════════════════════════════════════════════════════════════════════════

# Color palette (BGR for OpenCV)
COLORS = {
    "alert"    : (80,  200,  80),    # green
    "warning"  : (0,   200, 255),    # amber/yellow
    "drowsy"   : (0,   120, 255),    # orange
    "critical" : (0,    40, 220),    # red
    "white"    : (255, 255, 255),
    "black"    : (0,     0,   0),
    "panel_bg" : (20,   20,  30),
    "grid"     : (50,   50,  60),
}

STATE_COLORS = {
    DrowsinessState.ALERT    : COLORS["alert"],
    DrowsinessState.WARNING  : COLORS["warning"],
    DrowsinessState.DROWSY   : COLORS["drowsy"],
    DrowsinessState.CRITICAL : COLORS["critical"],
}


def draw_metric_panel(
    frame: np.ndarray,
    ear: float, mar: float, perclos: float,
    blink_count: int, yawn_count: int, yawn_rate: float,
    state: str, fps: float,
    x: int = 10, y: int = 10
):
    """
    Draw the HUD panel on the left side of the frame.
    """
    W = 280
    H = 380
    overlay = frame.copy()

    # Panel background
    cv2.rectangle(overlay, (x, y), (x + W, y + H), COLORS["panel_bg"], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title
    cv2.putText(frame, "DRIVER MONITOR", (x + 12, y + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, COLORS["white"], 1, cv2.LINE_AA)
    cv2.line(frame, (x + 10, y + 35), (x + W - 10, y + 35), COLORS["grid"], 1)

    def metric_row(label, value_str, row, color=COLORS["white"]):
        cy = y + 60 + row * 40
        cv2.putText(frame, label, (x + 14, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["grid"], 1, cv2.LINE_AA)
        cv2.putText(frame, value_str, (x + 14, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    metric_row("EAR",          f"{ear:.3f}",      0)
    metric_row("MAR",          f"{mar:.3f}",      1)
    metric_row("PERCLOS",      f"{perclos:.1%}",  2,
               COLORS["critical"] if perclos > CFG["PERCLOS_CRITICAL"]
               else COLORS["drowsy"] if perclos > CFG["PERCLOS_DROWSY"]
               else COLORS["white"])
    metric_row("BLINKS",       str(blink_count),  3)
    metric_row("YAWNS",        str(yawn_count),   4)
    metric_row("YAWN RATE/min",f"{yawn_rate:.1f}", 5)
    metric_row("FPS",          f"{fps:.1f}",       6)

    # State badge
    state_color = STATE_COLORS[state]
    by = y + H - 45
    cv2.rectangle(frame, (x + 10, by), (x + W - 10, by + 35), state_color, -1)
    cv2.putText(frame, state, (x + 28, by + 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, COLORS["black"], 2, cv2.LINE_AA)


def draw_eye_landmarks(frame, landmarks, eye_idx, color):
    pts = landmarks[eye_idx].astype(int)
    for i, pt in enumerate(pts):
        cv2.circle(frame, tuple(pt), 2, color, -1)
    # Draw polygon
    cv2.polylines(frame, [pts.reshape(-1, 1, 2)], True, color, 1, cv2.LINE_AA)


def draw_mouth_landmarks(frame, landmarks):
    for key, idx in MOUTH_IDX.items():
        pt = landmarks[idx].astype(int)
        cv2.circle(frame, tuple(pt), 2, COLORS["warning"], -1)


def draw_perclos_bar(frame, perclos: float, x: int, y: int, w: int = 200, h: int = 16):
    """Horizontal PERCLOS progress bar."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS["grid"], -1)
    fill_w = int(perclos * w)
    color  = (COLORS["critical"] if perclos > CFG["PERCLOS_CRITICAL"]
              else COLORS["drowsy"] if perclos > CFG["PERCLOS_DROWSY"]
              else COLORS["alert"])
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS["white"], 1)
    # Threshold markers
    cv2.line(frame,
             (x + int(CFG["PERCLOS_DROWSY"] * w), y),
             (x + int(CFG["PERCLOS_DROWSY"] * w), y + h),
             COLORS["white"], 1)
    cv2.line(frame,
             (x + int(CFG["PERCLOS_CRITICAL"] * w), y),
             (x + int(CFG["PERCLOS_CRITICAL"] * w), y + h),
             COLORS["white"], 1)


def draw_alert_banner(frame: np.ndarray, state: str):
    """Full-width alert banner at top for DROWSY/CRITICAL states."""
    if state == DrowsinessState.ALERT:
        return
    H, W = frame.shape[:2]
    color = STATE_COLORS[state]
    alpha = 0.6 if state == DrowsinessState.CRITICAL else 0.4
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 60), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    msgs = {
        DrowsinessState.WARNING  : "⚠  DROWSINESS WARNING — Stay alert!",
        DrowsinessState.DROWSY   : "😴  DROWSY DETECTED — Take a break!",
        DrowsinessState.CRITICAL : "🚨  CRITICAL — STOP DRIVING NOW!",
    }
    msg = msgs.get(state, "")
    cv2.putText(frame, msg, (20, 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, COLORS["white"], 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DETECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_detection():
    # ── MediaPipe setup ───────────────────────────────────────────────────────
    face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path='face_landmarker.task'
            ),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
    )

    # ── Camera setup ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CFG["CAMERA_INDEX"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG["FRAME_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG["FRAME_HEIGHT"])
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in CFG.")
        sys.exit(1)

    # ── State tracking ────────────────────────────────────────────────────────
    perclos_engine = PerclosEngine(CFG["PERCLOS_WINDOW_SEC"])
    yawn_tracker   = YawnTracker(
        CFG["MAR_THRESHOLD"],
        CFG["YAWN_DURATION_SEC"],
        CFG["YAWN_RATE_WINDOW"]
    )
    alert_system = AlertSystem(
        cooldown_sec=3.0,
        music_path='idol.mp3'  # ← put your mp3 filename here
    )

    blink_count      = 0
    blink_consec     = 0
    prev_eye_closed  = False

    # FPS tracking
    fps_deque  = collections.deque(maxlen=30)
    prev_time  = time.time()

    print("\n" + "═" * 60)
    print("  DROWSINESS DETECTION SYSTEM — Active")
    print("  Press 'q' to quit | 'r' to reset counters")
    print("═" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed — retrying...")
            continue

        now   = time.time()
        delta = now - prev_time
        prev_time = now
        fps_deque.append(1.0 / (delta + 1e-9))
        fps = np.mean(fps_deque)

        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = face_mesh.detect(mp_image)

        # ── Default values when no face detected ──────────────────────────────
        ear = 0.0
        mar = 0.0
        state = DrowsinessState.ALERT
        perclos_val = perclos_engine.perclos()

        if results.face_landmarks:
            landmarks = get_landmarks_array(results.face_landmarks[0], W, H)

            # ── EAR ───────────────────────────────────────────────────────────
            left_ear  = compute_ear(landmarks, LEFT_EYE_IDX)
            right_ear = compute_ear(landmarks, RIGHT_EYE_IDX)
            ear       = (left_ear + right_ear) / 2.0

            # ── MAR ───────────────────────────────────────────────────────────
            mar = compute_mar(landmarks)

            # ── Blink counting ────────────────────────────────────────────────
            eye_closed = ear < CFG["EAR_THRESHOLD"]
            if eye_closed:
                blink_consec += 1
            else:
                if blink_consec >= CFG["BLINK_CONSEC_FRAMES"]:
                    blink_count += 1
                blink_consec = 0
            prev_eye_closed = eye_closed

            # ── PERCLOS update ────────────────────────────────────────────────
            perclos_engine.update(eye_closed, now)
            perclos_val = perclos_engine.perclos()

            # ── Yawn update ───────────────────────────────────────────────────
            yawn_tracker.update(mar, now)

            # ── Temporal fusion ───────────────────────────────────────────────
            state = temporal_fusion(
                perclos     = perclos_val,
                yawn_rate   = yawn_tracker.yawn_rate(),
                is_yawning  = yawn_tracker.current_yawning,
                ear         = ear,
                cfg         = CFG
            )

            # ── Draw landmarks ────────────────────────────────────────────────
            eye_color = COLORS["critical"] if eye_closed else COLORS["alert"]
            draw_eye_landmarks(frame, landmarks, LEFT_EYE_IDX,  eye_color)
            draw_eye_landmarks(frame, landmarks, RIGHT_EYE_IDX, eye_color)
            draw_mouth_landmarks(frame, landmarks)

            # ── Yawning indicator ─────────────────────────────────────────────
            if yawn_tracker.current_yawning:
                cv2.putText(frame, "YAWNING", (W // 2 - 60, H - 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9,
                            COLORS["warning"], 2, cv2.LINE_AA)

        else:
            # No face detected
            cv2.putText(frame, "NO FACE DETECTED", (W // 2 - 140, H // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0,
                        COLORS["critical"], 2, cv2.LINE_AA)

        # ── Audio alert ───────────────────────────────────────────────────────
        alert_system.trigger(state, now)

        # ── HUD overlay ───────────────────────────────────────────────────────
        draw_alert_banner(frame, state)

        draw_metric_panel(
            frame,
            ear         = ear,
            mar         = mar,
            perclos     = perclos_val,
            blink_count = blink_count,
            yawn_count  = yawn_tracker.total_yawns(),
            yawn_rate   = yawn_tracker.yawn_rate(),
            state       = state,
            fps         = fps
        )

        # PERCLOS bar (bottom-right)
        bar_x = W - 240
        bar_y = H - 40
        cv2.putText(frame, "PERCLOS", (bar_x, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["white"], 1)
        draw_perclos_bar(frame, perclos_val, bar_x, bar_y)

        # EAR live graph (top-right)
        # (simplified: just show value)
        cv2.putText(frame,
                    f"EAR: {ear:.3f}  MAR: {mar:.3f}",
                    (W - 250, H - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["white"], 1, cv2.LINE_AA)

        # ── Show frame ────────────────────────────────────────────────────────
        cv2.imshow("Drowsiness Detection — Driver Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            blink_count = 0
            blink_consec = 0
            perclos_engine = PerclosEngine(CFG["PERCLOS_WINDOW_SEC"])
            yawn_tracker   = YawnTracker(
                CFG["MAR_THRESHOLD"],
                CFG["YAWN_DURATION_SEC"],
                CFG["YAWN_RATE_WINDOW"]
            )
            print("[INFO] Counters reset.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("\n[INFO] Session ended.")


# ══════════════════════════════════════════════════════════════════════════════
# KAGGLE / NOTEBOOK MODE — Demo without webcam
# ══════════════════════════════════════════════════════════════════════════════

def run_notebook_demo(image_paths: list, show_results: bool = True):
    """
    Kaggle-compatible demo: run detection on a list of static images
    instead of a live webcam feed.

    Args:
        image_paths: list of paths to test images
        show_results: whether to display results with matplotlib
    """
    import matplotlib.pyplot as plt

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    results_log = []

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Could not read: {img_path}")
            continue

        H, W = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear, mar = 0.0, 0.0

        if results.multi_face_landmarks:
            landmarks = get_landmarks_array(results.multi_face_landmarks[0], W, H)
            left_ear  = compute_ear(landmarks, LEFT_EYE_IDX)
            right_ear = compute_ear(landmarks, RIGHT_EYE_IDX)
            ear       = (left_ear + right_ear) / 2.0
            mar       = compute_mar(landmarks)

        eye_state  = "Closed" if ear < CFG["EAR_THRESHOLD"] else "Open"
        yawn_state = "Yawning" if mar > CFG["MAR_THRESHOLD"] else "No Yawn"
        results_log.append({
            "image"    : os.path.basename(str(img_path)),
            "EAR"      : round(ear, 4),
            "MAR"      : round(mar, 4),
            "Eye"      : eye_state,
            "Yawn"     : yawn_state,
        })

    face_mesh.close()

 
    df = pd.DataFrame(results_log)
    print("\n=== Notebook Demo Results ===")
    print(df.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Check if running in Kaggle / headless environment ────────────────────
    IN_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

    if IN_KAGGLE:
        print("[INFO] Kaggle environment detected → running notebook demo mode.")
        print("[INFO] To run live webcam detection, execute this script locally.")
        print()
        # Example: run on any face images available in the Kaggle environment
        # Modify the path list to point to actual test images
        test_images = list(Path('/kaggle/input').rglob('*.jpg'))[:10]
        if test_images:
            run_notebook_demo(test_images)
        else:
            print("[INFO] No test images found. Place face images in /kaggle/input/.")
    else:
        run_detection()
