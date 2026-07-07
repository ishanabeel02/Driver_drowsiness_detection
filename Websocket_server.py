"""
websocket_server.py — plain websockets, no FastAPI
Run: py -3.12 websocket_server.py

Changes in this version:
  - Streams the processed camera frame (with face-mesh landmarks drawn on it)
    to the browser as base64 JPEG inside the same JSON payload, so the
    dashboard can render it on a <canvas> instead of using getUserMedia().
    This avoids two processes fighting over the same physical webcam.
  - Re-integrates the AlertSystem (pygame tones + optional looping music)
    that plays locally on the machine running this server.
  - MAR_THRESHOLD raised so only a wide-open yawn triggers a yawn event.
"""
import asyncio, json, time, threading, collections, base64, os
import numpy as np
import cv2
import mediapipe as mp
import websockets

try:
    import pygame
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Audio unavailable ({e}); running without sound.")
    AUDIO_AVAILABLE = False

# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
# NOTE: 13/14 are INNER lip points — they sit right where the lips meet and
# barely separate even on a wide yawn, which made MAR look "stuck on one lip".
# Use OUTER lip landmarks instead, which actually track jaw drop, plus the
# mouth corners 61/291 (more stable across face angles than 78/308).
MOUTH_IDX = {
    "top":    13,    # kept only for the overlay text marker, not used in MAR math
    "bottom": 14,
    "left":   61,
    "right":  291,
}
# Outer-lip vertical pairs used for MAR — center pair + two off-center pairs,
# averaged for a more stable, less noisy reading than a single point pair.
MOUTH_VERTICAL_PAIRS = [(0, 17), (39, 181), (269, 405)]
MOUTH_H_LEFT, MOUTH_H_RIGHT = 61, 291

# Fuller landmark sets, used only for drawing the mesh overlay on the frame.
LEFT_EYE_OUTLINE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH_OUTLINE     = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
FACE_OVAL         = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                      152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

CFG = {
    "EAR_THRESHOLD"      : 0.20,
    "MAR_THRESHOLD"      : 0.55,   # outer-lip MAR scale: closed~0.05-0.1, talking~0.15-0.35, wide yawn~0.5+
    "PERCLOS_WINDOW_SEC" : 60,
    "PERCLOS_DROWSY"     : 0.40,
    "PERCLOS_CRITICAL"   : 0.55,   # was 0.70 — too high to hit in a live demo (needed 42s/60s eyes-closed)
    "YAWN_DURATION_SEC"  : 1.5,
    "YAWN_RATE_WINDOW"   : 60,
    "YAWN_RATE_WARNING"  : 2,
    "YAWN_RATE_CRITICAL" : 4,
    "ALERT_COOLDOWN_SEC" : 5.0,
    "MUSIC_PATH"         : "idol.mp3",   # set to a .mp3/.wav path to loop background music
    "JPEG_QUALITY"       : 70,     # 1-100, lower = smaller/faster over the socket
    "STREAM_FRAME"       : True,   # set False to fall back to metrics-only (no image)
}

class DrowsinessState:
    ALERT    = "ALERT"
    WARNING  = "WARNING"
    DROWSY   = "DROWSY"
    CRITICAL = "CRITICAL"

def euclidean(p1, p2): return np.linalg.norm(p1 - p2)

def compute_ear(lm, idx):
    p = lm[idx]
    return (euclidean(p[1],p[5]) + euclidean(p[2],p[4])) / (2.0*euclidean(p[0],p[3]) + 1e-6)

def compute_mar(lm):
    """
    MAR using OUTER lip landmarks (not inner 13/14, which barely separate).
    Averages three vertical pairs across the mouth width for a more stable
    reading, divided by mouth corner-to-corner width.
    """
    total_v = 0.0
    for top_idx, bot_idx in MOUTH_VERTICAL_PAIRS:
        total_v += euclidean(lm[top_idx], lm[bot_idx])
    v = total_v / len(MOUTH_VERTICAL_PAIRS)
    h = euclidean(lm[MOUTH_H_LEFT], lm[MOUTH_H_RIGHT])
    return v / (h + 1e-6)

def temporal_fusion(perclos, yawn_rate, is_yawning, ear):
    risk = 0.0
    if perclos >= CFG["PERCLOS_CRITICAL"]:       risk += 4.5   # was 3.0 — alone now reaches CRITICAL
    elif perclos >= CFG["PERCLOS_DROWSY"]:        risk += 2.0
    elif perclos >= 0.20:                         risk += 0.5
    if yawn_rate >= CFG["YAWN_RATE_CRITICAL"]:   risk += 2.0
    elif yawn_rate >= CFG["YAWN_RATE_WARNING"]:  risk += 1.0
    if is_yawning and perclos >= CFG["PERCLOS_DROWSY"]: risk += 1.5
    if ear < CFG["EAR_THRESHOLD"] * 0.7:         risk += 0.5   # was 0.3 — sustained near-shut eyes nudge harder
    if risk >= 4.5: return DrowsinessState.CRITICAL
    if risk >= 2.0: return DrowsinessState.DROWSY
    if risk >= 0.8: return DrowsinessState.WARNING
    return DrowsinessState.ALERT

# ── Alert system (sound) ───────────────────────────────────────────────────────
class AlertSystem:
    def __init__(self, cooldown_sec: float = 5.0, music_path: str = None):
        self.cooldown_sec   = cooldown_sec
        self._last_alert    = 0.0
        self._alert_thread  = None
        self._current_state = DrowsinessState.ALERT

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
        try:
            pygame.mixer.music.set_volume(volumes.get(state, 0.4))
        except Exception:
            pass  # no music loaded

    def trigger(self, state: str, now: float, force_silence: bool = False):
        # Fast-path: eyes have been visibly open for a bit — kill any sound
        # immediately rather than waiting for PERCLOS (60s rolling average)
        # to decay back down on its own.
        if force_silence:
            if AUDIO_AVAILABLE:
                pygame.mixer.stop()
                self._set_music_volume(DrowsinessState.ALERT)
            self._current_state = DrowsinessState.ALERT
            return

        if state == DrowsinessState.ALERT:
            if AUDIO_AVAILABLE:
                pygame.mixer.stop()
                self._set_music_volume(DrowsinessState.ALERT)
            self._current_state = state
            return

        if state == self._current_state:
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
                self._set_music_volume(DrowsinessState.CRITICAL)
                for _ in range(3):
                    self._generate_alarm(1000)
                    time.sleep(0.05)
                self._set_music_volume(DrowsinessState.DROWSY)

            elif state == DrowsinessState.DROWSY:
                for freq in [700, 900, 1100]:
                    self._generate_beep(freq, 400, volume=0.85)
                    time.sleep(0.05)

            elif state == DrowsinessState.WARNING:
                self._generate_beep(660, 350, volume=0.4)

        self._alert_thread = threading.Thread(target=_play, daemon=True)
        self._alert_thread.start()

alert_system = AlertSystem(cooldown_sec=CFG["ALERT_COOLDOWN_SEC"], music_path=CFG["MUSIC_PATH"])

# ── Mesh drawing ────────────────────────────────────────────────────────────────
def draw_mesh_overlay(frame, lm, ear, mar, eye_closed, mouth_open):
    """Draws eye/mouth/face-oval landmarks onto the BGR frame in-place."""
    eye_color   = (0, 0, 255) if eye_closed else (0, 220, 0)
    mouth_color = (0, 140, 255) if mouth_open else (0, 220, 0)
    face_color  = (180, 120, 0)

    for idx in FACE_OVAL:
        cv2.circle(frame, tuple(lm[idx].astype(int)), 1, face_color, -1)
    for idx in LEFT_EYE_OUTLINE + RIGHT_EYE_OUTLINE:
        cv2.circle(frame, tuple(lm[idx].astype(int)), 2, eye_color, -1)
    for idx in MOUTH_OUTLINE:
        cv2.circle(frame, tuple(lm[idx].astype(int)), 2, mouth_color, -1)

    cv2.putText(frame, f"EAR {ear:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
    cv2.putText(frame, f"MAR {mar:.2f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2)

def encode_frame_b64(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, CFG["JPEG_QUALITY"]])
    if not ok:
        return None
    return base64.b64encode(buf).decode("ascii")

# ── Shared state ──────────────────────────────────────────────────────────────
_state = {
    "ear":0.0,"mar":0.0,"perclos":0.0,
    "blinks":0,"yawns":0,"yawn_rate":0.0,
    "drowsy_state":"ALERT","face_detected":False,"fps":0.0,
    "frame":None,
}
_state_lock = threading.Lock()
_camera_running = False

def camera_thread():
    global _camera_running
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    closed_frames    = collections.deque()
    mouth_open_since = None
    current_yawning  = False
    yawn_timestamps  = collections.deque()
    blink_count = 0; blink_consec = 0; yawn_count = 0
    fps_deque   = collections.deque(maxlen=30)
    prev_time   = time.time()
    eyes_open_since = None   # tracks how long eyes have been continuously open, for fast alert cutoff
    _camera_running = True

    while _camera_running:
        ret, frame = cap.read()
        if not ret: continue
        now = time.time()
        fps_deque.append(1.0 / (now - prev_time + 1e-9))
        prev_time = now
        fps = float(np.mean(fps_deque))
        H, W = frame.shape[:2]
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ear = 0.0; mar = 0.0; face_detected = False
        eye_closed = False; mouth_open = False

        if result.multi_face_landmarks:
            face_detected = True
            lm = np.array([(l.x*W, l.y*H) for l in result.multi_face_landmarks[0].landmark])
            ear = (compute_ear(lm,LEFT_EYE_IDX) + compute_ear(lm,RIGHT_EYE_IDX)) / 2.0
            mar = compute_mar(lm)
            eye_closed = ear < CFG["EAR_THRESHOLD"]
            if eye_closed:
                blink_consec += 1
                eyes_open_since = None
            else:
                if blink_consec >= 3: blink_count += 1
                blink_consec = 0
                if eyes_open_since is None: eyes_open_since = now
            closed_frames.append({"closed":int(eye_closed),"ts":now})
            while closed_frames and closed_frames[0]["ts"] < now - CFG["PERCLOS_WINDOW_SEC"]:
                closed_frames.popleft()
            perclos = sum(f["closed"] for f in closed_frames) / max(len(closed_frames),1)
            mouth_open = mar > CFG["MAR_THRESHOLD"]
            if mouth_open:
                if mouth_open_since is None: mouth_open_since = now
                if now - mouth_open_since >= CFG["YAWN_DURATION_SEC"] and not current_yawning:
                    current_yawning = True; yawn_count += 1; yawn_timestamps.append(now)
            else:
                mouth_open_since = None; current_yawning = False
            while yawn_timestamps and yawn_timestamps[0] < now - CFG["YAWN_RATE_WINDOW"]:
                yawn_timestamps.popleft()
            yawn_rate = len(yawn_timestamps) / (CFG["YAWN_RATE_WINDOW"]/60.0)
            state = temporal_fusion(perclos, yawn_rate, current_yawning, ear)

            if CFG["STREAM_FRAME"]:
                draw_mesh_overlay(frame, lm, ear, mar, eye_closed, mouth_open)
        else:
            perclos=0.0; yawn_rate=0.0; state=DrowsinessState.ALERT
            eyes_open_since = None

        # Fast cutoff: if eyes have been continuously open for >1.5s, silence
        # alert sound immediately rather than waiting for PERCLOS (a 60s
        # rolling average) to decay back down on its own.
        eyes_open_now = eyes_open_since is not None and (now - eyes_open_since) >= 1.5
        alert_system.trigger(state, now, force_silence=eyes_open_now)

        frame_b64 = encode_frame_b64(frame) if CFG["STREAM_FRAME"] else None

        with _state_lock:
            _state.update({
                "ear":round(float(ear),4),"mar":round(float(mar),4),
                "perclos":round(float(perclos),4),"blinks":blink_count,
                "yawns":yawn_count,"yawn_rate":round(float(yawn_rate),2),
                "drowsy_state":state,"face_detected":face_detected,"fps":round(fps,1),
                "frame":frame_b64,
            })
    cap.release(); face_mesh.close()

# ── WebSocket handler ─────────────────────────────────────────────────────────
async def handler(websocket):
    print(f"[WS] Client connected: {websocket.remote_address}")
    try:
        while True:
            with _state_lock:
                payload = dict(_state)
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(0.1)
    except websockets.exceptions.ConnectionClosed:
        print("[WS] Client disconnected.")

async def main():
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()
    print("[INFO] Camera thread started.")

    print("=" * 50)
    print("  Drowsiness Detection WebSocket Server")
    print("  ws://localhost:8765")
    print("=" * 50)

    async with websockets.serve(handler, "0.0.0.0", 8765, max_size=2**22):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())