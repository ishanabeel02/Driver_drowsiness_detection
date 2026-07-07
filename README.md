# Driver Drowsiness Detection System

Real-time driver drowsiness detection using MediaPipe Face Mesh, EAR/PERCLOS eye tracking, and MAR-based yawn detection, with a live WebSocket dashboard.

## Screenshots
<img width="1488" height="844" alt="image" src="https://github.com/user-attachments/assets/3e744146-b710-4e02-86a7-765ba19b0fe9" />
<img width="1500" height="844" alt="image" src="https://github.com/user-attachments/assets/d0b59669-99c6-488d-b21d-75703b7878e0" />
<img width="1498" height="844" alt="image" src="https://github.com/user-attachments/assets/be3b963a-2db4-4b35-a38e-dd1ee99debf5" />

<img width="953" height="365" alt="image" src="https://github.com/user-attachments/assets/e7fa08b8-3fce-4ec1-98f1-7a8bfde88b46" />

## How It Works

- Tracks facial landmarks with MediaPipe
- Computes EAR (eye closure) and PERCLOS (rolling drowsiness score)
- Computes MAR to detect yawns
- Fuses signals into a risk state: `ALERT → WARNING → DROWSY → CRITICAL`
- Triggers audio alerts and streams live metrics to a browser dashboard

## Project Files

- `drowsiness_final_system.py` — standalone webcam detector
- `websocket_server.py` — WebSocket backend for the dashboard
- `dashboard.html` — live browser dashboard
- `face_landmarker.task` — MediaPipe model file
- `CNN_baseline.ipynb` — CNN eye-state classifier baseline

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Webcam only:**
```bash
python drowsiness_final_system.py
```

**With dashboard:**
```bash
python websocket_server.py
```
Then open `dashboard.html` in your browser.

## Tech Stack

MediaPipe · OpenCV · WebSockets · TensorFlow (CNN baseline) · pygame

## Author

Built by Isha
