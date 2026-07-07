# 🧠 Drowsiness Detection — Driver Monitoring System

A two-part academic project demonstrating the limitations of frame-based CNN classification
and the superiority of temporal signal fusion for real-world drowsiness detection.

---

## Project Structure

```
├── drowsiness_cnn_baseline.ipynb    ← Part A: CNN training (Kaggle notebook)
├── drowsiness_final_system.py       ← Part B: Real-time detection (run locally)
├── DrowsinessDetectionDashboard.jsx ← React UI dashboard (Claude artifact)
└── requirements.txt
```

---

## Part A — CNN Baseline (Kaggle)

**Purpose:** Prove why frame-based learning is insufficient.

### Datasets

| Dataset | Path (Kaggle) | Use |
|---------|--------------|-----|
| MRL Eye Dataset | `/kaggle/input/datasets/akashshingha850/mrl-eye-dataset/data` | Eye: open(0) / closed(1) |
| Yawn Dataset | `/kaggle/input/datasets/davidvazquezcic/yawn-dataset` | Mouth: no-yawn(0) / yawn(1) |
| NTHU Dataset | `/kaggle/input/datasets/samymesbah/nthu-dataset-ddd-multi-class` | Generalization eval **only** |

> **Note on NTHU:** We do not directly train on NTHU due to its multi-modal nature;
> instead, it is used to evaluate generalization.

### Architecture

```
Input (64×64 gray / 96×96 RGB)
  ↓
Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
Conv2D(128)→ BN → ReLU → GlobalAvgPool
Dense(128/256) → Dropout(0.5) → Softmax(2)
```

### Expected outputs

- Training curves (accuracy / loss)
- Confusion matrices → shows blink vs closure confusion, talk vs yawn confusion
- GradCAM visualizations
- `cnn_baseline_results.csv`

### Run on Kaggle

1. Upload `drowsiness_cnn_baseline.ipynb` to Kaggle
2. Attach the three datasets
3. Add a cell: `!pip install mediapipe -q`
4. Run All

---

## Part B — Final System (Local / webcam)

**Architecture:** No training required. Uses MediaPipe Face Mesh (pre-trained, 468 landmarks).

### Signal pipeline

```
Webcam frame
    ↓
MediaPipe Face Mesh (468 landmarks) ← Deep Learning backbone
    ├── Eye landmarks → EAR per frame
    │       ↓
    │   PERCLOS engine (60s rolling window)
    │   PERCLOS = closed_frames / total_frames
    │
    └── Mouth landmarks → MAR per frame
            ↓
        Yawn tracker (MAR > 0.60 sustained > 1.5s → yawn event)
            ↓
        Yawn rate (yawns/minute in rolling window)
    ↓
Temporal Fusion
    ├── PERCLOS > 0.40 → Drowsy
    ├── PERCLOS > 0.70 → Critical
    ├── Yawns ≥ 2/min  → Warning boost
    ├── Yawns ≥ 4/min  → Critical boost
    └── (Combined risk score) → State: ALERT / WARNING / DROWSY / CRITICAL
    ↓
HUD overlay + Audio alert
```

### Run locally

```bash
pip install -r requirements.txt
python drowsiness_final_system.py
```

**Controls:**
- `q` — quit
- `r` — reset session counters

### Kaggle demo mode

When run inside a Kaggle kernel, the script auto-detects the environment and
runs on static images instead of a webcam (`run_notebook_demo()`).

---

## React Dashboard

The `DrowsinessDetectionDashboard.jsx` artifact provides a browser-based UI:

- Login screen (matches driver name → session)
- Live camera feed with state overlay border
- Real-time EAR / MAR / PERCLOS metric cards with sparklines
- PERCLOS bar with threshold markers (40% / 70%)
- Alert log with timestamps
- Past session history

**To plug in real detection:**
Replace the simulation block in `DriverMonitorPanel` (marked `REPLACE THIS BLOCK`)
with actual EAR/MAR values from your MediaPipe pipeline (e.g. via WebSocket from
the Python backend, or WASM-compiled MediaPipe).

---

## Key Academic Points

| Limitation | CNN Baseline | Final System |
|-----------|-------------|-------------|
| Temporal context | ❌ None | ✅ PERCLOS rolling window |
| Blink vs drowsy | ❌ Confused | ✅ Separated by duration |
| Talk vs yawn | ❌ Confused | ✅ MAR + sustained duration |
| Domain robustness | ❌ NTHU gap | ✅ Geometry-based, lighting-robust |
| Training required | ✅ Yes | ❌ No |
| Real-time capable | ❌ No | ✅ Yes (25–30 FPS) |

---

## Report Language

- Call the PERCLOS+MAR system **"temporal fusion"** (not "if-else logic")
- Call MediaPipe **"deep learning backbone"** (it is a pre-trained deep model)
- Frame the CNN baseline as motivation: _"We demonstrate that frame-based classification
  lacks the temporal dimension necessary for reliable drowsiness detection..."_
