# Smart Glasses Stereo Obstacle Detection Module

## Overview

This repository contains a **real-time obstacle detection component** designed for **smart glasses used by visually impaired users**.

The system uses a **stereo (dual-camera) vision setup** to estimate depth and assess **collision risk**, providing **minimal audio feedback** only when necessary. The focus is on **safe navigation**, not object recognition.

---

## Goal

**Help visually impaired users walk safely by warning them about dangerous obstacles before a collision occurs.**

The system answers one simple question:

> *“Is there something in the user’s path that could cause harm soon?”*

---

## Key Design Principles

- Depth-first perception (distance matters more than object labels)
- Real-time performance on wearable or edge hardware
- Minimal cognitive load (no constant narration)
- Safety-first design (false positives are preferable to false negatives)
- Privacy-preserving (no cloud dependency)

---

## Inputs and Outputs

### Inputs
- Synchronized **left and right camera frames** (stereo vision)
- Precomputed **stereo calibration parameters**

### Outputs
- Risk level: `SAFE`, `CAUTION`, or `DANGER`
- Obstacle direction: `left`, `center`, or `right`
- Estimated distance to nearest obstacle (meters)
- Short audio warnings via text-to-speech (TTS)

---

## System Pipeline

```
Stereo Frames
   ↓
Rectification (Calibration)
   ↓
Disparity Estimation (SGBM)
   ↓
Depth Map Generation
   ↓
Walking Corridor Zoning
   ↓
Nearest Obstacle Detection
   ↓
Time-to-Collision (TTC)
   ↓
Risk Scoring
   ↓
Smart Audio Feedback
```

---

## Hardware Assumptions

- Stereo USB camera (dual lens, synchronized)
- Global shutter preferred (reduces motion artifacts)
- Wide field of view (~120° recommended)
- Compute device:
  - Windows laptop
  - Jetson Nano / Xavier
  - Embedded Linux system
  - Android phone (OTG, future support)

---

## Software Requirements

- Python 3.8 or newer
- OpenCV
- NumPy
- pyttsx3 (for audio feedback)

### Install Dependencies

```bash
pip install opencv-python numpy pyttsx3
```

---

## Repository Structure

```
.
├── stereo_obstacle_module.py   # Main real-time pipeline
├── stereo_calib.npz            # Saved stereo calibration file
└── README.md                   # Documentation
```

---

## Stereo Calibration (Required Once)

Stereo calibration must be completed **before running the system**.

The calibration file (`stereo_calib.npz`) must contain:

- Rectification maps:
  - `mapLx`, `mapLy`
  - `mapRx`, `mapRy`
- Either:
  - `Q` reprojection matrix  
  **OR**
  - focal length `f` (pixels) and baseline `B` (meters)

⚠️ **Calibration quality directly affects depth accuracy.**  
Poor calibration will cause noisy depth maps and false warnings.

---

## Running the System

### Case 1: Side-by-Side Stereo Stream (Single Camera Device)

Most stereo USB cameras output a single frame with left and right images concatenated horizontally.

```bash
python stereo_obstacle_module.py --calib stereo_calib.npz --src 0
```

---

### Case 2: Two Separate Camera Devices

If the left and right cameras appear as separate devices:

```bash
python stereo_obstacle_module.py --calib stereo_calib.npz --left_src 0 --right_src 1
```

---

### Debug Visualization (Optional)

```bash
python stereo_obstacle_module.py --calib stereo_calib.npz --src 0 --show
```

Press **`q`** to exit.

---

## Risk Logic (Simplified)

| Condition | Risk Level |
|--------|------------|
| Distance < 1.0 m **OR** TTC < 1.0 s | `DANGER` |
| Distance < 2.0 m **OR** TTC < 2.0 s | `CAUTION` |
| Otherwise | `SAFE` |

Additional safeguards:
- Confidence checks for unreliable depth
- Warning cooldown to prevent audio spam
- Risk persistence across multiple frames

---

## Example Audio Feedback

- `"Obstacle center. 1.3 meters."`
- `"Stop. Obstacle left. 0.7 meters."`
- Silence when the path is safe

---

## Known Limitations

- Reduced performance in:
  - Low-light conditions
  - Textureless surfaces
  - Strong glare or reflections
- Requires sufficient ambient lighting
- Does not perform object classification (by design)

---

## Planned Extensions

- Vertical zoning (head / torso / leg)
- Adaptive risk thresholds based on walking speed
- Optional lightweight segmentation for depth cleanup
- Haptic feedback integration
- Android deployment support

---

## Research Context

This module is part of a broader research effort on **smart-glasses–based assistive navigation systems** that prioritize:

- Depth-based reasoning
- Conservative risk modeling
- Human-centered feedback design

---

## License

This project is intended for **research and educational use**.  
License to be defined.

---

## Contact

For questions, collaboration, or research inquiries, please contact the project maintainer.

