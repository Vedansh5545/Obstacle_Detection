#!/usr/bin/env python3
"""
stereo_obstacle_module.py
Real-time stereo depth -> risk -> TTS warnings (starter skeleton).

Works on Windows / Linux (Jetson) as long as OpenCV can access the camera.

Assumptions:
- You have stereo calibration saved in an .npz with rectification maps.
- Camera provides either:
  (A) a single side-by-side frame (left|right), or
  (B) two separate camera indices (left_idx, right_idx)

Usage (single-line):
python stereo_obstacle_module.py --calib stereo_calib.npz --src 0
"""

import time
import argparse
from collections import deque

import numpy as np
import cv2

# Optional TTS (works on Windows; on Jetson it usually works too)
try:
    import pyttsx3
    TTS_OK = True
except Exception:
    TTS_OK = False


# -----------------------------
# Calibration Loading
# -----------------------------
def load_calib_npz(path: str):
    """
    Expected keys in npz:
      - mapLx, mapLy, mapRx, mapRy  (rectification maps)
      - Q (4x4 reprojection matrix) OR (f, B) for depth conversion
    """
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    required_maps = {"mapLx", "mapLy", "mapRx", "mapRy"}
    if not required_maps.issubset(keys):
        raise ValueError(f"Calibration file missing maps. Need {required_maps}, got {keys}")

    maps = {
        "mapLx": data["mapLx"], "mapLy": data["mapLy"],
        "mapRx": data["mapRx"], "mapRy": data["mapRy"],
    }

    Q = data["Q"] if "Q" in keys else None

    # Fallback: if no Q, you can store f and baseline B and compute Z = f*B / disparity
    f = float(data["f"]) if "f" in keys else None
    B = float(data["B"]) if "B" in keys else None

    if Q is None and (f is None or B is None):
        raise ValueError("Need either Q matrix OR (f and B) in calib npz.")

    return maps, Q, f, B


# -----------------------------
# Camera Reading
# -----------------------------
def open_camera(src: int):
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    if not cap.isOpened():
        # fallback
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source {src}")
    return cap


def split_side_by_side(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    left = frame[:, :mid]
    right = frame[:, mid:]
    return left, right


# -----------------------------
# Depth + Risk Helpers
# -----------------------------
def make_sgbm(min_disp=0, num_disp=128, block=5):
    # num_disp must be divisible by 16
    num_disp = int(np.ceil(num_disp / 16) * 16)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 1 * block * block,
        P2=32 * 1 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return sgbm


def disparity_to_depth(disparity, Q=None, f=None, B=None):
    """
    disparity: float32 disparity in pixels (same size as image)
    Returns depth in meters (approx), invalid where disparity <= 0
    """
    disp = disparity.copy()
    disp[disp <= 0.0] = np.nan

    if Q is not None:
        # Reproject to 3D: returns (x,y,z) in the Q's units
        points_3d = cv2.reprojectImageTo3D(disp, Q)  # shape (H,W,3)
        Z = points_3d[:, :, 2].astype(np.float32)
        return Z

    # If using f and B (baseline) in meters and f in pixels:
    Z = (f * B) / disp
    return Z.astype(np.float32)


def corridor_masks(h, w):
    """
    Simple 3 zones left/center/right in a central corridor region.
    Return boolean masks.
    """
    # corridor = middle 60% of width, full height (you can refine)
    x0 = int(0.20 * w)
    x1 = int(0.80 * w)

    # split into 3 vertical zones
    z0 = x0
    z1 = x0 + (x1 - x0) // 3
    z2 = x0 + 2 * (x1 - x0) // 3
    z3 = x1

    mask_left = np.zeros((h, w), dtype=bool);  mask_left[:, z0:z1] = True
    mask_center = np.zeros((h, w), dtype=bool); mask_center[:, z1:z2] = True
    mask_right = np.zeros((h, w), dtype=bool);  mask_right[:, z2:z3] = True

    return {"left": mask_left, "center": mask_center, "right": mask_right}


def zone_min_depth(depth_m, zone_mask):
    z = depth_m[zone_mask]
    if z.size == 0:
        return np.nan
    return np.nanpercentile(z, 5)  # robust nearest estimate


def compute_ttc(depth_hist, dt_hist):
    """
    depth_hist: deque of recent nearest distances (meters)
    dt_hist: deque of frame-to-frame dt (seconds)
    TTC ~ d / v, with v from slope.
    """
    if len(depth_hist) < 3:
        return np.inf

    d = np.array(depth_hist, dtype=np.float32)
    t = np.cumsum(np.array(dt_hist, dtype=np.float32))
    t = t - t[0]

    # fit d(t) ~ a*t + b, closing speed v = -a (if distance decreasing)
    try:
        a, b = np.polyfit(t, d, 1)
    except Exception:
        return np.inf

    v_closing = max(0.0, -a)  # m/s
    d_now = float(d[-1])

    if v_closing < 1e-3:
        return np.inf
    return d_now / v_closing


def risk_from(distance_m, ttc_s, confidence=1.0):
    """
    Simple conservative risk rules.
    """
    if not np.isfinite(distance_m):
        # no valid depth -> low confidence situation
        return "CAUTION"

    # confidence gate
    if confidence < 0.4:
        return "CAUTION"

    if distance_m < 1.0 or ttc_s < 1.0:
        return "DANGER"
    if distance_m < 2.0 or ttc_s < 2.0:
        return "CAUTION"
    return "SAFE"


# -----------------------------
# Feedback Policy (anti-spam)
# -----------------------------
class FeedbackPolicy:
    def __init__(self, cooldown_s=1.0, persist_frames=3):
        self.cooldown_s = cooldown_s
        self.persist_frames = persist_frames
        self.last_spoken_t = 0.0
        self.risk_buf = deque(maxlen=persist_frames)

    def should_speak(self, risk):
        now = time.time()
        self.risk_buf.append(risk)

        # require persistence
        if len(self.risk_buf) < self.persist_frames:
            return False

        persistent = all(r == self.risk_buf[-1] for r in self.risk_buf)
        if not persistent:
            return False

        # cooldown
        if (now - self.last_spoken_t) < self.cooldown_s:
            return False

        # speak only for caution/danger
        if risk in ("CAUTION", "DANGER"):
            self.last_spoken_t = now
            return True

        return False


def risk_message(zone, risk, dist):
    if risk == "DANGER":
        return f"Stop. Obstacle {zone}. {dist:.1f} meters."
    if risk == "CAUTION":
        return f"Obstacle {zone}. {dist:.1f} meters."
    return ""


# -----------------------------
# Main Loop
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, help="Path to stereo_calib.npz")
    ap.add_argument("--src", type=int, default=0, help="Camera source (side-by-side)")
    ap.add_argument("--left_src", type=int, default=None, help="Optional separate left cam index")
    ap.add_argument("--right_src", type=int, default=None, help="Optional separate right cam index")
    ap.add_argument("--show", action="store_true", help="Show debug windows")
    args = ap.parse_args()

    maps, Q, f, B = load_calib_npz(args.calib)
    sgbm = make_sgbm(num_disp=128, block=5)
    zones = None

    # TTS init
    engine = None
    if TTS_OK:
        engine = pyttsx3.init()
        engine.setProperty("rate", 185)

    policy = FeedbackPolicy(cooldown_s=1.0, persist_frames=3)

    # Camera init
    if args.left_src is not None and args.right_src is not None:
        capL = open_camera(args.left_src)
        capR = open_camera(args.right_src)
        sbs_mode = False
    else:
        cap = open_camera(args.src)
        sbs_mode = True
        capL = capR = None

    depth_hist = deque(maxlen=12)
    dt_hist = deque(maxlen=12)
    last_t = time.time()

    while True:
        t0 = time.time()

        if sbs_mode:
            ok, frame = cap.read()
            if not ok:
                break
            left, right = split_side_by_side(frame)
        else:
            okL, left = capL.read()
            okR, right = capR.read()
            if not (okL and okR):
                break

        # init zones once
        if zones is None:
            h, w = left.shape[:2]
            zones = corridor_masks(h, w)

        # rectify
        left_r = cv2.remap(left, maps["mapLx"], maps["mapLy"], cv2.INTER_LINEAR)
        right_r = cv2.remap(right, maps["mapRx"], maps["mapRy"], cv2.INTER_LINEAR)

        # grayscale for stereo
        gL = cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY)

        # disparity (OpenCV returns fixed-point disparity*16)
        disp = sgbm.compute(gL, gR).astype(np.float32) / 16.0

        # depth
        Z = disparity_to_depth(disp, Q=Q, f=f, B=B)

        # confidence: fraction of valid depth in corridor
        corridor_valid = 0.0
        corridor_pixels = 0
        for m in zones.values():
            zvals = Z[m]
            corridor_pixels += zvals.size
            corridor_valid += np.isfinite(zvals).sum()
        confidence = float(corridor_valid / max(1, corridor_pixels))

        # compute nearest per zone
        zone_d = {k: zone_min_depth(Z, m) for k, m in zones.items()}
        # choose most dangerous = smallest distance
        zone_best = min(zone_d.keys(), key=lambda k: np.nan_to_num(zone_d[k], nan=np.inf))
        d_best = float(zone_d[zone_best]) if np.isfinite(zone_d[zone_best]) else np.nan

        # ttc
        now = time.time()
        dt = now - last_t
        last_t = now
        if np.isfinite(d_best):
            depth_hist.append(d_best)
            dt_hist.append(dt)
        ttc = compute_ttc(depth_hist, dt_hist)

        # risk
        risk = risk_from(d_best, ttc, confidence=confidence)

        # speak
        if policy.should_speak(risk) and engine is not None and np.isfinite(d_best):
            msg = risk_message(zone_best, risk, d_best)
            if msg:
                engine.say(msg)
                engine.runAndWait()

        # debug view
        if args.show:
            disp_vis = disp.copy()
            disp_vis[disp_vis < 0] = 0
            disp_vis = (disp_vis / np.nanmax(disp_vis + 1e-6) * 255).astype(np.uint8)
            cv2.putText(left_r, f"risk={risk} zone={zone_best} d={d_best:.2f}m ttc={ttc:.2f}s conf={confidence:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("left_rect", left_r)
            cv2.imshow("disp", disp_vis)

        # exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # latency print
        t1 = time.time()
        loop_ms = (t1 - t0) * 1000.0
        # uncomment if you want logs:
        # print(f"loop {loop_ms:.1f} ms | risk {risk} | d {d_best:.2f} | ttc {ttc:.2f} | conf {confidence:.2f}")

    if sbs_mode:
        cap.release()
    else:
        capL.release(); capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
