"""
error_detection/contact_estimator.py

Estimates the exact racket-shuttle contact frame for a shot sequence.
Combines shuttle-wrist proximity, wrist velocity peaks, and shuttle trajectory directional shifts.
Falls back to wrist velocity proxy when shuttle coordinates are missing.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def estimate_contact_frame(
    start_frame: int,
    end_frame: int,
    frame_poses: List[Dict[str, Any]],
    shuttle_frames: Optional[List[Dict[str, Any]]] = None,
    hitter: str = "far_player"
) -> Dict[str, Any]:
    """
    Estimates contact frame and returns contact confidence + method used.

    Returns:
        {
            "contact_frame": int,
            "contact_confidence": float,
            "method": list or str
        }
    """
    if start_frame >= end_frame or not frame_poses:
        mid = (start_frame + end_frame) // 2
        return {
            "contact_frame": mid,
            "contact_confidence": 0.30,
            "method": "midpoint_fallback"
        }

    frames = list(range(start_frame, end_frame + 1))
    n_frames = len(frames)

    # 1. Extract wrist positions & torso lengths across frames
    wrist_pts = []
    torso_lengths = []
    wrist_scores = []

    for item in frame_poses:
        pose = item.get("pose", {})
        ls = pose.get("left_shoulder")
        rs = pose.get("right_shoulder")
        lh = pose.get("left_hip")
        rh = pose.get("right_hip")
        rw = pose.get("right_wrist")
        lw = pose.get("left_wrist")
        rw_score = rw.get("score", 0.0) if isinstance(rw, dict) else 0.0
        lw_score = lw.get("score", 0.0) if isinstance(lw, dict) else 0.0
        hitting_w = rw if (rw_score >= lw_score and rw_score > 0) else lw
        if isinstance(hitting_w, dict) and hitting_w.get("score", 0.0) >= 0.20:
            wrist_pts.append((hitting_w["x"], hitting_w["y"]))
            wrist_scores.append(hitting_w["score"])
        else:
            wrist_pts.append(None)
            wrist_scores.append(0.0)

        # Torso length normalization
        if ls and rs and lh and rh and all(k.get("score", 0) >= 0.20 for k in [ls, rs, lh, rh]):
            mid_sh = ((ls["x"] + rs["x"]) / 2, (ls["y"] + rs["y"]) / 2)
            mid_hp = ((lh["x"] + rh["x"]) / 2, (lh["y"] + rh["y"]) / 2)
            t_len = _dist(mid_sh, mid_hp)
            torso_lengths.append(max(t_len, 20.0))
        else:
            torso_lengths.append(100.0)  # Default fallback torso length in pixels

    avg_torso = float(np.median(torso_lengths)) if torso_lengths else 100.0

    # Calculate wrist velocities
    wrist_speeds = [0.0] * n_frames
    for i in range(1, n_frames - 1):
        p_prev, p_next = wrist_pts[i - 1], wrist_pts[i + 1]
        if p_prev and p_next:
            spd = _dist(p_prev, p_next) / 2.0
            wrist_speeds[i] = spd

    # 2. Check if valid shuttle frames exist
    valid_shuttle = False
    shuttle_map = {}
    if shuttle_frames:
        for s in shuttle_frames:
            if s.get("confidence", 0) >= 0.40 and s.get("x") is not None and s.get("y") is not None:
                shuttle_map[s["frame"]] = (s["x"], s["y"], s["confidence"])

    if len(shuttle_map) >= max(3, int(n_frames * 0.3)):
        valid_shuttle = True

    if not valid_shuttle:
        # Fallback to wrist velocity peak proxy
        max_spd_idx = int(np.argmax(wrist_speeds)) if any(s > 0 for s in wrist_speeds) else (n_frames // 2)
        contact_f = frames[max_spd_idx]
        return {
            "contact_frame": contact_f,
            "contact_confidence": 0.49,
            "method": "wrist_velocity_proxy"
        }

    # 3. Full multi-modal contact scoring with Shuttle Data
    scores = [0.0] * n_frames
    proximities = [0.0] * n_frames
    shuttle_dirs = [0.0] * n_frames

    max_spd = max(wrist_speeds) if max(wrist_speeds) > 0 else 1.0

    for i, f in enumerate(frames):
        w_pt = wrist_pts[i]
        s_data = shuttle_map.get(f)

        # Proximity score
        if w_pt and s_data:
            s_pt = (s_data[0], s_data[1])
            raw_d = _dist(w_pt, s_pt)
            norm_d = raw_d / avg_torso
            proximities[i] = 1.0 / (1.0 + norm_d)

        # Shuttle direction change
        if i >= 1 and i < n_frames - 1:
            s_prev = shuttle_map.get(frames[i - 1])
            s_curr = s_data
            s_next = shuttle_map.get(frames[i + 1])
            if s_prev and s_curr and s_next:
                v1 = (s_curr[0] - s_prev[0], s_curr[1] - s_prev[1])
                v2 = (s_next[0] - s_curr[0], s_next[1] - s_curr[1])
                mag1 = math.sqrt(v1[0]**2 + v1[1]**2) + 1e-6
                mag2 = math.sqrt(v2[0]**2 + v2[1]**2) + 1e-6
                dot = (v1[0]*v2[0] + v1[1]*v2[1]) / (mag1 * mag2)
                shuttle_dirs[i] = 1.0 - max(-1.0, min(1.0, dot))  # Higher change = higher score

    for i in range(n_frames):
        w_spd_score = wrist_speeds[i] / max_spd
        prox_score = proximities[i]
        dir_score = shuttle_dirs[i]
        scores[i] = 0.45 * prox_score + 0.25 * w_spd_score + 0.30 * dir_score

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    contact_f = frames[best_idx]
    confidence = min(0.95, max(0.60, round(0.50 + best_score * 0.45, 2)))

    return {
        "contact_frame": contact_f,
        "contact_confidence": confidence,
        "method": [
            "shuttle_wrist_proximity",
            "wrist_velocity_peak",
            "shuttle_velocity_change"
        ]
    }
