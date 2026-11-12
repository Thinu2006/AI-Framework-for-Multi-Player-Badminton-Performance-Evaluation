"""
error_detection/phase_segmenter.py

Segments a shot sequence into temporal biomechanical phases:
  - prepare
  - backswing
  - forward_swing
  - contact
  - follow_through

Also applies Savitzky-Golay smoothing across continuous valid keypoint series,
producing `smoothed_pose` without mutating `raw_pose`.
Computes `phase_confidence` for each phase.
"""

import numpy as np
from typing import Dict, Any, List, Optional
try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except ImportError:
    HAS_SAVGOL = False


def smooth_pose_sequence(frame_poses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applies Savitzky-Golay filter (or moving average fallback) to continuous keypoints
    without altering raw_pose values.
    """
    if not frame_poses:
        return []

    # Deep copy structure for smoothed pose
    smoothed = []
    for item in frame_poses:
        smoothed.append({
            "frame": item.get("frame"),
            "pose": {k: dict(v) for k, v in item.get("pose", {}).items()},
            "shuttle": dict(item.get("shuttle")) if item.get("shuttle") else None
        })

    if len(smoothed) < 5:
        return smoothed

    # Collect keypoint names present
    sample_pose = smoothed[0].get("pose", {})
    kp_names = list(sample_pose.keys())

    for kp in kp_names:
        xs = [f["pose"].get(kp, {}).get("x") for f in smoothed]
        ys = [f["pose"].get(kp, {}).get("y") for f in smoothed]
        scores = [f["pose"].get(kp, {}).get("score", 0.0) for f in smoothed]

        valid_mask = [x is not None and score >= 0.20 for x, score in zip(xs, scores)]
        if sum(valid_mask) >= 5:
            # Smooth valid segments
            x_arr = np.array([x if x is not None else 0.0 for x in xs], dtype=np.float64)
            y_arr = np.array([y if y is not None else 0.0 for y in ys], dtype=np.float64)

            win = min(5, len(x_arr))
            if win % 2 == 0:
                win -= 1

            if HAS_SAVGOL and win >= 3:
                try:
                    sx = savgol_filter(x_arr, window_length=win, polyorder=2)
                    sy = savgol_filter(y_arr, window_length=win, polyorder=2)
                except Exception:
                    sx, sy = x_arr, y_arr
            else:
                # Moving average fallback
                kernel = np.ones(win) / win
                sx = np.convolve(x_arr, kernel, mode='same')
                sy = np.convolve(y_arr, kernel, mode='same')

            for idx in range(len(smoothed)):
                if valid_mask[idx]:
                    smoothed[idx]["pose"][kp]["x"] = round(float(sx[idx]), 2)
                    smoothed[idx]["pose"][kp]["y"] = round(float(sy[idx]), 2)

    return smoothed


def segment_shot_phases(
    start_frame: int,
    end_frame: int,
    contact_frame: int,
    frame_poses: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Segments temporal shot movement into 4 main phases around contact.

    Returns:
        {
            "phases": {
                "prepare": {"start": int, "end": int},
                "backswing": {"start": int, "end": int},
                "forward_swing": {"start": int, "end": int},
                "contact": {"frame": int},
                "follow_through": {"start": int, "end": int}
            },
            "phase_confidence": {
                "prepare": float,
                "backswing": float,
                "forward_swing": float,
                "follow_through": float
            },
            "smoothed_pose": list
        }
    """
    smoothed = smooth_pose_sequence(frame_poses)
    total_len = end_frame - start_frame + 1

    # Clamp contact frame to valid bounds
    c_frame = max(start_frame + 1, min(end_frame - 1, contact_frame))

    # Calculate wrist displacement to find maximum backswing point
    wrist_xs = []
    frames = []
    for item in smoothed:
        f = item.get("frame", start_frame)
        pose = item.get("pose", {})
        rw = pose.get("right_wrist")
        lw = pose.get("left_wrist")
        rw_sc = rw.get("score", 0.0) if isinstance(rw, dict) else 0.0
        lw_sc = lw.get("score", 0.0) if isinstance(lw, dict) else 0.0
        w = rw if (rw_sc >= lw_sc and rw_sc > 0) else lw
        if isinstance(w, dict) and w.get("score", 0.0) >= 0.20:
            wrist_xs.append(w["x"])
        else:
            wrist_xs.append(None)
        frames.append(f)

    # Estimate backswing boundary (max backward wrist displacement prior to contact)
    c_idx = frames.index(c_frame) if c_frame in frames else len(frames) // 2
    pre_contact_xs = wrist_xs[:c_idx]

    if any(x is not None for x in pre_contact_xs):
        # Peak backward wrist displacement index
        valid_indices = [i for i, x in enumerate(pre_contact_xs) if x is not None]
        backswing_idx = valid_indices[int(np.argmax([pre_contact_xs[i] for i in valid_indices]))]
    else:
        backswing_idx = max(1, c_idx // 2)

    backswing_frame = frames[backswing_idx] if backswing_idx < len(frames) else start_frame + 1

    # Define frame ranges
    prep_start = start_frame
    prep_end = max(prep_start, backswing_frame - 1)

    back_start = backswing_frame
    back_end = max(back_start, c_frame - 1)

    fwd_start = c_frame - 1 if c_frame > start_frame else c_frame
    fwd_end = c_frame

    fol_start = min(end_frame, c_frame + 1)
    fol_end = end_frame

    phases = {
        "prepare": {"start": prep_start, "end": prep_end},
        "backswing": {"start": back_start, "end": back_end},
        "forward_swing": {"start": fwd_start, "end": fwd_end},
        "contact": {"frame": c_frame},
        "follow_through": {"start": fol_start, "end": fol_end}
    }

    # Calculate phase confidence based on ratio of valid keypoint frames in each phase
    def _phase_conf(p_start, p_end):
        if p_start > p_end:
            return 0.50
        p_len = p_end - p_start + 1
        sub_poses = [p for p in smoothed if p_start <= p.get("frame", 0) <= p_end]
        valid_count = 0
        for sp in sub_poses:
            pose = sp.get("pose", {})
            rw = pose.get("right_wrist")
            lw = pose.get("left_wrist")
            r_sc = rw.get("score", 0.0) if isinstance(rw, dict) else 0.0
            l_sc = lw.get("score", 0.0) if isinstance(lw, dict) else 0.0
            if max(r_sc, l_sc) >= 0.20:
                valid_count += 1
        ratio = valid_count / max(1, p_len)
        return min(0.95, max(0.40, round(0.50 + ratio * 0.45, 2)))

    phase_confidence = {
        "prepare": _phase_conf(prep_start, prep_end),
        "backswing": _phase_conf(back_start, back_end),
        "forward_swing": _phase_conf(fwd_start, fwd_end),
        "follow_through": _phase_conf(fol_start, fol_end)
    }

    return {
        "phases": phases,
        "phase_confidence": phase_confidence,
        "smoothed_pose": smoothed
    }
