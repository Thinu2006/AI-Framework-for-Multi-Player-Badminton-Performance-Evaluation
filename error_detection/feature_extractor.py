"""
error_detection/feature_extractor.py

Extracts frame-level and phase-level biomechanical features from smoothed pose sequences.
Computes torso-normalized distances, joint angles, wrist height, and arm extension deltas.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _joint_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Optional[float]:
    """Angle ABC at vertex B in degrees."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return None
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def extract_biomechanical_features(
    smoothed_pose: List[Dict[str, Any]],
    phases: Dict[str, Any],
    contact_frame: int,
    hitter: str = "far_player"
) -> Dict[str, Any]:
    """
    Extracts structured biomechanical features across phases and key frames.

    Returns:
        {
            "torso_length": float,
            "prepare": {...},
            "backswing": {...},
            "forward_swing": {...},
            "contact": {...},
            "follow_through": {...}
        }
    """
    if not smoothed_pose:
        return {}

    # 1. Compute median torso length across valid frames
    torso_lengths = []
    for item in smoothed_pose:
        p = item.get("pose", {})
        ls, rs = p.get("left_shoulder"), p.get("right_shoulder")
        lh, rh = p.get("left_hip"), p.get("right_hip")
        if all(k and k.get("score", 0) >= 0.20 for k in [ls, rs, lh, rh]):
            m_sh = ((ls["x"] + rs["x"]) / 2, (ls["y"] + rs["y"]) / 2)
            m_hp = ((lh["x"] + rh["x"]) / 2, (lh["y"] + rh["y"]) / 2)
            torso_lengths.append(_dist(m_sh, m_hp))

    torso_len = float(np.median(torso_lengths)) if torso_lengths else 100.0
    torso_len = max(20.0, torso_len)

    # Helper to get frame pose item
    frame_map = {item.get("frame"): item for item in smoothed_pose}

    def _get_hitter_arm_pts(pose: Dict[str, Any]):
        rw = pose.get("right_wrist")
        lw = pose.get("left_wrist")
        r_sc = rw.get("score", 0.0) if isinstance(rw, dict) else 0.0
        l_sc = lw.get("score", 0.0) if isinstance(lw, dict) else 0.0
        if r_sc >= l_sc and r_sc > 0:
            return pose.get("right_shoulder"), pose.get("right_elbow"), rw, pose.get("right_hip")
        return pose.get("left_shoulder"), pose.get("left_elbow"), lw, pose.get("left_hip")

    # Feature calculations for PREPARE phase
    prep_range = phases.get("prepare", {})
    prep_frames = [
        frame_map[f] for f in range(prep_range.get("start", 0), prep_range.get("end", 0) + 1)
        if f in frame_map
    ]

    prep_in_front_vals = []
    prep_elbow_height_vals = []
    knee_angles = []

    for item in prep_frames:
        pose = item.get("pose", {})
        sh, el, wr, hp = _get_hitter_arm_pts(pose)
        head = pose.get("nose")

        if sh and wr and hp and all(k.get("score", 0) >= 0.20 for k in [sh, wr, hp]):
            # In front of torso check: x-distance or y-height relative to shoulder/hip
            # Image y increases downward. Wr in front = wrist extending past torso
            in_front = (hp["y"] - wr["y"]) / torso_len
            prep_in_front_vals.append(in_front)

        if sh and el and all(k.get("score", 0) >= 0.20 for k in [sh, el]):
            # Elbow height relative to shoulder: (sh.y - el.y) / torso_len (positive means elbow above shoulder)
            elb_h = (sh["y"] - el["y"]) / torso_len
            prep_elbow_height_vals.append(elb_h)

        lk, rk = pose.get("left_knee"), pose.get("right_knee")
        lh, rh = pose.get("left_hip"), pose.get("right_hip")
        la, ra = pose.get("left_ankle"), pose.get("right_ankle")

        if lh and lk and la and all(k.get("score", 0) >= 0.20 for k in [lh, lk, la]):
            ang = _joint_angle((lh["x"], lh["y"]), (lk["x"], lk["y"]), (la["x"], la["y"]))
            if ang is not None:
                knee_angles.append(ang)

    # Feature calculations for FORWARD SWING & CONTACT
    contact_item = frame_map.get(contact_frame, smoothed_pose[len(smoothed_pose) // 2])
    c_pose = contact_item.get("pose", {})
    c_sh, c_el, c_wr, c_hp = _get_hitter_arm_pts(c_pose)
    c_head = c_pose.get("nose")

    wrist_h_head = 0.0
    c_elbow_angle = 180.0

    if c_wr and c_head and c_wr.get("score", 0) >= 0.20 and c_head.get("score", 0) >= 0.20:
        # In image coordinates, y increases downwards. (head.y - wr.y) / torso_len > 0 means wrist above head
        wrist_h_head = (c_head["y"] - c_wr["y"]) / torso_len

    if c_sh and c_el and c_wr and all(k.get("score", 0) >= 0.20 for k in [c_sh, c_el, c_wr]):
        ang = _joint_angle((c_sh["x"], c_sh["y"]), (c_el["x"], c_el["y"]), (c_wr["x"], c_wr["y"]))
        if ang is not None:
            c_elbow_angle = ang

    # Backswing elbow angle
    back_range = phases.get("backswing", {})
    back_frames = [
        frame_map[f] for f in range(back_range.get("start", 0), back_range.get("end", 0) + 1)
        if f in frame_map
    ]

    back_elbow_angles = []
    for item in back_frames:
        pose = item.get("pose", {})
        sh, el, wr, _ = _get_hitter_arm_pts(pose)
        if sh and el and wr and all(k.get("score", 0) >= 0.20 for k in [sh, el, wr]):
            ang = _joint_angle((sh["x"], sh["y"]), (el["x"], el["y"]), (wr["x"], wr["y"]))
            if ang is not None:
                back_elbow_angles.append(ang)

    min_back_angle = min(back_elbow_angles) if back_elbow_angles else 120.0
    elbow_ext_delta = c_elbow_angle - min_back_angle

    # Path length during forward swing
    fwd_range = phases.get("forward_swing", {})
    fwd_frames = [
        frame_map[f] for f in range(fwd_range.get("start", 0), fwd_range.get("end", 0) + 1)
        if f in frame_map
    ]

    fwd_path_len = 0.0
    for i in range(1, len(fwd_frames)):
        _, _, w_prev, _ = _get_hitter_arm_pts(fwd_frames[i - 1].get("pose", {}))
        _, _, w_curr, _ = _get_hitter_arm_pts(fwd_frames[i].get("pose", {}))
        if w_prev and w_curr:
            fwd_path_len += _dist((w_prev["x"], w_prev["y"]), (w_curr["x"], w_curr["y"]))

    norm_fwd_path = fwd_path_len / torso_len

    # Vertical wrist displacement between backswing and contact
    v_disp = 0.0
    if back_frames and c_wr:
        _, _, b_wr, _ = _get_hitter_arm_pts(back_frames[0].get("pose", {}))
        if b_wr:
            # (b_wr.y - c_wr.y) / torso_len > 0 means wrist moved upward
            v_disp = (b_wr["y"] - c_wr["y"]) / torso_len

    return {
        "torso_length": round(torso_len, 2),
        "hitting_wrist_in_front_of_torso": round(float(np.median(prep_in_front_vals)), 3) if prep_in_front_vals else 0.08,
        "elbow_height_relative_to_shoulder": round(float(np.median(prep_elbow_height_vals)), 3) if prep_elbow_height_vals else 0.02,
        "knee_flexion_angle": round(float(np.median(knee_angles)), 1) if knee_angles else 160.0,
        "wrist_height_relative_to_head": round(wrist_h_head, 3),
        "elbow_extension_delta": round(elbow_ext_delta, 1),
        "normalised_wrist_path_length": round(norm_fwd_path, 3),
        "vertical_wrist_displacement": round(v_disp, 3),
        "recovery_wrist_height_gap": 0.15
    }
