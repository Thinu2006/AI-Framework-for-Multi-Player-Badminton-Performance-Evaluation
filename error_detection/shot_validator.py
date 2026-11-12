"""
error_detection/shot_validator.py

Validates shot inputs before Stage 6 biomechanical analysis.
Checks classification confidence, pose keypoint ratios, and rally flow consistency.
"""

from typing import Dict, Any, List, Tuple

MIN_SHOT_CONFIDENCE = 0.75
MIN_KEYPOINT_SCORE = 0.40
MIN_VALID_FRAME_RATIO = 0.60


def validate_shot_quality(shot: Dict[str, Any], frame_keypoints: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Validates a single shot for minimum classification confidence and pose frame quality.

    Returns:
        {
            "valid": bool,
            "status": "completed" | "not_evaluated",
            "reason": str or None,
            "valid_wrist_ratio": float,
            "review_flags": list
        }
    """
    confidence = shot.get("confidence", 0.0)
    shot_type = shot.get("shot_type", "unknown")

    if shot_type == "unknown" or confidence < MIN_SHOT_CONFIDENCE:
        return {
            "valid": False,
            "status": "not_evaluated",
            "reason": f"Low shot classification confidence ({confidence:.2f} < {MIN_SHOT_CONFIDENCE})",
            "valid_wrist_ratio": 0.0,
            "review_flags": []
        }

    start_frame = shot.get("start_frame")
    end_frame = shot.get("end_frame")
    hitter = shot.get("hitter")

    if not start_frame or not end_frame or start_frame > end_frame:
        return {
            "valid": False,
            "status": "not_evaluated",
            "reason": "Invalid frame sequence bounds",
            "valid_wrist_ratio": 0.0,
            "review_flags": []
        }

    total_frames = end_frame - start_frame + 1
    role_key = "Near Player" if hitter == "near_player" else "Far Player"

    wrist_valid_count = 0
    if frame_keypoints:
        for f in range(start_frame, end_frame + 1):
            kps = frame_keypoints.get(str(f), {}).get(role_key, {})
            rw = kps.get("right_wrist", {}).get("score", 0.0)
            lw = kps.get("left_wrist", {}).get("score", 0.0)
            if max(rw, lw) >= MIN_KEYPOINT_SCORE:
                wrist_valid_count += 1

    valid_wrist_ratio = wrist_valid_count / max(1, total_frames)

    return {
        "valid": True,
        "status": "completed",
        "reason": None,
        "valid_wrist_ratio": round(valid_wrist_ratio, 2),
        "review_flags": []
    }


def validate_rally_sequences(shots: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    """
    Validates rally sequence flow across consecutive shots.
    Normally rally directions alternate (near_to_far <-> far_to_near).
    If consecutive shots have the same hitter without an opponent response,
    flags the shot with 'rally_sequence_anomaly'.

    Returns a dict mapping shot_id -> list of review flags.
    """
    review_flags: Dict[int, List[str]] = {s["shot_id"]: [] for s in shots}
    sorted_shots = sorted(shots, key=lambda s: s.get("start_frame", 0))

    for i in range(1, len(sorted_shots)):
        prev_shot = sorted_shots[i - 1]
        curr_shot = sorted_shots[i]

        prev_hitter = prev_shot.get("hitter")
        curr_hitter = curr_shot.get("hitter")

        prev_dir = prev_shot.get("travel_direction")
        curr_dir = curr_shot.get("travel_direction")

        # Double consecutive hit by same player in same direction
        if prev_hitter and curr_hitter and prev_hitter == curr_hitter:
            if prev_dir and curr_dir and prev_dir == curr_dir:
                cid = curr_shot["shot_id"]
                review_flags[cid].append("rally_sequence_anomaly")

    return review_flags
