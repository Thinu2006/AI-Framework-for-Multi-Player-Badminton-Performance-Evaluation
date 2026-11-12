"""
error_detection/shot_variant_resolver.py

Resolves coarse shot types (e.g., net_shot, lift) into exact shot variants 
(e.g., forehand_net_shot, backhand_net_shot, forehand_smash).
Calculates variant_confidence; if confidence is low (< 0.70), side-specific rules are bypassed.
"""

import json
import os
from typing import Dict, Any, Optional

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "handedness_config.json")


def load_handedness_config() -> Dict[str, Any]:
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"near_player": {"handedness": "right"}, "far_player": {"handedness": "right"}}


def resolve_exact_shot_variant(
    coarse_shot_type: str,
    hitter: str,
    hitter_zone: str = "unknown",
    keypoints_seq: Dict[str, Any] = None,
    handedness_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Resolves exact shot variant (e.g., 'forehand_net_shot', 'backhand_net_lift', 'forehand_smash').
    When a player from left_rear, right_rear, or any rear zone plays a drop, smash, or clear,
    it strictly maps to forehand_drop, forehand_smash, and forehand_clear rules.

    Returns:
        {
            "coarse_shot_type": str,
            "stroke_side": "forehand" | "backhand" | "unknown",
            "exact_shot_type": str,
            "variant_confidence": float
        }
    """
    clean_shot = str(coarse_shot_type).lower()
    clean_zone = str(hitter_zone).lower()
    is_rear_zone = "rear" in clean_zone or clean_zone in ("left_rear", "right_rear", "far_left_rear", "far_right_rear", "near_left_rear", "near_right_rear")

    # Enforce rear-zone rule mapping for drop, smash, and clear
    if "drop" in clean_shot or (is_rear_zone and "drop" in clean_shot):
        return {
            "coarse_shot_type": coarse_shot_type,
            "stroke_side": "forehand",
            "exact_shot_type": "forehand_drop",
            "variant_confidence": 0.95
        }
    if "smash" in clean_shot or (is_rear_zone and "smash" in clean_shot):
        return {
            "coarse_shot_type": coarse_shot_type,
            "stroke_side": "forehand",
            "exact_shot_type": "forehand_smash",
            "variant_confidence": 0.95
        }
    if "clear" in clean_shot or (is_rear_zone and "clear" in clean_shot):
        return {
            "coarse_shot_type": coarse_shot_type,
            "stroke_side": "forehand",
            "exact_shot_type": "forehand_clear",
            "variant_confidence": 0.95
        }

    cfg = load_handedness_config()
    player_handedness = handedness_override or cfg.get(hitter, {}).get("handedness", "right")

    if player_handedness not in ("right", "left"):
        # Unknown handedness -> cannot resolve forehand vs backhand with confidence
        exact = f"generic_{coarse_shot_type}" if not coarse_shot_type.startswith("forehand") and not coarse_shot_type.startswith("backhand") else coarse_shot_type
        return {
            "coarse_shot_type": coarse_shot_type,
            "stroke_side": "unknown",
            "exact_shot_type": exact,
            "variant_confidence": 0.50
        }

    # If coarse shot is already specific (e.g. forehand_smash), preserve it
    if coarse_shot_type.startswith("forehand_") or coarse_shot_type.startswith("backhand_"):
        side = "forehand" if coarse_shot_type.startswith("forehand_") else "backhand"
        return {
            "coarse_shot_type": coarse_shot_type,
            "stroke_side": side,
            "exact_shot_type": coarse_shot_type,
            "variant_confidence": 0.90
        }

    # Determine stroke side using hitter zone & player orientation
    # Near player faces UP (towards far court), Far player faces DOWN (towards near court)
    stroke_side = "forehand"  # Default assumption
    confidence = 0.75

    is_left_court = "left" in hitter_zone.lower()
    is_right_court = "right" in hitter_zone.lower()

    if player_handedness == "right":
        if hitter == "near_player":
            # Right-handed near player facing net: right side = forehand, left side = backhand
            if is_left_court:
                stroke_side = "backhand"
                confidence = 0.82
            elif is_right_court:
                stroke_side = "forehand"
                confidence = 0.85
        elif hitter == "far_player":
            # Right-handed far player facing camera: left side of image (their right) = forehand, right side of image = backhand
            if is_right_court:
                stroke_side = "backhand"
                confidence = 0.82
            elif is_left_court:
                stroke_side = "forehand"
                confidence = 0.85

    # Build exact shot name
    exact_map = {
        "net_shot": f"{stroke_side}_net_shot",
        "lift": f"{stroke_side}_net_lift",
        "smash": "forehand_smash",
        "drop": "forehand_drop",
        "clear": "forehand_clear",
        "drive": f"{stroke_side}_drive",
    }

    exact_shot = exact_map.get(coarse_shot_type, f"{stroke_side}_{coarse_shot_type}")

    return {
        "coarse_shot_type": coarse_shot_type,
        "stroke_side": stroke_side,
        "exact_shot_type": exact_shot,
        "variant_confidence": round(confidence, 2)
    }
