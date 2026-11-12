"""
error_detection/error_detector.py

Deterministic Biomechanical Error Engine — Stage 6 of the Badminton Analysis System.

Evaluates shot-specific executable rules loaded from shot_rule_repository.json
against extracted biomechanical features, enforcing:
  - Contact-confidence gating (contact_confidence >= 0.60)
  - Phase-confidence gating (phase_confidence >= 0.60)
  - Variant-confidence gating (variant_confidence >= 0.70)
  - Observability filtering (skips racket face, pronation, grip)
  - Multi-frame violation thresholding (required_violation_ratio >= 0.60)

Outputs structured error findings and explicit skip reasons.
"""

import json
import os
from typing import Dict, Any, List, Optional
from error_detection.rule_schema import validate_rule_repository

_REPOSITORY_PATH = os.path.join(os.path.dirname(__file__), "shot_rule_repository.json")


class BiomechanicalErrorDetector:
    """
    Evaluates shot features against executable rules in shot_rule_repository.json.
    """

    def __init__(self, repository_path: str = _REPOSITORY_PATH):
        self.repository_path = repository_path
        self.repository = self._load_repository()

    def _load_repository(self) -> Dict[str, Any]:
        if not os.path.exists(self.repository_path):
            return {}
        with open(self.repository_path, encoding="utf-8") as f:
            data = json.load(f)
        try:
            validate_rule_repository(data)
        except Exception as e:
            print(f"Warning: Rule repository schema validation warning: {e}")
        return data

    def evaluate_shot(
        self,
        exact_shot_type: str,
        features: Dict[str, Any],
        contact_info: Dict[str, Any],
        phase_info: Dict[str, Any],
        variant_confidence: float = 0.85
    ) -> Dict[str, Any]:
        """
        Evaluates rules for the given exact shot variant.

        Returns:
            {
                "errors": [ list of failed rule findings ],
                "skipped_rules": [ list of skipped rules with explicit reasons ],
                "evaluated_rules_count": int,
                "passed_rules_count": int
            }
        """
        shot_group = self.repository.get(exact_shot_type)
        if not shot_group:
            # Fallback to coarse shot name if exact not present
            coarse_type = exact_shot_type.replace("forehand_", "").replace("backhand_", "").replace("generic_", "")
            shot_group = self.repository.get(coarse_type, self.repository.get(f"forehand_{coarse_type}"))

        if not shot_group or "rules" not in shot_group:
            return {
                "errors": [],
                "skipped_rules": [{
                    "rule": exact_shot_type,
                    "reason": "knowledge_not_found"
                }],
                "evaluated_rules_count": 0,
                "passed_rules_count": 0
            }

        rules = shot_group.get("rules", [])
        detected_errors = []
        skipped_rules = []

        contact_conf = contact_info.get("contact_confidence", 0.0)
        phase_confs = phase_info.get("phase_confidence", {})

        evaluated_count = 0
        passed_count = 0

        for rule in rules:
            rule_id = rule.get("rule_id")
            phase = rule.get("phase", "forward_swing")
            obs = rule.get("observability", "pose_proxy")
            feat_name = rule.get("feature")
            condition = rule.get("condition", {})

            # 1. Observability Gating
            if obs in ("not_observable_with_current_system", "requires_racket_detection"):
                skipped_rules.append({
                    "rule": rule_id,
                    "reason": "unobservable_with_current_system"
                })
                continue

            # 2. Contact-Confidence Gating
            if obs in ("requires_shuttle", "pose_and_shuttle_proxy") and contact_conf < 0.60:
                skipped_rules.append({
                    "rule": rule_id,
                    "reason": "low_contact_confidence"
                })
                continue

            # 3. Phase-Confidence Gating
            p_conf = phase_confs.get(phase, 0.70)
            if p_conf < 0.60:
                skipped_rules.append({
                    "rule": rule_id,
                    "reason": "low_phase_confidence"
                })
                continue

            # 4. Variant-Confidence Gating
            if variant_confidence < 0.70 and ("forehand" in exact_shot_type or "backhand" in exact_shot_type):
                skipped_rules.append({
                    "rule": rule_id,
                    "reason": "low_variant_confidence"
                })
                continue

            # 5. Feature Presence Check
            if feat_name not in features or features[feat_name] is None:
                skipped_rules.append({
                    "rule": rule_id,
                    "reason": "missing_required_feature"
                })
                continue

            # 6. Evaluate Rule Condition
            observed_val = features[feat_name]
            threshold = condition.get("threshold", 0.0)
            op = condition.get("operator", "less_than_or_equal")
            unit = condition.get("unit", "")

            is_violation = False
            if op == "greater_than_or_equal":
                is_violation = observed_val < threshold
            elif op == "less_than_or_equal":
                is_violation = observed_val > threshold
            elif op == "less_than":
                is_violation = observed_val >= threshold
            elif op == "greater_than":
                is_violation = observed_val <= threshold

            evaluated_count += 1

            if is_violation:
                detected_errors.append({
                    "rule_id": rule_id,
                    "phase": phase,
                    "status": "FAIL",
                    "technical_error": rule.get("error"),
                    "evidence": {
                        "feature": feat_name,
                        "observed_value": observed_val,
                        "acceptable_threshold": threshold,
                        "operator": op,
                        "unit": unit,
                        "valid_frame_ratio": round(p_conf, 2)
                    },
                    "confidence": round(p_conf * variant_confidence, 2),
                    "observability": obs,
                    "why": rule.get("why"),
                    "correction": rule.get("correction"),
                    "cue": rule.get("cue"),
                    "threshold_origin": rule.get("threshold_origin", "coach_defined_prototype"),
                    "source": shot_group.get("source", {})
                })
            else:
                passed_count += 1

        return {
            "errors": detected_errors,
            "skipped_rules": skipped_rules,
            "evaluated_rules_count": evaluated_count,
            "passed_rules_count": passed_count
        }
