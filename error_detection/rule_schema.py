"""
error_detection/rule_schema.py

Schema definition and validator for shot_rule_repository.json.
Supports Pydantic model validation with clean fallback if Pydantic is unavailable.
"""

from typing import Dict, Any, List, Optional

try:
    from pydantic import BaseModel, Field, field_validator

    class RuleCondition(BaseModel):
        operator: str
        threshold: float
        unit: str

    class RuleSource(BaseModel):
        document: str
        pages: List[int]

    class RuleMetadata(BaseModel):
        rule_id: str
        phase: str
        source_instruction: str
        observability: str
        feature: str
        condition: RuleCondition
        error: str
        why: str
        correction: str
        cue: str
        threshold_origin: str = "coach_defined_prototype"
        reference_sample_count: int = 0
        version: str = "1.0"
        source: Optional[RuleSource] = None

        @field_validator("observability")

        def validate_observability(cls, v):
            valid = {
                "directly_observable",
                "pose_proxy",
                "requires_shuttle",
                "requires_racket_detection",
                "not_observable_with_current_system",
                "pose_and_shuttle_proxy",
            }
            if v not in valid:
                raise ValueError(f"Invalid observability classification: {v}")
            return v

    class ShotRulesGroup(BaseModel):
        source: Optional[RuleSource] = None
        rules: List[RuleMetadata]

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


def validate_rule_repository(data: Dict[str, Any]) -> bool:
    """
    Validates shot_rule_repository dict against the schema.
    Raises ValueError if validation fails.
    """
    if not isinstance(data, dict):
        raise ValueError("Repository root must be a dictionary")

    valid_observabilities = {
        "directly_observable",
        "pose_proxy",
        "requires_shuttle",
        "requires_racket_detection",
        "not_observable_with_current_system",
        "pose_and_shuttle_proxy",
    }

    for shot_type, group in data.items():
        if not isinstance(group, dict) or "rules" not in group:
            raise ValueError(f"Shot group '{shot_type}' missing 'rules' list")
        
        for rule in group["rules"]:
            for req in ["rule_id", "phase", "feature", "condition", "error", "observability"]:
                if req not in rule:
                    raise ValueError(f"Rule in '{shot_type}' missing required field: '{req}'")
            
            obs = rule["observability"]
            if obs not in valid_observabilities:
                raise ValueError(f"Rule '{rule['rule_id']}' has invalid observability '{obs}'")
            
            cond = rule["condition"]
            if not isinstance(cond, dict) or "operator" not in cond or "threshold" not in cond:
                raise ValueError(f"Rule '{rule['rule_id']}' has invalid condition structure")

    return True
