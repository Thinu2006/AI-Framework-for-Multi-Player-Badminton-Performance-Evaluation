"""
RAG/stage7_feedback_pipeline.py

Stage 7 Pipeline Orchestrator — Grounded LLM Coaching Feedback Generation
========================================================================

Flow:
  1. Load Match/biomechanical_errors.json (Stage 6 output)
  2. Load RAG coaching knowledge (RAG/extracted_shots.json)
  3. For each evaluated shot:
       a. Perform deterministic exact shot lookup in RAG repository
       b. If exact lookup fails -> return 'knowledge_not_found' status (no semantic fallback for rules)
       c. Check 'feedback_allowed' flag (suppress if false due to review flags like sequence anomalies)
       d. Generate grounded coaching feedback via LLM (or deterministic fallback)
  4. Save Match/coaching_feedback.json
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llm_prompts import generate_shot_diagnostic, simplify_flawed_text, simplify_clean_text

_RAG_JSON_PATH = os.path.join(_PROJECT_ROOT, "RAG", "extracted_shots.json")
_LLM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "llm_config.json")

STRICT_GROUNDING_PROMPT = (
    "You are an experienced badminton coach, not an AI engineer. Your task is to convert biomechanical analysis "
    "into simple, concise, and actionable coaching feedback that a coach can understand in 5 seconds.\n\n"
    "Requirements:\n"
    "- Plain English that any badminton coach can understand.\n"
    "- NEVER include raw telemetry, numerical values, coordinate measurements, body ratios, thresholds, "
    "confidence scores, or technical feature names (e.g., 'Telemetry Evidence', 'Kinematic Diagnostics', "
    "'vertical_wrist_displacement', 'torso length').\n"
    "- Explain the mistake in coaching language rather than technical language.\n"
    "- Structure output strictly with:\n"
    "  1. Main Mistake (1 short sentence)\n"
    "  2. Why It Matters (1 short sentence on tactical consequence)\n"
    "  3. How to Correct It (1-3 short bullet points)\n"
    "  4. Coaching Tip (1 memorable coaching cue)\n"
    "- Include the stroke phase (Preparation, Backswing, Forward Swing, Follow Through) if available.\n"
    "- Prioritize immediate coaching action and positive practice focus."
)


def load_json(path: str) -> Any:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_rag_knowledge() -> List[Dict[str, Any]]:
    if os.path.exists(_RAG_JSON_PATH):
        with open(_RAG_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def exact_rag_lookup(exact_shot_type: str, rag_chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Performs exact deterministic lookup for matching shot object in RAG repository.
    Returns None if exact match is not found (prevents semantic hallucination of rules).
    """
    normalized_query = exact_shot_type.lower().replace("_", " ").strip()
    
    # Map common exact shot types to RAG shot_type strings
    name_map = {
        "forehand net shot": "Forehand Net Shot",
        "backhand net shot": "Backhand Net Shot",
        "forehand net lift": "Forehand Net Lift",
        "backhand net lift": "Backhand Net Lift",
        "forehand smash": "Smash",
        "backhand smash": "Smash",
        "smash": "Smash",
        "forehand drop": "Drop",
        "drop": "Drop",
        "forehand clear": "Clear",
        "clear": "Clear"
    }

    target_name = name_map.get(normalized_query)
    if not target_name:
        for chunk in rag_chunks:
            if chunk.get("shot_type", "").lower() == normalized_query:
                return chunk
        return None

    for chunk in rag_chunks:
        if chunk.get("shot_type", "").lower() == target_name.lower():
            return chunk
    return None


def clean_drill_text(raw_text: str) -> str:
    if not raw_text:
        return "Partner feeding practice and shadow swing repetitions."
    text = str(raw_text)
    text = text.replace("â€¢", "•").replace("â€“", "-").replace("â€”", "-")
    text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    
    parts = text.replace("•", "\n•").split("\n")
    cleaned_items = []
    for item in parts:
        c = item.strip("• \t\r\n-")
        if c and c not in cleaned_items and not c.endswith("..."):
            cleaned_items.append(c)
    
    if not cleaned_items:
        return "Partner feeding practice and shadow swing repetitions."
    
    # Pick top unique drill instructions
    selected = cleaned_items[:3]
    return " • ".join(selected)


def generate_grounded_coaching_feedback(
    shot_data: Dict[str, Any],
    rag_knowledge: Optional[Dict[str, Any]],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Translates rule engine error evidence and retrieved SLBA knowledge into structured coaching feedback.
    Strictly restricted from introducing unobserved errors.
    """
    sid = shot_data.get("shot_id")
    exact_shot = shot_data.get("exact_shot_type")
    errors = shot_data.get("errors", [])
    feedback_allowed = shot_data.get("feedback_allowed", True)
    review_flags = shot_data.get("review_flags", [])

    if not feedback_allowed or shot_data.get("analysis_status") == "not_evaluated":
        reason = shot_data.get("reason", "Suppressed due to review flags / low confidence")
        return {
            "shot_id": sid,
            "exact_shot_type": exact_shot,
            "status": "suppressed",
            "reason": reason,
            "review_flags": review_flags,
            "feedback": []
        }

    if not rag_knowledge:
        return {
            "shot_id": sid,
            "exact_shot_type": exact_shot,
            "status": "knowledge_not_found",
            "reason": f"Exact RAG coaching knowledge for '{exact_shot}' not found in repository.",
            "review_flags": review_flags,
            "feedback": []
        }

    if not errors:
        # No errors detected — generate positive grounded feedback based on evaluated rules
        skipped_rules = shot_data.get("skipped_rules", [])
        has_limitations = len(skipped_rules) > 0 or len(review_flags) > 0

        pos_happened = "Good technique was observed for this shot based on the analysed biomechanical features. The player's posture and movement satisfied the evaluated coaching rules."
        pos_todo = "Maintain this technique and continue practising for consistency."
        if has_limitations:
            pos_todo += " Note: Assessment is limited by the available analysis."

        raw_drill = rag_knowledge.get("structured_fields", {}).get("how_to_coach", "Continue standard shadow swing practice and footwork recovery drills.")
        drill_text = clean_drill_text(raw_drill)

        return {
            "shot_id": sid,
            "exact_shot_type": exact_shot,
            "status": "completed",
            "review_flags": review_flags,
            "coaching_feedback": {
                "what_happened": pos_happened,
                "what_to_do": pos_todo,
                "recommended_drill": drill_text
            },
            "feedback": [{
                "error": "None",
                "visual_evidence": pos_happened,
                "why_it_is_wrong": "Satisfied evaluated coaching rules.",
                "correction": pos_todo,
                "coaching_cue": "Maintain clean form",
                "drill": drill_text,
                "confidence_note": "High confidence pass across observable keypoints."
            }]
        }

    # Format feedback for each detected error
    feedback_items = []
    for err in errors:
        rule_id = err.get("rule_id")
        tech_err = err.get("technical_error") or err.get("error_type") or "Stroke Execution Deviation"
        phase = err.get("phase") or "Forward Swing"

        why_txt = err.get("why", "Creates defensive vulnerability and limits shuttle depth.")
        corr_txt = err.get("correction", "Adjust racket path and complete follow-through.")
        cue_txt = err.get("cue", "Lift high, then finish the swing.")

        raw_drill = rag_knowledge.get("structured_fields", {}).get("how_to_coach", "Partner feeding practice and shadow swing repetitions.")
        drill_text = clean_drill_text(raw_drill)

        # Build clean bullet points for how_to_correct
        if isinstance(corr_txt, list):
            how_to_correct_list = corr_txt
        else:
            # Parse or split into clean bullet points
            how_to_correct_list = [c.strip() for c in corr_txt.split(". ") if c.strip()]
            if not how_to_correct_list:
                how_to_correct_list = ["Make contact with the shuttle higher.", "Extend the arm fully during execution."]

        feedback_items.append({
            "phase": phase,
            "main_mistake": f"The player's {tech_err.lower()} reduced stroke effectiveness during the {phase.lower()} phase.",
            "why_it_matters": why_txt,
            "how_to_correct": how_to_correct_list,
            "coaching_tip": cue_txt,
            "drill": drill_text,
            # Legacy fallback keys for backward compatibility
            "rule_id": rule_id,
            "error": tech_err,
            "why_it_is_wrong": why_txt,
            "correction": corr_txt,
            "coaching_cue": cue_txt
        })

    return {
        "shot_id": sid,
        "exact_shot_type": exact_shot,
        "status": "completed",
        "review_flags": review_flags,
        "feedback": feedback_items
    }


def run_stage7_pipeline(
    errors_json_path: str = "Match/biomechanical_errors.json",
    output_dir: str = "Match",
    output_json_name: str = "coaching_feedback.json"
) -> Dict[str, Any]:
    """
    Executes Stage 7 Grounded LLM Coaching Feedback Pipeline.
    Writes Match/coaching_feedback.json.
    """
    print("--- Stage 7: Running Grounded Coaching Feedback Pipeline ---")

    errors_payload = load_json(errors_json_path)
    if not isinstance(errors_payload, dict) or "shots" not in errors_payload:
        print(f"[Stage 7] Error: Invalid biomechanical errors payload at {errors_json_path}")
        return {}

    shots = errors_payload.get("shots", [])
    rag_chunks = load_rag_knowledge()
    print(f"[Stage 7] Loaded {len(shots)} shots from Stage 6 and {len(rag_chunks)} RAG knowledge chunks.")

    feedback_records = []

    for shot in shots:
        sid = shot["shot_id"]
        exact_shot = shot.get("exact_shot_type", "unknown")
        print(f"\n[Stage 7] Processing Shot #{sid}: {exact_shot}")

        # Deterministic Exact RAG Lookup
        rag_match = exact_rag_lookup(exact_shot, rag_chunks)
        if rag_match:
            print(f"  Exact RAG Match: {rag_match['shot_type']} (pages {rag_match['page_start']}-{rag_match['page_end']})")
        else:
            print(f"  [Warning] No exact RAG match for '{exact_shot}' -> returning knowledge_not_found")

        fb_record = generate_grounded_coaching_feedback(shot, rag_match)

        # GENERATE PER-SHOT LLM DIAGNOSTIC
        if fb_record.get("status") == "completed":
            print(f"  [Stage 7] Generating LLM diagnostic for shot #{sid}...")
            errors = shot.get("errors", [])
            pipeline_fb = {}
            if not errors:
                pipeline_fb = fb_record.get("coaching_feedback", {})
            else:
                if fb_record.get("feedback"):
                    f_item = fb_record["feedback"][0]
                    pipeline_fb = {
                        "what_happened": f_item.get("main_mistake") or f_item.get("error"),
                        "how_to_improve": f_item.get("correction") or "",
                        "practice_drill": f_item.get("drill") or ""
                    }
            telemetry = ""
            if errors:
                ev = errors[0].get("visual_evidence") or errors[0].get("evidence") or ""
                telemetry = json.dumps(ev) if isinstance(ev, (dict, list)) else str(ev)

            wh, hti, pd = generate_shot_diagnostic(sid, exact_shot, errors, pipeline_fb, telemetry)
            fb_record["llm_diagnostic"] = {
                "what_happened": wh,
                "how_to_improve": hti,
                "practice_drill": pd
            }
        else:
             fb_record["llm_diagnostic"] = None

        feedback_records.append(fb_record)

    # GENERATE GROUP-LEVEL LLM DIAGNOSTIC
    group_diagnostics = {}
    grouped_shots = {}
    for shot in shots:
        exact_shot = shot.get("exact_shot_type", "unknown")
        if exact_shot not in grouped_shots:
            grouped_shots[exact_shot] = []
        grouped_shots[exact_shot].append(shot)

    print("\n[Stage 7] Generating group-level LLM diagnostics...")
    for exact_shot, group in grouped_shots.items():
        group_errors = []
        for shot in group:
            if shot.get("errors"):
                group_errors.extend(shot.get("errors"))
        
        if group_errors:
            first_err = group_errors[0]
            flaw_name = first_err.get("technical_error") or first_err.get("error_type") or "Technique Deviation"
            main_mistake = first_err.get("main_mistake") or first_err.get("why_it_is_wrong") or first_err.get("why", "")
            why_wrong = first_err.get("why_it_is_wrong") or first_err.get("why", "")
            correction = first_err.get("correction") or first_err.get("how_to_correct") or ""
            cue = first_err.get("coaching_cue") or first_err.get("coaching_tip") or first_err.get("cue", "")
            measured_evidence = first_err.get("evidence") or first_err.get("visual_evidence") or ""

            wh, hti, pd = simplify_flawed_text(
                exact_shot, flaw_name, main_mistake, why_wrong, correction, cue, measured_evidence
            )
        else:
            wh, hti, pd = simplify_clean_text(exact_shot)

        group_diagnostics[exact_shot] = {
            "what_happened": wh,
            "how_to_improve": hti,
            "practice_drill": pd
        }

    out_payload = {
        "schema_version": "1.0",
        "pipeline_version": "1.0",
        "group_diagnostics": group_diagnostics,
        "coaching_feedback": feedback_records
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_json_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[Stage 7] Saved coaching feedback payload to: {output_path}")
    return out_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7: Grounded LLM Coaching Feedback Pipeline")
    parser.add_argument("--errors", default="Match/biomechanical_errors.json")
    parser.add_argument("--outdir", default="Match")
    parser.add_argument("--outfile", default="coaching_feedback.json")
    args = parser.parse_args()

    run_stage7_pipeline(args.errors, args.outdir, args.outfile)
