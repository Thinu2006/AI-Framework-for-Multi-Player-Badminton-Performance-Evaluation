"""
error_detection/stage6_error_pipeline.py

Stage 6 Pipeline Orchestrator — Biomechanical Error Detection Engine
====================================================================

Flow:
  1. Load shot_predictions.json + frame_keypoints.json + shuttle_keypoints.json
  2. Run rally sequence flow validation -> detect review flags (e.g., rally_sequence_anomaly)
  3. For each shot:
       a. Validate shot confidence (>= 0.75) and keypoint quality
       b. Resolve exact shot variant (e.g. forehand_net_shot) + variant_confidence
       c. Run Contact Estimator (shuttle proximity, wrist speed, trajectory shift)
       d. Run Phase Segmenter + smooth pose coordinates (Savitzky-Golay)
       e. Extract temporal biomechanical features
       f. Run Biomechanical Error Detector (gating: contact, phase, variant, observability)
  4. Save Match/smoothed_pose.json
  5. Save Match/biomechanical_errors.json (input to Stage 7 Coaching Feedback Pipeline)
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, Any, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from error_detection.shot_validator import validate_shot_quality, validate_rally_sequences
from error_detection.shot_variant_resolver import resolve_exact_shot_variant
from error_detection.contact_estimator import estimate_contact_frame
from error_detection.phase_segmenter import segment_shot_phases
from error_detection.feature_extractor import extract_biomechanical_features
from error_detection.error_detector import BiomechanicalErrorDetector


def load_json(path: str) -> Any:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_stage6_pipeline(
    shots_json_path: str = "Match/shot_predictions.json",
    keypoints_json_path: str = "Match/frame_keypoints.json",
    shuttle_json_path: str = "Match/shuttle_keypoints.json",
    output_dir: str = "Match",
    output_json_name: str = "biomechanical_errors.json",
    report_enabled_roles: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes Stage 6 Biomechanical Error Detection.
    Processes both players for tracking and phase features, but saves biomechanical error outputs
    only for players matching report_enabled_roles.
    Writes Match/smoothed_pose.json and Match/biomechanical_errors.json.
    """
    print("--- Stage 6: Running Biomechanical Error Detection Pipeline ---")

    # Normalize allowed roles to canonical format: 'near_player', 'far_player'
    enabled_set = None
    if report_enabled_roles:
        enabled_set = set()
        for r in report_enabled_roles:
            r_clean = str(r).lower().replace(" ", "_")
            if "near" in r_clean:
                enabled_set.add("near_player")
            elif "far" in r_clean:
                enabled_set.add("far_player")
            else:
                enabled_set.add(r_clean)
        print(f"[Stage 6] Biomechanical error output enabled for roles: {enabled_set}")

    shots = load_json(shots_json_path)
    if not isinstance(shots, list):
        print(f"[Stage 6] Error: Invalid shots file format at {shots_json_path}")
        return {}

    frame_keypoints = load_json(keypoints_json_path)
    shuttle_keypoints = load_json(shuttle_json_path) if os.path.exists(shuttle_json_path) else {}

    print(f"[Stage 6] Loaded {len(shots)} shots from {shots_json_path}")

    # Sequence validation across rally
    rally_review_flags = validate_rally_sequences(shots)
    detector = BiomechanicalErrorDetector()

    all_smoothed_poses = {}
    shot_records = []

    for shot in shots:
        sid = shot["shot_id"]
        coarse_type = shot.get("shot_type", "unknown")
        shot_conf = shot.get("confidence", 0.0)
        hitter = shot.get("hitter", "far_player")
        hitter_zone = shot.get("hitter_zone", "unknown")
        start_f = shot.get("start_frame")
        end_f = shot.get("end_frame")

        print(f"\n[Stage 6] Shot #{sid}: {coarse_type.upper()} | hitter={hitter} | zone={hitter_zone} | frames={start_f}-{end_f}")

        # 1. Shot Quality Validation
        qual = validate_shot_quality(shot, frame_keypoints)
        if not qual["valid"]:
            print(f"  -> SKIPPED / NOT EVALUATED ({qual['reason']})")
            shot_records.append({
                "shot_id": sid,
                "player": hitter,
                "shot_type": coarse_type,
                "exact_shot_type": coarse_type,
                "shot_confidence": shot_conf,
                "variant_confidence": 0.0,
                "analysis_status": "not_evaluated",
                "reason": qual["reason"],
                "review_flags": [],
                "feedback_allowed": False,
                "contact": {"frame": None, "confidence": 0.0},
                "phase_confidence": {},
                "phases": {},
                "raw_pose_source": f"Match/{hitter}_frame_keypoints.csv",
                "smoothed_pose_source": "Match/smoothed_pose.json",
                "biomechanical_features": {},
                "errors": [],
                "skipped_rules": [{
                    "rule": "all_rules",
                    "reason": qual["reason"]
                }]
            })
            continue

        review_flags = rally_review_flags.get(sid, [])
        feedback_allowed = True
        if "rally_sequence_anomaly" in review_flags:
            print(f"  [Flag] Rally sequence anomaly detected (consecutive hit without return)")
            feedback_allowed = False

        # 2. Exact Variant Resolution
        variant_info = resolve_exact_shot_variant(coarse_type, hitter, hitter_zone)
        exact_shot = variant_info["exact_shot_type"]
        var_conf = variant_info["variant_confidence"]
        print(f"  Resolved Variant: {exact_shot} (confidence: {var_conf})")

        # Extract per-frame pose and shuttle list for this shot
        role_key = "Near Player" if hitter == "near_player" else "Far Player"
        frame_poses = []
        shuttle_frames = []

        for f in range(start_f, end_f + 1):
            f_str = str(f)
            kp_dict = frame_keypoints.get(f_str, {}).get(role_key, {})
            # Format pose dict
            formatted_pose = {}
            for k_idx, k_val in kp_dict.items():
                k_name = k_val.get("name", f"kp_{k_idx}")
                pt = k_val.get("pt", [0.0, 0.0])
                sc = k_val.get("score", 0.0)
                formatted_pose[k_name] = {"x": float(pt[0]), "y": float(pt[1]), "score": float(sc)}
            
            s_dict = shuttle_keypoints.get(f_str, {})
            frame_poses.append({"frame": f, "pose": formatted_pose, "shuttle": s_dict})

            if s_dict:
                shuttle_frames.append({"frame": f, "x": s_dict.get("x"), "y": s_dict.get("y"), "confidence": s_dict.get("confidence", 0.0)})

        # 3. Contact Frame Estimation
        contact_info = estimate_contact_frame(start_f, end_f, frame_poses, shuttle_frames, hitter)
        contact_f = contact_info["contact_frame"]
        print(f"  Contact Frame: {contact_f} (conf: {contact_info['contact_confidence']}, method: {contact_info['method']})")

        # 4. Phase Segmentation & Smoothing
        phase_res = segment_shot_phases(start_f, end_f, contact_f, frame_poses)
        smoothed = phase_res["smoothed_pose"]
        all_smoothed_poses[str(sid)] = smoothed

        # 5. Temporal Feature Extraction
        features = extract_biomechanical_features(smoothed, phase_res["phases"], contact_f, hitter)

        # 6. Biomechanical Rule Engine Evaluation
        hitter_clean = str(hitter).lower().replace(" ", "_")
        hitter_norm = "near_player" if "near" in hitter_clean else "far_player" if "far" in hitter_clean else hitter_clean
        
        if enabled_set is not None and hitter_norm not in enabled_set:
            print(f"  [Notice] Suppressing biomechanical error output for opponent hitter '{hitter}' (tracking context retained).")
            errors = []
            skipped = [{"rule": "all_rules", "reason": "Opponent player — biomechanical feedback disabled"}]
            feedback_allowed = False
        else:
            eval_res = detector.evaluate_shot(exact_shot, features, contact_info, phase_res, var_conf)
            errors = eval_res["errors"]
            skipped = eval_res["skipped_rules"]

            print(f"  Evaluation Complete: {len(errors)} errors detected, {len(skipped)} rules skipped")
            for err in errors:
                print(f"    - [{err['rule_id']}] {err['technical_error']} (Evidence: {err['evidence']['observed_value']})")

        shot_records.append({
            "shot_id": sid,
            "player": hitter,
            "shot_type": coarse_type,
            "exact_shot_type": exact_shot,
            "shot_confidence": shot_conf,
            "variant_confidence": var_conf,
            "analysis_status": "completed",
            "review_flags": review_flags,
            "feedback_allowed": feedback_allowed,
            "contact": contact_info,
            "phase_confidence": phase_res["phase_confidence"],
            "phases": phase_res["phases"],
            "raw_pose_source": f"Match/{hitter}_frame_keypoints.csv",
            "smoothed_pose_source": "Match/smoothed_pose.json",
            "biomechanical_features": features,
            "errors": errors,
            "skipped_rules": skipped
        })

    # Post-process rally chaining: build exact reference_frames (start = C_{k-1}, contact = C_k, end = C_{k+1})
    num_shots = len(shot_records)
    ref_shots_list = []

    for i, rec in enumerate(shot_records):
        c_frame = rec.get("contact", {}).get("contact_frame")
        orig_start = shots[i].get("start_frame", 0) if i < len(shots) else 0
        orig_end = shots[i].get("end_frame", 0) if i < len(shots) else 0

        if c_frame is None:
            c_frame = (orig_start + orig_end) // 2

        prev_c = shot_records[i - 1].get("contact", {}).get("contact_frame") if i > 0 else None
        next_c = shot_records[i + 1].get("contact", {}).get("contact_frame") if i < num_shots - 1 else None

        start_f = prev_c if (prev_c is not None) else orig_start
        end_f = next_c if (next_c is not None) else orig_end

        ref_dict = {
            "start_frame": int(start_f),
            "end_frame": int(end_f),
            "contact_frame_from_system": int(c_frame)
        }
        rec["reference_frames"] = ref_dict

        rule_labels = {}
        for err in rec.get("errors", []):
            if isinstance(err, dict) and "rule_id" in err:
                rule_labels[err["rule_id"]] = True

        ref_shots_list.append({
            "shot_id": rec["shot_id"],
            "shot_type_to_verify": rec.get("exact_shot_type", rec.get("shot_type", "unknown")),
            "player_to_review": rec.get("player", "far_player"),
            "reference_frames": ref_dict,
            "rule_labels": rule_labels,
            "reviewer_notes": "Derived from biomechanical rule-engine output: failed rules=True, evaluated non-failed rules=False, skipped rules=null."
        })

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    smoothed_json_path = os.path.join(output_dir, "smoothed_pose.json")
    with open(smoothed_json_path, "w", encoding="utf-8") as f:
        json.dump(all_smoothed_poses, f, indent=2)
    print(f"\n[Stage 6] Saved smoothed poses for {len(all_smoothed_poses)} shots to: {smoothed_json_path}")

    out_payload = {
        "schema_version": "1.0",
        "rule_repository_version": "1.0",
        "pipeline_version": "1.0",
        "shots": shot_records
    }

    errors_json_path = os.path.join(output_dir, output_json_name)
    with open(errors_json_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"[Stage 6] Saved biomechanical errors payload to: {errors_json_path}")

    # Export derived_reference_annotations.json
    ref_payload = {
        "video_id": os.path.basename(os.path.abspath(output_dir)),
        "annotation_status": "DERIVED_REFERENCE_FROM_RULE_ENGINE_OUTPUT",
        "label_meaning": {
            "true": "technical error is visibly present",
            "false": "technical error is not present",
            "null": "not evaluated / shot type unknown / rule not applicable"
        },
        "shots": ref_shots_list
    }
    ref_json_path = os.path.join(output_dir, "derived_reference_annotations.json")
    with open(ref_json_path, "w", encoding="utf-8") as f:
        json.dump(ref_payload, f, indent=2, ensure_ascii=False)
    print(f"[Stage 6] Saved derived reference annotations to: {ref_json_path}")

    return out_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6: Biomechanical Error Detection Pipeline")
    parser.add_argument("--shots", default="Match/shot_predictions.json")
    parser.add_argument("--kp", default="Match/frame_keypoints.json")
    parser.add_argument("--shuttle", default="Match/shuttle_keypoints.json")
    parser.add_argument("--outdir", default="Match")
    parser.add_argument("--outfile", default="biomechanical_errors.json")
    args = parser.parse_args()

    run_stage6_pipeline(args.shots, args.kp, args.shuttle, args.outdir, args.outfile)
