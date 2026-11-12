import os
import json
from llm_prompts import (
    simplify_flawed_text,
    simplify_clean_text,
    build_flawed_shot_prompt,
)

def sanitize_string(val):
    if not isinstance(val, str):
        return val
    s = val.replace("â€¢", "•").replace("â€“", "-").replace("â€”", "-")
    s = s.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    if "•" in s and ("..." in s or len(s) > 200):
        parts = [p.strip("• \t\r\n-") for p in s.replace("•", "\n•").split("\n")]
        unique = []
        for p in parts:
            if p and p not in unique and not p.endswith("..."):
                unique.append(p)
        if unique:
            s = " • ".join(unique[:3])
    return s

def sanitize_model(obj):
    if isinstance(obj, dict):
        return {k: sanitize_model(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_model(i) for i in obj]
    elif isinstance(obj, str):
        return sanitize_string(obj)
    return obj

def format_shot_name(raw_type):
    if not raw_type or str(raw_type).lower() in ("unknown", "none", ""):
        return "Stroke Execution"
    clean = str(raw_type).strip().replace("_", " ")
    name_map = {
        "forehand clear": "Forehand Clear",
        "backhand clear": "Backhand Clear",
        "clear": "Clear",
        "forehand smash": "Forehand Smash",
        "backhand smash": "Backhand Smash",
        "smash": "Smash",
        "forehand drop": "Forehand Drop",
        "forehand dropshot": "Forehand Drop",
        "backhand drop": "Backhand Drop",
        "drop": "Drop",
        "forehand net shot": "Forehand Net Shot",
        "backhand net shot": "Backhand Net Shot",
        "forehand net lift": "Forehand Lift",
        "backhand net lift": "Backhand Lift",
        "lift": "Lift",
        "serve": "Serve"
    }
    lower = clean.lower()
    if lower in name_map:
        return name_map[lower]
    return " ".join(word.capitalize() for word in clean.split())

def format_evidence_for_coach(evidence):
    if not evidence:
        return "Biomechanical rule threshold limit exceeded."

    if isinstance(evidence, dict):
        feat = str(evidence.get("feature", ""))
        obs = evidence.get("observed_value")
        thresh = evidence.get("acceptable_threshold")
        unit = str(evidence.get("unit", ""))

        if "wrist_height_relative_to_head" in feat:
            try:
                val = float(obs)
                pct = abs(int(round(val * 100)))
                is_below = val < 0
                loc = "BELOW head level" if is_below else "at head level"
                return f"Impact Height: Racket contact was {pct}% of torso length {loc} (Target: High overhead contact above head)."
            except Exception:
                return f"Impact Height: Racket contact was low relative to head height (Observed: {obs} {unit})."

        if "elbow_extension_delta" in feat:
            return f"Arm Extension: Elbow extended only {obs} deg at contact (Target: Full arm extension >= {thresh} deg)."

        if "vertical_wrist_displacement" in feat:
            return f"Upward Swing Arc: Upward wrist displacement was {obs} torso length (Target: High upward lift arc >= {thresh})."

        if "normalised_wrist_path_length" in feat:
            return f"Swing Stroke Control: Wrist path length was {obs} torso units (Target: Controlled net stroke <= {thresh})."

        if "hitting_wrist_in_front_of_torso" in feat:
            return f"Contact Position: Racket contact was caught behind body plane (Observed: {obs} {unit})."

        if "knee_flexion_angle" in feat:
            return f"Lunge Knee Flexion: Knee flexion angle was {obs}° (Target: Deeper knee bend)."

        clean_feat = feat.replace("_relative_to_", " relative to ").replace("_", " ").title()
        return f"Measured {clean_feat}: {obs} {unit} (Target: {thresh} {unit})"

    if isinstance(evidence, str):
        s = evidence
        if "wrist_height_relative_to_head" in s:
            import re
            m = re.findall(r"[-+]?\d*\.?\d+", s)
            if m:
                val = float(m[0])
                pct = abs(int(round(val * 100)))
                is_below = val < 0
                loc = "BELOW head level" if is_below else "at head level"
                return f"Impact Height: Racket contact was {pct}% of torso length {loc} (Target: High overhead contact above head)."
            return "Impact Height: Racket contact was low relative to head height."

        if "elbow_extension_delta" in s:
            import re
            m = re.findall(r"[-+]?\d*\.?\d+", s)
            obs = m[0] if m else "0"
            thresh = m[1] if len(m) > 1 else "25.0"
            return f"Arm Extension: Elbow extended only {obs} deg at contact (Target: Full arm extension >= {thresh} deg)."

        if "vertical_wrist_displacement" in s:
            import re
            m = re.findall(r"[-+]?\d*\.?\d+", s)
            obs = m[0] if m else "0"
            thresh = m[1] if len(m) > 1 else "0.25"
            return f"Upward Swing Arc: Upward wrist displacement was {obs} torso length (Target: High upward lift arc >= {thresh})."

        if "normalised_wrist_path_length" in s:
            import re
            m = re.findall(r"[-+]?\d*\.?\d+", s)
            obs = m[0] if m else "0"
            thresh = m[1] if len(m) > 1 else "0.45"
            return f"Swing Stroke Control: Wrist path length was {obs} torso units (Target: Controlled net stroke <= {thresh})."

        return s.replace("Observed ", "Measured ").replace("torso_length", "body length").replace("_relative_to_", " relative to ").replace("_", " ")

    return str(evidence)

def build_report_view_model(
    session=None,
    participants=None,
    shot_predictions=None,
    biomechanical_errors=None,
    coaching_feedback=None,
    frame_keypoints=None,
    shuttle_keypoints=None,
    output_dir=None
):
    """
    Builds a normalized, single-source-of-truth report view object exposing:
    - Player Details (player_name, player_id, coach_name, session_date, session_id, court_side)
    - 5 Executive Summary Metrics (total_shots, correct_shots, shots_with_errors, technique_accuracy, most_frequent_shot)
    - Simplified Shot Feedback (what_happened, how_to_improve, practice_drill, feedback, keep_practising)
    - Persists output model to Match/report_view.json.
    """
    session_info = session if isinstance(session, dict) else {}
    participants_list = participants if isinstance(participants, list) else []
    
    # 1. Audit Data Quality & Limitations
    # Only mark pose as unavailable if we explicitly received an empty/None keypoints dict
    # (not when the list-view shortcut skips loading keypoints for performance)
    pose_avail = True  # optimistic default — report view is best-effort
    if frame_keypoints is not None:
        # We were explicitly given keypoints — check if they are actually populated
        if isinstance(frame_keypoints, list) and len(frame_keypoints) == 0:
            pose_avail = False
        elif isinstance(frame_keypoints, dict) and len(frame_keypoints) == 0:
            pose_avail = False

    shuttle_avail = True  # optimistic default
    if shuttle_keypoints is not None:
        if isinstance(shuttle_keypoints, list) and len(shuttle_keypoints) == 0:
            shuttle_avail = False
        elif isinstance(shuttle_keypoints, dict) and len(shuttle_keypoints) == 0:
            shuttle_avail = False

    limitations = []
    if not pose_avail:
        limitations.append("Full body pose keypoints not available for trajectory alignment.")
    if not shuttle_avail:
        limitations.append("Shuttle trajectory telemetry was not detected in this video segment.")

    data_quality = {
        "pose_available": pose_avail,
        "shuttle_available": shuttle_avail,
        "limitations": limitations,
        "source": "pipeline"
    }

    # 2. Filter Shot Predictions (exclude 'unknown')
    raw_shots = shot_predictions if isinstance(shot_predictions, list) else []
    valid_shots = [
        s for s in raw_shots 
        if isinstance(s, dict) and str(s.get("shot_type", "")).lower() != "unknown"
    ]

    # 3. Filter Biomechanical Errors (flatten & exclude 'unknown')
    flat_errors = []
    if isinstance(biomechanical_errors, dict):
        shots_err = biomechanical_errors.get("shots", [])
        if isinstance(shots_err, list):
            for s in shots_err:
                if not isinstance(s, dict):
                    continue
                s_id = s.get("shot_id")
                s_player = s.get("player")
                s_type = s.get("shot_type") or s.get("exact_shot_type")
                if str(s_type).lower() == "unknown":
                    continue
                for e_item in s.get("errors", []):
                    if isinstance(e_item, dict):
                        ec = e_item.copy()
                        ec["shot_id"] = s_id
                        ec["role"] = s_player
                        ec["shot_type"] = s_type
                        flat_errors.append(ec)
        else:
            for k, v in biomechanical_errors.items():
                if isinstance(v, list):
                    flat_errors.extend([item for item in v if isinstance(item, dict)])
    elif isinstance(biomechanical_errors, list):
        flat_errors = [e for e in biomechanical_errors if isinstance(e, dict)]

    valid_errors = [
        e for e in flat_errors
        if str(e.get("shot_type", "")).lower() != "unknown"
    ]

    # 4. Filter Coaching Feedback (exclude 'unknown')
    raw_fb = []
    if isinstance(coaching_feedback, dict):
        raw_fb = coaching_feedback.get("coaching_feedback", [])
    elif isinstance(coaching_feedback, list):
        raw_fb = coaching_feedback

    valid_feedback = []
    for fb in raw_fb:
        if isinstance(fb, dict):
            st = fb.get("exact_shot_type") or fb.get("shot_type") or ""
            if str(st).lower() != "unknown":
                valid_feedback.append(fb)

    # 5. Format Participants and Build Player Reports
    if not participants_list:
        sess_players_str = str(session_info.get("players", "")).lower()
        far_has_shots = any("far" in str(s.get("hitter") or s.get("player") or "").lower() for s in valid_shots)
        is_far_primary = ("far" in sess_players_str) or (far_has_shots and "near" not in sess_players_str)

        primary_name = session_info.get("players") or "Target Player"

        participants_list = [
            {
                "player_id": 1,
                "full_name": primary_name,
                "court_side": "far_player" if is_far_primary else "near_player",
                "role": "target_player",
                "report_enabled": True
            },
            {
                "player_id": 2,
                "full_name": "Second Player",
                "court_side": "near_player" if is_far_primary else "far_player",
                "role": "opponent",
                "report_enabled": True if far_has_shots else False
            }
        ]

    sess_date_val = session_info.get("created_at") or session_info.get("timestamp") or "Today"

    players_report = []
    for p in participants_list:
        c_side = str(p.get("court_side") or "").lower()
        side_key = "near" if "near" in c_side else ("far" if "far" in c_side else c_side)

        has_valid_shots = any(side_key in str(s.get("hitter") or s.get("role") or s.get("player") or "").lower() for s in valid_shots)

        is_enabled = p.get("report_enabled") in (True, 1, "true", "True") or has_valid_shots
        if not is_enabled:
            continue

        p_shots = []
        for s in valid_shots:
            hitter = str(s.get("hitter") or s.get("role") or s.get("player") or "").lower()
            if not hitter or side_key in hitter:
                p_shots.append(s)

        shot_dist = {}
        conf_sum = 0.0
        conf_count = 0
        timeline = []

        for idx, s in enumerate(p_shots):
            st = s.get("shot_type") or "stroke"
            s_formatted = format_shot_name(st)
            shot_dist[s_formatted] = shot_dist.get(s_formatted, 0) + 1
            
            conf = s.get("confidence") or s.get("shot_confidence")
            if isinstance(conf, (int, float)):
                conf_sum += float(conf)
                conf_count += 1

            frame_start = s.get("start_frame") or s.get("frame_start") or (s.get("phases", {}).get("prepare", {}).get("start")) or 0
            frame_end = s.get("end_frame") or s.get("frame_end") or (s.get("phases", {}).get("follow_through", {}).get("end")) or 0
            contact_frame = s.get("contact_frame") or (s.get("contact", {}).get("contact_frame")) or frame_start

            sec = float(contact_frame) / 30.0 if isinstance(contact_frame, (int, float)) else 0.0
            mins = int(sec // 60)
            secs = sec % 60
            timestamp_str = f"{mins:02d}:{secs:04.1f}"
            s_id = s.get("shot_id") or (idx + 1)

            timeline.append({
                "shot_id": s_id,
                "shot_type": st,
                "formatted_shot_name": s_formatted,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "contact_frame": contact_frame,
                "timestamp": timestamp_str,
                "time_seconds": round(sec, 2),
                "court_zone": s.get("court_zone") or s.get("zone") or "Not available from the current analysis.",
                "confidence": conf if conf is not None else "Not available from the current analysis.",
                "source": "pipeline"
            })

        avg_conf = round(conf_sum / conf_count, 3) if conf_count > 0 else "Not available from the current analysis."

        p_errors = []
        for e in valid_errors:
            r = str(e.get("role") or e.get("player") or e.get("hitter") or "").lower()
            if not r or side_key in r:
                raw_ev = e.get("visual_evidence") or e.get("evidence") or e.get("description")
                e_shot_id = e.get("shot_id")
                matching_t = next((t for t in timeline if str(t["shot_id"]) == str(e_shot_id)), {})
                e_contact_frame = matching_t.get("contact_frame") or e.get("contact_frame") or 0
                e_timestamp = matching_t.get("timestamp") or "N/A"
                e_time_seconds = matching_t.get("time_seconds") or 0.0

                p_errors.append({
                    "shot_id": e_shot_id,
                    "shot_type": e.get("shot_type"),
                    "formatted_shot_name": format_shot_name(e.get("shot_type")),
                    "contact_frame": e_contact_frame,
                    "timestamp": e_timestamp,
                    "time_seconds": e_time_seconds,
                    "phase": e.get("phase") or "Forward Swing",
                    "technical_error": e.get("technical_error") or e.get("error_type") or e.get("error") or "Biomechanical Flaw",
                    "main_mistake": e.get("main_mistake") or f"Execution deviation detected during the {e.get('phase', 'forward swing').lower()} phase.",
                    "why_it_matters": e.get("why_it_matters") or e.get("why") or "Limits stroke depth and gives opponent an attacking opportunity.",
                    "how_to_correct": e.get("how_to_correct") or e.get("correction") or "Adjust racket contact point and complete the upward swing.",
                    "coaching_tip": e.get("coaching_tip") or e.get("cue") or "Lift high, then finish the swing.",
                    "measured_evidence": format_evidence_for_coach(raw_ev),
                    "why": e.get("why"),
                    "correction": e.get("correction"),
                    "cue": e.get("cue"),
                    "source": "rule_engine"
                })

        fb_map = {}
        for fb in valid_feedback:
            r = str(fb.get("role") or fb.get("hitter") or fb.get("player") or "").lower()
            if not r or side_key in r:
                s_id = fb.get("shot_id")
                if s_id is not None:
                    fb_map[s_id] = fb

        # Calculate Executive Summary Metrics for this Player
        unique_error_shot_ids = set(str(e["shot_id"]) for e in p_errors)
        total_shots = len(p_shots)
        shots_with_errors = len(unique_error_shot_ids)
        correct_shots = max(0, total_shots - shots_with_errors)
        tech_acc = round((correct_shots / total_shots) * 100) if total_shots > 0 else 100
        most_freq_shot = list(shot_dist.keys())[0] if shot_dist else "Stroke Execution"

        # ── Diagnostic Cache: load or generate ──────────────────────────────
        # Cache key = "<player_side>_<shot_id>"  →  stored per-session in
        # coaching_diagnostics_cache.json so the LLM/random generator only
        # runs ONCE per shot, not on every page refresh.
        diag_cache = {}
        diag_cache_path = os.path.join(output_dir, "coaching_diagnostics_cache.json") if output_dir else None
        if diag_cache_path and os.path.exists(diag_cache_path):
            try:
                with open(diag_cache_path, "r", encoding="utf-8") as _cf:
                    diag_cache = json.load(_cf)
            except Exception:
                diag_cache = {}

        diag_cache_dirty = False  # track if we need to write back

        p_feedback = []
        for idx, s in enumerate(p_shots):
            s_id = s.get("shot_id") or (idx + 1)
            s_type = s.get("shot_type") or s.get("exact_shot_type") or "stroke"
            s_formatted_name = format_shot_name(s_type)
            s_errs = [e for e in p_errors if str(e.get("shot_id")) == str(s_id)]
            fb_item = fb_map.get(s_id, {})
            cfg = fb_item.get("coaching_feedback") if isinstance(fb_item.get("coaching_feedback"), dict) else fb_item

            cache_key = f"{side_key}_{s_id}"

            if cache_key in diag_cache:
                # ── HIT: use cached diagnostics (no LLM call, no randomness)
                cached = diag_cache[cache_key]
                wh = cached["what_happened"]
                wtd = cached["how_to_improve"]
                drill = cached["practice_drill"]
                fb_text = wh
                keep_practising = wtd
            else:
                # ── MISS: generate diagnostics and store in cache
                if s_errs:
                    first_err = s_errs[0]
                    flaw_name = first_err.get("technical_error") or "execution flaw"
                    main_mistake = first_err.get("main_mistake")
                    why_wrong = first_err.get("why_it_matters") or first_err.get("why")
                    corr = first_err.get("how_to_correct") or first_err.get("correction")
                    cue = first_err.get("coaching_tip") or first_err.get("cue")
                    measured_ev = first_err.get("measured_evidence") or first_err.get("visual_evidence") or ""

                    wh, wtd, drill_from_prompt = simplify_flawed_text(s_formatted_name, flaw_name, main_mistake, why_wrong, corr, cue, measured_ev)
                    fb_text = wh
                    keep_practising = wtd
                    drill = cfg.get("recommended_drill") or cfg.get("drill") or drill_from_prompt
                else:
                    fb_text, keep_practising, drill_from_prompt = simplify_clean_text(s_formatted_name)
                    wh = fb_text
                    wtd = keep_practising
                    drill = cfg.get("recommended_drill") or cfg.get("drill") or drill_from_prompt

                # Store in cache
                diag_cache[cache_key] = {
                    "what_happened": wh,
                    "how_to_improve": wtd,
                    "practice_drill": drill,
                    "shot_type": s_type,
                    "formatted_shot_name": s_formatted_name,
                    "has_errors": len(s_errs) > 0
                }
                diag_cache_dirty = True

            p_feedback.append({
                "shot_id": s_id,
                "exact_shot_type": s_type,
                "formatted_shot_name": s_formatted_name,
                "what_happened": wh,
                "what_to_do": wtd,
                "how_to_improve": wtd,
                "feedback": fb_text,
                "keep_practising": keep_practising,
                "practice_drill": drill,
                "recommended_drill": drill,
                "source": "llm_prompts_cached" if cache_key in diag_cache and not diag_cache_dirty else "llm_prompts_grounded"
            })

        # ── Persist cache to disk if new entries were generated ────────────
        if diag_cache_dirty and diag_cache_path:
            try:
                with open(diag_cache_path, "w", encoding="utf-8") as _cf:
                    json.dump(diag_cache, _cf, indent=2, ensure_ascii=False)
            except Exception as _e:
                print(f"[Warning] Failed to persist coaching_diagnostics_cache.json: {_e}")

        # Build Player Details payload
        player_details = {
            "player_name": p.get("full_name") or f"Player ({side_key.title()} Side)",
            "player_id": p.get("player_id") or 1,
            "coach_name": session_info.get("coach_name") or "Coach",
            "session_date": str(sess_date_val),
            "session_id": session_info.get("session_id") or session_info.get("id") or "Latest",
            "court_side": side_key.title() + " Side"
        }

        # Build Executive Summary Metrics payload
        summary_metrics = {
            "total_shots": total_shots,
            "correct_shots": correct_shots,
            "shots_with_errors": shots_with_errors,
            "technique_accuracy": tech_acc,
            "most_frequent_shot": most_freq_shot
        }

        shot_stats = {
            "total_shots": total_shots,
            "shot_distribution": shot_dist,
            "average_confidence": avg_conf,
            "source": "pipeline"
        }

        review_flags = []
        if len(p_shots) == 0:
            review_flags.append("No valid shot detections recorded for this player segment.")
        if len(p_errors) > 0:
            review_flags.append(f"{len(p_errors)} biomechanical flaws flagged during execution.")

        p_name = player_details["player_name"]
        p_summary = (
            f"Coaching evaluation for {p_name}: Executed {total_shots} analyzed shots with primary stroke emphasis on {most_freq_shot}. "
            f"{shots_with_errors} unique shot(s) flagged for biomechanical correction ({tech_acc}% technique accuracy)."
        )

        players_report.append({
            "player_id": p.get("player_id"),
            "full_name": p_name,
            "court_side": p.get("court_side") or (side_key + "_player"),
            "role": p.get("role") or "target_player",
            "report_enabled": True,
            "player_details": player_details,
            "summary_metrics": summary_metrics,
            "summary": p_summary,
            "summary_source": "llm_summary",
            "shot_statistics": shot_stats,
            "shot_timeline": timeline,
            "detected_errors": p_errors,
            "coaching_feedback": p_feedback,
            "review_flags": review_flags
        })

    target_names = [pr["full_name"] for pr in players_report]
    targets_str = " and ".join(target_names) if target_names else "Target Player"
    total_match_shots = sum(pr["summary_metrics"]["total_shots"] for pr in players_report)
    total_match_errs = sum(pr["summary_metrics"]["shots_with_errors"] for pr in players_report)
    overall_summary = f"Match analysis for {targets_str}. Total shots analyzed: {total_match_shots}. Flaws detected: {total_match_errs}."

    raw_report_view_model = {
        "session": {
            "session_id": session_info.get("session_id") or session_info.get("id"),
            "status": session_info.get("status") or "Completed",
            "created_at": str(sess_date_val),
            "players": session_info.get("players"),
            "source": "pipeline"
        },
        "participants": participants_list,
        "players": players_report,
        "overall_summary": overall_summary,
        "overall_summary_source": "llm_summary",
        "data_quality": data_quality
    }

    report_view_model = sanitize_model(raw_report_view_model)

    try:
        _base_match_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Match")
        os.makedirs(_base_match_dir, exist_ok=True)
        match_json_path = os.path.join(_base_match_dir, "report_view.json")
        with open(match_json_path, "w", encoding="utf-8") as f:
            json.dump(report_view_model, f, indent=2, ensure_ascii=False)

        if output_dir and os.path.exists(output_dir):
            out_json_path = os.path.join(output_dir, "report_view.json")
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(report_view_model, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Warning] Failed to persist report_view.json: {e}")

    return report_view_model

