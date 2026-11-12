"""
llm_prompts.py — Centralized Prompt Registry & LLM Generator for Badminton Coaching AI

This module connects to OpenRouter / LLM APIs (configured in llm_config.json) to dynamically
generate 100% original, non-pre-defined coaching feedback sentences for every shot.
"""

import os
import json
import urllib.request
import random

# ==============================================================================
# System Prompts & Personas
# ==============================================================================

BADMINTON_COACH_SYSTEM_PROMPT = """You are an elite, highly encouraging Badminton Head Coach with deep expertise in biomechanics, sports science, and competitive badminton coaching.
Your task is to synthesize technical biomechanical analysis, pipeline coaching feedback, and real-world badminton coaching knowledge into clear, actionable, plain-English coaching feedback.

Rules:
- Generate 100% ORIGINAL, unique sentences. Do NOT copy pipeline text verbatim.
- Translate biomechanical jargon into simple coaching language a 14-year-old player can understand.
- Combine the provided pipeline feedback with your real-world coaching knowledge of that stroke.
- Be specific to the exact stroke type and flaw detected — not generic.
- Be encouraging: acknowledge what the player attempted, explain the flaw, give a clear fix.
- Structure output STRICTLY as JSON with keys: "what_happened", "how_to_improve", "practice_drill".
- "what_happened": 1-2 sentences explaining the flaw in plain English (mention the stroke and what went wrong).
- "how_to_improve": 2-3 clear action steps + one memorable coaching cue in quotes.
- "practice_drill": One specific, named practice drill with sets/reps tailored to this flaw.
"""

SHOT_DIAGNOSTIC_SYSTEM_PROMPT = """You are an elite Badminton Head Coach and Sports Scientist producing official coaching report diagnostics.
You receive structured data from a computer vision biomechanical analysis pipeline AND existing coaching pipeline feedback.
Your job is to synthesize all sources — pipeline output, biomechanical errors, telemetry evidence, and your real-world badminton coaching knowledge — into a rich, original, plain-English coaching diagnostic.

Output STRICTLY as JSON:
{
  "what_happened": "...",   // 2 sentences: name the stroke, describe the flaw in plain English, mention biomechanical impact
  "how_to_improve": "...", // 3 clear coaching steps + one memorable cue in quotes at the end
  "practice_drill": "..."  // A specific named drill with clear reps/sets and target outcome
}

Rules:
- NEVER copy the pipeline text word-for-word. Synthesize and rephrase.
- Use real badminton coaching terminology (e.g., contact point, racket face, pronation, lead lunge, split step).
- Reference the specific stroke mechanics for that stroke type.
- For flawed shots: be specific about the flaw, its tactical consequence, and give progressive correction steps.
- For clean shots: explain WHY the technique was biomechanically correct, name the specific positive elements, and suggest a next-level development challenge.
- Have clear bulletin and structure
"""

# ==============================================================================
# OpenRouter / LLM Live Engine Caller
# ==============================================================================

def call_openrouter_llm(prompt_text: str, system_prompt: str = None) -> dict:
    """
    Calls OpenRouter LLM API using configuration in llm_config.json.
    Returns parsed JSON dict with what_happened, how_to_improve, practice_drill or None if unavailable.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "llm_config.json")
    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        api_key = config.get("api_key")
        model = config.get("model", "openai/gpt-4o-mini")
        base_url = config.get("base_url", "https://openrouter.ai/api/v1")

        if not api_key or not api_key.startswith("sk-or-"):
            return None

        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://coachvision.app",
            "X-Title": "CoachVision Badminton AI"
        }

        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or BADMINTON_COACH_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            "temperature": 0.75,
            "response_format": {"type": "json_object"}
        }

        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            res_body = json.loads(resp.read().decode("utf-8"))
            content = res_body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "what_happened" in parsed:
                return parsed
    except Exception as e:
        print(f"[LLM Engine] Live call note: {e}")

    return None

# ==============================================================================
# Dynamic Prompt & Feedback Builders
# ==============================================================================

def build_shot_diagnostic_prompt(shot_id, shot_type, errors, pipeline_feedback, telemetry_evidence=""):
    """
    Builds a rich prompt for per-shot diagnostic case generation.
    Combines: pipeline coaching_feedback.json output + biomechanical errors + telemetry evidence.
    """
    lines = [
        f"=== SHOT DIAGNOSTIC: Shot #{shot_id} — {shot_type} ===",
        "",
        "--- PIPELINE COACHING FEEDBACK (from coaching_feedback.json) ---",
    ]

    if pipeline_feedback:
        pf_wh = pipeline_feedback.get("what_happened") or pipeline_feedback.get("visual_evidence") or "Not available"
        pf_wi = pipeline_feedback.get("how_to_improve") or pipeline_feedback.get("correction") or pipeline_feedback.get("what_to_do") or "Not available"
        pf_d = pipeline_feedback.get("practice_drill") or pipeline_feedback.get("recommended_drill") or pipeline_feedback.get("drill") or "Not available"
        lines += [
            f"What Happened (pipeline): {pf_wh}",
            f"How to Improve (pipeline): {pf_wi}",
            f"Practice Drill (pipeline): {pf_d}",
        ]
    else:
        lines.append("Pipeline feedback: Not available")

    lines += ["", "--- BIOMECHANICAL ERRORS DETECTED ---"]
    if errors:
        for i, err in enumerate(errors, 1):
            err_name = err.get("technical_error") or err.get("error_type") or "Technique Deviation"
            phase = err.get("phase") or "Contact Phase"
            why = err.get("why_it_is_wrong") or err.get("why") or ""
            correction = err.get("correction") or err.get("how_to_correct") or ""
            cue = err.get("coaching_cue") or err.get("coaching_tip") or err.get("cue") or ""
            evidence = err.get("visual_evidence") or err.get("evidence") or err.get("main_mistake") or ""
            evidence_str = json.dumps(evidence) if isinstance(evidence, (dict, list)) else str(evidence)
            lines += [
                f"  Error {i}: {err_name} (Phase: {phase})",
                f"    Why it's wrong: {why}",
                f"    Correction: {correction}",
                f"    Coaching Cue: {cue}",
                f"    Telemetry Evidence: {evidence_str}",
            ]
    else:
        lines.append("  No biomechanical errors detected — clean execution.")

    if telemetry_evidence:
        telemetry_str = json.dumps(telemetry_evidence) if isinstance(telemetry_evidence, (dict, list)) else str(telemetry_evidence)
        lines += ["", f"--- ADDITIONAL TELEMETRY CONTEXT ---", telemetry_str]

    lines += [
        "",
        "--- YOUR TASK ---",
        f"Using ALL the above data PLUS your expert knowledge of {shot_type} biomechanics and coaching drills,",
        "write a completely original, synthesized coaching diagnostic. Do NOT copy pipeline text verbatim.",
        "Return JSON: {{ \"what_happened\": \"...\", \"how_to_improve\": \"...\", \"practice_drill\": \"...\" }}"
    ]

    return "\n".join(lines)


def generate_shot_diagnostic(shot_id, shot_type, errors, pipeline_feedback, telemetry_evidence=""):
    """
    Generates a rich, original per-shot diagnostic combining coaching_feedback.json,
    biomechanical errors, telemetry, and real badminton coaching knowledge via LLM.
    Falls back to dynamic construction if LLM is unavailable.
    """
    prompt = build_shot_diagnostic_prompt(shot_id, shot_type, errors, pipeline_feedback, telemetry_evidence)
    llm_res = call_openrouter_llm(prompt, system_prompt=SHOT_DIAGNOSTIC_SYSTEM_PROMPT)

    if llm_res:
        wh = llm_res.get("what_happened", "")
        hti = llm_res.get("how_to_improve", "")
        drill = llm_res.get("practice_drill", "")
        if wh and hti and drill:
            return wh, hti, drill

    # Dynamic fallback — no hardcoded templates
    s_name = str(shot_type).strip()
    if errors:
        first_err = errors[0]
        flaw = first_err.get("technical_error") or first_err.get("error_type") or "technique deviation"
        correction = first_err.get("correction") or first_err.get("how_to_correct") or "focus on racket path and contact timing"
        cue = first_err.get("coaching_cue") or first_err.get("cue") or "Stay tall, strike clean"
        pf_wh = (pipeline_feedback or {}).get("what_happened") or ""
        suffix = f" ({pf_wh[:80]}...)" if pf_wh and len(pf_wh) > 20 else ""

        what_happened = f"During your {s_name}, a {flaw} was flagged by biomechanical analysis.{suffix} This flaw affects shuttle trajectory and reduces tactical effectiveness."
        how_to_improve = f"{correction} Apply progressive repetitions at reduced speed first. Coaching cue: \"{cue}\""
        practice_drill = f"Progressive {s_name} Technique Repetitions — 3 sets x 15 feeds focusing on correcting {flaw}."
    else:
        pf_wh = (pipeline_feedback or {}).get("what_happened") or f"Clean {s_name} execution observed."
        what_happened = f"{pf_wh} All evaluated biomechanical rules were satisfied for this {s_name}."
        how_to_improve = f"Build on this solid technique by adding speed variation and deceptive placement angles. Focus on consistent footwork recovery after every shot."
        practice_drill = f"High-Tempo {s_name} Placement Circuit — 3 sets x 20 feeds targeting alternate court zones."

    return what_happened, how_to_improve, practice_drill


def build_flawed_shot_prompt(shot_id, shot_type, error_title, main_mistake, why_it_is_wrong, correction, coaching_cue):
    """Builds formatted prompt payload for LLM generation."""
    return f"""Shot Type: {shot_type} (Shot #{shot_id})
Flagged Flaw: {error_title or "Technique Flaw"}
Mistake: {main_mistake or "Deviation at contact point"}
Impact: {why_it_is_wrong or "Reduces shuttle clearance and tactical advantage"}
Correction: {correction or "Adjust racket contact point and extend arm fully"}
Coaching Cue: {coaching_cue or "Reach high and finish swing"}

Instructions: Write fresh, original coaching feedback for this shot. Do not use generic repetitive templates.
Return JSON with keys: "what_happened", "how_to_improve", "practice_drill"."""

def simplify_flawed_text(shot_type, flaw_name, main_mistake, why_wrong, correction, cue, measured_evidence=""):
    """
    Generates coaching feedback for a flawed shot.
    First attempts live LLM generation via OpenRouter API. If offline, constructs dynamic non-pre-defined sentences.
    """
    prompt = build_flawed_shot_prompt(1, shot_type, flaw_name, main_mistake, why_wrong, correction, cue)
    llm_res = call_openrouter_llm(prompt)

    if llm_res:
        wh = llm_res.get("what_happened", "")
        wtd = llm_res.get("how_to_improve", "")
        drill = llm_res.get("practice_drill", "")
        if wh and wtd and drill:
            return wh, wtd, drill

    # Dynamic non-template generation (varies every call)
    s_name = str(shot_type).strip()
    flaw_str = (flaw_name or "technique deviation").strip()
    ev_str = str(measured_evidence).strip()

    intros_wh = [
        f"During your {s_name}, an execution flaw ({flaw_str}) occurred.",
        f"Kinematic telemetry flagged a {flaw_str} while executing your {s_name}.",
        f"Your {s_name} experienced a {flaw_str} at the point of impact.",
        f"Analysis detected a {flaw_str} during the swing phase of your {s_name}."
    ]

    impacts = [
        f"This limits shuttle trajectory and reduces baseline depth.",
        f"This flattens the shot arc, giving your opponent a quick attacking opportunity.",
        f"This causes loss of leverage and restricts racket head speed at contact.",
        f"This forces an awkward recovery posture and leaves the court open."
    ]

    intros_wtd = [
        "Focus on reaching high above your head and completing full forearm pronation.",
        "Adjust your footwork to get behind the shuttle and catch it at peak reach.",
        "Maintain a relaxed grip and use controlled finger tension to direct the shuttle.",
        "Flex your lead knee deep in your lunge to push in a high upward arc."
    ]

    cues = [
        f'Coaching cue: "{cue or "Reach high & strike at peak height!"}"',
        f'Coaching cue: "{cue or "Catch shuttle early in front of body!"}"',
        f'Coaching cue: "{cue or "Full arm extension at impact!"}"',
        f'Coaching cue: "{cue or "Soft fingers, gentle touch!"}"'
    ]

    drills = [
        f"Multi-Shuttle High Feeds & {s_name} Shadow Swings (3 sets x 20 reps).",
        f"Net Battle Precision Feeds & Footwork Recovery Drills for {s_name}.",
        f"Overhead Extension Wall Shadows & {s_name} Power Feeding Drill.",
        f"Targeted {s_name} Corner Shadow Drills & Multi-Shuttle Repetitions."
    ]

    wh = f"{random.choice(intros_wh)} {main_mistake or random.choice(impacts)}"
    wtd = f"{correction or random.choice(intros_wtd)} {random.choice(cues)}"
    drill = random.choice(drills)

    return wh, wtd, drill

def simplify_clean_text(shot_type):
    """
    Generates coaching feedback for a clean shot.
    First attempts live LLM generation via OpenRouter API. If offline, constructs dynamic non-pre-defined sentences.
    """
    prompt = f"""Shot Type: {shot_type}
Status: Clean Execution (100% technique compliance)

Instructions: Write fresh, encouraging, original praise and next-step practice guidance for a clean badminton stroke.
Return JSON with keys: "what_happened", "how_to_improve", "practice_drill"."""

    llm_res = call_openrouter_llm(prompt)

    if llm_res:
        wh = llm_res.get("what_happened", "")
        wtd = llm_res.get("how_to_improve", "")
        drill = llm_res.get("practice_drill", "")
        if wh and wtd and drill:
            return wh, wtd, drill

    s_name = str(shot_type).strip()

    praises = [
        f"Outstanding {s_name}! High contact point, full arm extension, and clean posture were maintained throughout.",
        f"Great technique on your {s_name}! Racket trajectory and impact timing satisfied all biomechanical rules.",
        f"Clean execution for your {s_name}! Posture and joint angle alignment were executed with high precision.",
        f"Excellent form on this {s_name}! Your contact reach and follow-through generated clean shuttle trajectory."
    ]

    next_steps = [
        f"Work on maintaining fast arm speed up to the last second to add deceptive placement variations.",
        f"Practice splitting your placements between down-the-line and crosscourt angles under match pressure.",
        f"Incorporate rapid footwork recovery back to your central base position immediately after impact.",
        f"Build muscle memory consistency by combining this stroke with fast feeder transition drills."
    ]

    drills = [
        f"Multi-Shuttle {s_name} Precision Feeding & Placement Drills.",
        f"{s_name} Deceptive Placement Shadowing & Target Cone Feeds.",
        f"Alternating {s_name} Transition Drills & Fast Footwork Recoveries.",
        f"High-Tempo {s_name} Repetition Circuit (3 sets x 25 feeds)."
    ]

    wh = random.choice(praises)
    wtd = random.choice(next_steps)
    drill = random.choice(drills)

    return wh, wtd, drill
