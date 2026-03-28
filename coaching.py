"""
PitchGuard AI — Gemini AI Coaching Module
Generates personalized HTML coaching reports using Google Gemini 2.5 Flash.
"""

from typing import Dict, List, Optional
from biomechanics import OPTIMAL_RANGES


# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_prompt(
    features: Dict[str, float],
    injury: Dict,
    outcome_result: Optional[Dict],
) -> str:
    """Construct a rich structured prompt for Gemini."""

    delivery_label = outcome_result.get("label", "Unknown").replace("_", " ") if outcome_result else "Unknown"
    confidence     = outcome_result.get("confidence", 0.0) if outcome_result else 0.0
    risk_level     = injury.get("overall_risk", "Low")
    risk_index     = injury.get("risk_index", 0)
    warnings       = injury.get("warnings", [])
    feat_scores    = injury.get("feature_scores", {})

    # Build feature comparison table text
    feature_lines: List[str] = []
    for feat, val in features.items():
        opt = OPTIMAL_RANGES.get(feat, {})
        lo  = opt.get("min", "?")
        hi  = opt.get("max", "?")
        sc  = feat_scores.get(feat, 0.0)
        status = "✅ OK" if sc < 15 else ("⚠️ WARN" if sc < 50 else "🔴 HIGH RISK")
        feature_lines.append(
            f"  • {feat.replace('_', ' ').title():35s} | Val: {val:7.2f} | Range: {lo}–{hi} | {status} (score {sc:.0f})"
        )
    feature_table = "\n".join(feature_lines)

    warnings_text = "\n".join(f"  - {w}" for w in warnings) if warnings else "  None"

    prompt = f"""You are an elite baseball biomechanics coach with 20+ years experience working with MLB pitchers. 
You have deep knowledge of ASMI (American Sports Medicine Institute) research, Driveline Baseball methodology, 
and modern pitch design principles.

A pitcher has just been analyzed by the PitchGuard AI system. Here are the full results:

═══════════════════════════════════════════════
DELIVERY CLASSIFICATION
═══════════════════════════════════════════════
Classification : {delivery_label}
Model Confidence: {confidence*100:.1f}%

═══════════════════════════════════════════════
INJURY RISK ASSESSMENT
═══════════════════════════════════════════════
Overall Risk Level : {risk_level}
Risk Index         : {risk_index} / 100

Active Biomechanical Warnings:
{warnings_text}

═══════════════════════════════════════════════
FULL BIOMECHANICAL FEATURE BREAKDOWN
═══════════════════════════════════════════════
{feature_table}

═══════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════
Generate a detailed, personalized coaching report in HTML format. 
The report must:
1. Be written in the voice of an expert MLB pitching coach speaking directly to the pitcher
2. Be specific — reference actual feature values and ranges, not generic advice
3. Be actionable — provide concrete drills, cues, and progressions
4. Be evidence-based — cite ASMI, Driveline, or biomechanics research where relevant
5. Prioritize the highest-risk issues first

OUTPUT FORMAT (strict HTML — no markdown, no code blocks):
Return ONLY the inner HTML content (no <html>, <head>, or <body> tags).
Use only these HTML elements: <h3>, <p>, <ul>, <li>, <strong>, <em>, <span>.

Structure your response with these exact 5 section headers as <h3> tags:
1. Executive Summary
2. Priority Corrections
3. Drill Recommendations
4. Injury Prevention Protocol
5. Progress Metrics to Track

For Priority Corrections, list the TOP 3 issues with the highest injury risk scores.
For Drill Recommendations, give at least 4 specific, named drills with sets/reps.
For Injury Prevention, include warm-up, strengthening, and recovery protocols.
For Progress Metrics, list 5 measurable targets the pitcher can track over 4–6 weeks.

Remember: speak directly to the pitcher ("your elbow flexion is…", "focus on…").
Do NOT include any preamble, sign-off, or text outside the 5 sections.
"""
    return prompt


# ── Main coaching function ────────────────────────────────────────────────────
def generate_coaching_plan(
    features: Dict[str, float],
    injury: Dict,
    outcome_result: Optional[Dict],
    api_key: str,
) -> Optional[str]:
    """
    Call Gemini 2.5 Flash and return an HTML coaching report string.
    Returns None on failure (caller handles st.error display).
    """
    if not api_key or not api_key.strip():
        return None

    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai package not installed. Add 'google-genai' to requirements.txt."
        )

    prompt = _build_prompt(features, injury, outcome_result)

    try:
        client   = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw_text = response.text or ""
    except Exception as exc:
        error_msg = str(exc)
        # Surface a clean error message
        if "API_KEY" in error_msg.upper() or "invalid" in error_msg.lower():
            raise ValueError(
                "❌ Invalid Gemini API key. Please check your key at aistudio.google.com."
            )
        elif "quota" in error_msg.lower() or "429" in error_msg:
            raise ValueError(
                "⏳ Gemini API quota exceeded. Please wait a moment and try again."
            )
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            raise ValueError(
                "❌ Model 'gemini-2.5-flash' not available. Check your API tier."
            )
        else:
            raise ValueError(f"❌ Gemini API error: {error_msg[:200]}")

    # Strip any accidental markdown code fences
    html_content = _strip_code_fences(raw_text)

    # Wrap in our styled container
    final_html = _wrap_report(html_content, injury, outcome_result)
    return final_html


# ── Post-processing helpers ───────────────────────────────────────────────────
def _strip_code_fences(text: str) -> str:
    """Remove ```html ... ``` or ``` ... ``` wrappers if Gemini added them."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```html or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines[0].startswith("```"):
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return text


def _risk_color(level: str) -> str:
    return {"Low": "#00c853", "Medium": "#ffab00", "High": "#ff5252"}.get(level, "#00c853")


def _wrap_report(html_content: str, injury: Dict, outcome_result: Optional[Dict]) -> str:
    """Wrap the Gemini response in a branded header + styled container."""
    risk_level = injury.get("overall_risk", "Low")
    risk_index = injury.get("risk_index", 0)
    delivery   = (outcome_result.get("label", "—").replace("_", " ") if outcome_result else "—")
    conf       = outcome_result.get("confidence", 0.0) if outcome_result else 0.0
    col        = _risk_color(risk_level)

    header = f"""
<div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1.4rem;
     padding-bottom:1rem;border-bottom:1px solid #2a2a3e;">
  <div style="text-align:center;">
    <div style="font-size:2rem;font-weight:700;color:{col};line-height:1">{risk_index}</div>
    <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:1.5px;color:#888">Risk Index</div>
  </div>
  <div style="width:1px;height:40px;background:#2a2a3e;"></div>
  <div style="text-align:center;">
    <div style="font-size:1rem;font-weight:600;color:#7c4dff">{delivery}</div>
    <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:1.5px;color:#888">Delivery Type</div>
  </div>
  <div style="width:1px;height:40px;background:#2a2a3e;"></div>
  <div style="text-align:center;">
    <div style="font-size:1rem;font-weight:600;color:{col}">{risk_level}</div>
    <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:1.5px;color:#888">Risk Level</div>
  </div>
  <div style="width:1px;height:40px;background:#2a2a3e;"></div>
  <div style="text-align:center;">
    <div style="font-size:1rem;font-weight:600;color:#00e5ff">{conf*100:.0f}%</div>
    <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:1.5px;color:#888">Confidence</div>
  </div>
  <div style="flex:1;text-align:right;">
    <span style="font-size:0.72rem;color:#444466;font-style:italic;">
      Generated by Gemini 2.5 Flash · PitchGuard AI
    </span>
  </div>
</div>
{html_content}
"""
    return header