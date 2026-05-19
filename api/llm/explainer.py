"""
LLM Explainer — Google Gemini 1.5 Flash
Generates human-readable DPF analysis explanations grounded in:
  - ML prediction output
  - Feature context
  - RAG-retrieved domain knowledge

Why Gemini Flash?
  ✅ Free tier: 15 requests/min, 1M tokens/day
  ✅ No credit card needed for development
  ✅ Fast (sub-2s for this prompt size)
  ✅ Easy to swap for OpenAI/Claude with one change
"""

import os
import json
import httpx
from typing import Dict, Any, List


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

SYSTEM_PROMPT = """You are an expert automotive engineer specializing in diesel engine emissions systems, 
specifically Diesel Particulate Filter (DPF) diagnostics and maintenance.

Your role is to analyze vehicle telemetry data and ML predictions, then provide:
1. A clear explanation of WHY the current soot load is at its predicted level
2. What the key contributing sensor readings mean
3. Specific, actionable recommendations for the driver

Rules:
- Be concise but thorough (aim for 3-4 short paragraphs)
- Use plain language a non-engineer driver can understand
- Always ground your explanation in the actual numbers provided
- Structure your response with these exact sections:
  📊 DIAGNOSIS
  🔍 KEY FACTORS
  ✅ RECOMMENDATIONS
  ⏱️ URGENCY

Do not repeat the raw numbers back verbatim — interpret them.
"""


async def generate_explanation(
    feature_context: str,
    rag_chunks: List[str],
    prediction: Dict[str, Any],
) -> Dict[str, str]:
    """
    Generate an LLM explanation using Gemini Flash.
    Returns dict with 'explanation' and 'model_used'.
    Falls back to a rule-based explanation if no API key is set.
    """
    if not GEMINI_API_KEY:
        return {
            "explanation": _rule_based_fallback(feature_context, prediction),
            "model_used": "rule-based-fallback (set GEMINI_API_KEY for LLM explanations)",
        }

    rag_context = "\n".join(f"• {chunk}" for chunk in rag_chunks)

    user_prompt = f"""
Analyze this vehicle's DPF status and explain it to the driver.

--- TELEMETRY & PREDICTION DATA ---
{feature_context}

--- RELEVANT DPF DOMAIN KNOWLEDGE ---
{rag_context}

Provide your expert analysis following the structure specified.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": SYSTEM_PROMPT + "\n\n" + user_prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 600,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"explanation": text, "model_used": GEMINI_MODEL}

    except httpx.HTTPStatusError as e:
        return {
            "explanation": _rule_based_fallback(feature_context, prediction),
            "model_used": f"rule-based-fallback (Gemini error: {e.response.status_code})",
        }
    except Exception as e:
        return {
            "explanation": _rule_based_fallback(feature_context, prediction),
            "model_used": f"rule-based-fallback (error: {str(e)})",
        }


def _rule_based_fallback(feature_context: str, prediction: Dict[str, Any]) -> str:
    """Rule-based explanation when no LLM API key is configured."""
    soot = prediction["soot_load_percent"]
    regen = prediction["regen_recommended"]

    if soot < 30:
        diagnosis = "Your DPF is in excellent health. Soot levels are well within normal operating range."
        urgency = "No action required. Continue normal driving."
    elif soot < 60:
        diagnosis = "Your DPF has moderate soot accumulation. This is normal for vehicles that do frequent city driving."
        urgency = "Low urgency. A 30-minute highway drive soon would help clear soot."
    elif soot < 80:
        diagnosis = "Your DPF has high soot levels. Regeneration conditions have not been met recently."
        urgency = "Medium urgency. Plan a highway drive at 60-80 km/h for at least 30 minutes."
    else:
        diagnosis = "Your DPF is critically clogged. Immediate regeneration is required to prevent costly damage."
        urgency = "HIGH URGENCY. Drive at highway speeds immediately or visit a service center."

    regen_advice = "Active regeneration is recommended." if regen else "Regeneration is not currently required."

    return f"""📊 DIAGNOSIS
{diagnosis} Current soot load: {soot:.1f}%.

🔍 KEY FACTORS
Based on the telemetry data, exhaust temperatures and driving patterns have been analyzed. {regen_advice}

✅ RECOMMENDATIONS
{"Perform a sustained highway drive to trigger passive regeneration." if regen else "Continue your current driving habits to maintain DPF health."}
Avoid prolonged idling and short urban trips if possible.

⏱️ URGENCY
{urgency}

💡 Note: Set GEMINI_API_KEY environment variable for AI-powered explanations."""
