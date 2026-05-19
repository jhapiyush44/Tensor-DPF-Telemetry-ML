"""
Feature Context Builder
Converts raw ML input + prediction output into a structured natural-language
context string that can be injected into the LLM prompt alongside RAG chunks.
"""

from typing import Dict, Any


def build_feature_context(inputs: Dict[str, float], prediction: Dict[str, Any]) -> str:
    """
    Build a human-readable context string from telemetry inputs and ML predictions.
    This is passed to the LLM as factual grounding.
    """
    soot = prediction["soot_load_percent"]
    regen = prediction["regen_recommended"]
    ci = prediction.get("confidence_interval", 0.0)

    # Interpret soot severity
    if soot < 30:
        severity = "LOW (healthy)"
        severity_emoji = "🟢"
    elif soot < 60:
        severity = "MODERATE"
        severity_emoji = "🟡"
    elif soot < 80:
        severity = "HIGH"
        severity_emoji = "🟠"
    else:
        severity = "CRITICAL"
        severity_emoji = "🔴"

    # Interpret driving conditions
    speed = inputs.get("speed", 0)
    rpm = inputs.get("rpm", 0)
    load = inputs.get("engine_load", 0)
    exhaust_pre = inputs.get("exhaust_temp_pre", 0)
    exhaust_post = inputs.get("exhaust_temp_post", 0)
    flow = inputs.get("flow_rate", 0)
    idle_ratio = inputs.get("idle_ratio_30", 0)
    high_load_ratio = inputs.get("high_load_ratio_30", 0)
    temp_roll_10 = inputs.get("temp_roll_mean_10", 0)
    temp_roll_60 = inputs.get("temp_roll_mean_60", 0)
    temp_delta = inputs.get("temp_delta", 0)
    ambient = inputs.get("ambient_temp", 25)

    driving_mode = _classify_driving(speed, rpm, load)
    regen_capability = _assess_regen_capability(exhaust_pre, load, speed)

    context = f"""
=== ML PREDICTION RESULTS ===
Soot Load: {soot:.1f}% {severity_emoji} [{severity}]
Regeneration Recommended: {"YES ⚠️" if regen else "NO ✅"}
Confidence Interval: ±{ci:.1f}%

=== CURRENT TELEMETRY SNAPSHOT ===
Driving Mode: {driving_mode}
Engine RPM: {rpm:.0f} RPM
Vehicle Speed: {speed:.1f} km/h
Engine Load: {load*100:.1f}%
Exhaust Temp (Pre-DPF): {exhaust_pre:.1f}°C
Exhaust Temp (Post-DPF): {exhaust_post:.1f}°C
Exhaust Temp Delta: {temp_delta:.1f}°C
Exhaust Flow Rate: {flow:.1f} g/s
Ambient Temperature: {ambient:.1f}°C

=== BEHAVIORAL PATTERNS (Last 30 min) ===
Idle Ratio: {idle_ratio*100:.1f}% {"⚠️ High idling" if idle_ratio > 0.3 else "✅ Normal"}
High Load Ratio: {high_load_ratio*100:.1f}% {"✅ Good load" if high_load_ratio > 0.4 else "⚠️ Low load"}

=== TEMPERATURE TRENDS ===
Rolling Mean (10 min): {temp_roll_10:.1f}°C
Rolling Mean (60 min): {temp_roll_60:.1f}°C
Temp Trend: {"Heating up" if temp_roll_10 > temp_roll_60 else "Cooling down"}

=== REGENERATION CAPABILITY ASSESSMENT ===
Current Regen Potential: {regen_capability}
Regen Temperature Threshold Met: {"YES (≥550°C)" if exhaust_pre >= 550 else f"NO ({exhaust_pre:.0f}°C < 550°C)"}
""".strip()

    return context


def _classify_driving(speed: float, rpm: float, load: float) -> str:
    if speed < 10:
        return "Idling / Stationary"
    elif speed < 40:
        return "Urban / City Driving"
    elif speed < 80:
        return "Suburban / Mixed"
    elif speed >= 80:
        if load > 0.6:
            return "Highway High Load"
        return "Highway Cruise"
    return "Unknown"


def _assess_regen_capability(exhaust_temp: float, load: float, speed: float) -> str:
    if exhaust_temp >= 600 and load > 0.6:
        return "EXCELLENT — Active regeneration likely occurring"
    elif exhaust_temp >= 550:
        return "GOOD — Passive regeneration possible"
    elif exhaust_temp >= 400:
        return "POOR — Below regeneration threshold"
    else:
        return "NONE — Cold exhaust, soot accumulating"


def build_rag_query(inputs: Dict[str, float], prediction: Dict[str, Any]) -> str:
    """Build a semantic query string for FAISS retrieval."""
    soot = prediction["soot_load_percent"]
    regen = prediction["regen_recommended"]
    speed = inputs.get("speed", 0)
    exhaust = inputs.get("exhaust_temp_pre", 0)
    load = inputs.get("engine_load", 0)
    idle = inputs.get("idle_ratio_30", 0)

    parts = []
    if soot > 80:
        parts.append("critical soot load DPF regeneration urgent")
    elif soot > 60:
        parts.append("high soot load DPF regeneration needed")
    elif soot > 30:
        parts.append("moderate soot load DPF monitoring")
    else:
        parts.append("low soot load DPF healthy")

    if exhaust < 350:
        parts.append("cold exhaust temperature no regeneration")
    elif exhaust > 550:
        parts.append("high exhaust temperature passive regeneration")

    if idle > 0.3:
        parts.append("high idle ratio soot accumulation")

    if speed > 80:
        parts.append("highway driving regeneration conditions")
    elif speed < 30:
        parts.append("city driving short trips soot buildup")

    if regen:
        parts.append("regeneration recommended DPF action")

    return " ".join(parts)
