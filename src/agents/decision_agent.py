"""
Multi-horizon decision agent with honest product messaging.

Interview talking point: we do NOT claim the ML model predicts 1-year prices.
It is trained on ~20-day direction; longer horizons blend trend analysis with
explicitly documented weights (see src/config/horizons.py).
"""

from __future__ import annotations

from src.config.horizons import (
    BUY_THRESHOLD,
    DEFAULT_HORIZON,
    HORIZONS,
    SELL_THRESHOLD,
)

# Re-export for backward compatibility
__all__ = ["HORIZONS", "DEFAULT_HORIZON", "MLDecisionAgent", "BUY_THRESHOLD", "SELL_THRESHOLD"]


class MLDecisionAgent:
    """Maps ML probability + trend score → BUY / SELL / HOLD with plain-English copy."""

    def decide(
        self,
        ml_result: dict,
        trend_result: dict | None = None,
        horizon_key: str = DEFAULT_HORIZON,
        risk_overlay: dict | None = None,
    ) -> dict:
        horizon_cfg = HORIZONS.get(horizon_key, HORIZONS[DEFAULT_HORIZON])
        horizon_days = horizon_cfg["days"]
        horizon_lbl = horizon_cfg["label"]
        ml_w = horizon_cfg["ml_weight"]
        trend_w = horizon_cfg["trend_weight"]

        if not ml_result.get("available", False):
            ml_score = 0.5
            model_name = "unavailable"
            ml_note = "ML model not ready — run `python train.py` or place Sniper v5 model in artifacts/models/."
        else:
            ml_score = ml_result["probability_up"]
            model_name = ml_result.get("model_name", "unknown")
            ml_note = None

        trend_score = 0.0
        trend_label = "Unknown"
        trend_summary = ""
        if trend_result:
            trend_score = float(trend_result.get("trend_score", 0.0))
            trend_label = trend_result.get("trend_label", "Unknown")
            trend_summary = trend_result.get("summary", "")

        trend_prob = (trend_score + 1.0) / 2.0
        composite = ml_w * ml_score + trend_w * trend_prob

        if composite >= BUY_THRESHOLD:
            decision = "BUY"
            confidence = composite
        elif composite <= SELL_THRESHOLD:
            decision = "SELL"
            confidence = 1.0 - composite
        else:
            decision = "HOLD"
            confidence = max(composite, 1.0 - composite)

        # Risk overlay: downgrade BUY in extreme volatility
        risk_adjusted = decision
        risk_note = ""
        if risk_overlay and decision == "BUY":
            if risk_overlay.get("risk_level") == "HIGH":
                risk_adjusted = "HOLD"
                risk_note = (
                    " BUY downgraded to HOLD due to elevated volatility "
                    f"(risk score {risk_overlay.get('risk_score', 'N/A')})."
                )

        ml_horizon = ml_result.get("forecast_horizon_days", horizon_cfg["ml_training_horizon_days"])

        if ml_note:
            technical_reason = ml_note
        else:
            technical_reason = (
                f"ML ({model_name}): {ml_score:.1%} P(up) over ~{ml_horizon} trading days. "
                f"Trend: {trend_label} (score {trend_score:+.2f}). "
                f"Composite for {horizon_lbl}: {composite:.1%} "
                f"(ML {ml_w:.0%} + Trend {trend_w:.0%})."
                f"{risk_note}"
            )

        plain_english = _plain_english(
            risk_adjusted, confidence, horizon_days, horizon_lbl,
            trend_label, ml_w, horizon_cfg,
        )

        return {
            "final_decision": risk_adjusted,
            "raw_decision": decision,
            "confidence": round(confidence, 4),
            "horizon": horizon_lbl,
            "horizon_key": horizon_key,
            "horizon_days": horizon_days,
            "composite_score": round(composite, 4),
            "ml_probability": round(ml_score, 4),
            "trend_score": round(trend_score, 4),
            "ml_weight": ml_w,
            "trend_weight": trend_w,
            "ml_training_horizon_days": horizon_cfg["ml_training_horizon_days"],
            "primary_signal": horizon_cfg["primary_signal"],
            "user_expectation": horizon_cfg["user_expectation"],
            "honest_disclaimer": horizon_cfg["honest_disclaimer"],
            "reasoning": technical_reason,
            "plain_english": plain_english,
            "trend_summary": trend_summary,
            "risk_overlay": risk_overlay or {},
            "agent_summary": {"ml": ml_result, "trend": trend_result or {}},
        }


def _plain_english(
    decision: str,
    confidence: float,
    horizon_days: int,
    horizon_lbl: str,
    trend_label: str,
    ml_weight: float,
    horizon_cfg: dict,
) -> str:
    pct = f"{confidence:.0%}"
    horizon_plain = {
        5: "about 1 week",
        21: "about 1 month",
        63: "about 3 months",
        126: "about 6 months",
        252: "about 1 year",
    }.get(horizon_days, f"{horizon_days} trading days")

    method_note = (
        "based mainly on our ML model (trained for ~20-day price direction)"
        if ml_weight >= 0.6 else
        "based on a blend of ML and trend analysis"
        if ml_weight >= 0.3 else
        "based mainly on long-term trend analysis (ML plays a small supporting role)"
    )

    disclaimer = horizon_cfg["honest_disclaimer"]

    if decision == "BUY":
        action = (
            f"Our analysis ({method_note}) suggests a bullish bias over {horizon_plain}. "
            f"Trend: {trend_label.lower()}. Confidence: {pct}. "
        )
    elif decision == "SELL":
        action = (
            f"Our analysis ({method_note}) suggests a bearish bias over {horizon_plain}. "
            f"Trend: {trend_label.lower()}. Confidence: {pct}. "
        )
    else:
        action = (
            f"No strong directional signal for {horizon_plain}. "
            f"Trend: {trend_label.lower()}. "
            f"Waiting for clearer confirmation is reasonable. "
        )

    return (
        f"{action}"
        f"Note: {disclaimer} "
        f"This is educational analysis, not financial advice."
    )
