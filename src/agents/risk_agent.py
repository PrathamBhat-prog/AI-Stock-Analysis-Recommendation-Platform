"""Risk analysis agent — volatility-based uncertainty overlay."""

from __future__ import annotations

import pandas as pd


class RiskAnalysisAgent:
    """
    Evaluates short-term risk from realized volatility.

    Interview: we use vol instead of a black-box 'risk score' because it is
    interpretable, cheap to compute, and directly ties to position sizing.
    """

    def analyze(self, df: pd.DataFrame) -> dict:
        latest = df.iloc[-1]

        if "Return_std_20" in df.columns:
            volatility = float(latest["Return_std_20"])
        elif "Volatility_20" in df.columns:
            volatility = float(latest["Volatility_20"])
        else:
            volatility = float(df["Close"].pct_change().rolling(20).std().iloc[-1])

        if volatility != volatility:  # NaN check
            volatility = 0.015

        risk_score = min(volatility / 0.03, 1.0)

        if risk_score < 0.33:
            risk_level = "LOW"
            reason = "Low 20-day volatility — relatively stable price action."
        elif risk_score < 0.66:
            risk_level = "MEDIUM"
            reason = "Moderate volatility — expect wider daily swings."
        else:
            risk_level = "HIGH"
            reason = "High volatility — elevated uncertainty; consider smaller positions."

        return {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "realized_vol_20d": round(volatility, 6),
            "reason": reason,
        }
