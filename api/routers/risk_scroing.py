"""
api/risk_scoring.py
===================

Explainable contract risk scoring.

Risk scale:
    0 - 29    -> LOW
    30 - 59   -> MEDIUM
    60 - 100  -> HIGH

The score is calculated from detected clauses and their confidence.
Higher-risk clauses add points.
Protective clauses reduce points.

This module does NOT perform clause classification.
It only converts detected clauses into a risk score.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# RISK WEIGHTS
# ============================================================
#
# These weights are converted into a 0–100 risk contribution.
# The values are maximum contribution points before confidence.
#

HIGH_RISK_CLAUSES = {
    "UNCAPPED_LIABILITY": 30,
    "LIQUIDATED_DAMAGES": 20,
    "NON_COMPETE": 15,
    "EXCLUSIVITY": 15,
    "IRREVOCABLE_OR_PERPETUAL_LICENSE": 20,
    "MOST_FAVORED_NATION": 15,
    "CHANGE_OF_CONTROL": 10,
    "MINIMUM_COMMITMENT": 10,
    "VOLUME_RESTRICTION": 10,
    "PRICE_RESTRICTIONS": 10,
    "NON_DISPARAGEMENT": 5,
    "COVENANT_NOT_TO_SUE": 10,
    "UNLIMITED_ALL_YOU_CAN_EAT_LICENSE": 10,
    "JOINT_IP_OWNERSHIP": 10,
    "AFFILIATE_LICENSE_LICENSOR": 5,
}


# Protective clauses reduce the overall risk.
# Values are negative because they reduce the score.
#

PROTECTIVE_CLAUSES = {
    "CAP_ON_LIABILITY": -15,
    "TERMINATION_FOR_CONVENIENCE": -10,
    "INSURANCE": -5,
    "AUDIT_RIGHTS": -5,
}


# ============================================================
# RISK THRESHOLDS
# ============================================================

LOW_MAX = 29
MEDIUM_MAX = 59
MAX_SCORE = 100


# ============================================================
# CLAUSE TYPE NORMALIZATION
# ============================================================

def normalize_clause_type(clause_type: Any) -> str:
    """
    Normalize classifier clause names so formatting
    differences do not break risk mapping.

    Examples:
        "Non-Compete" -> "NON_COMPETE"
        "non compete" -> "NON_COMPETE"
        "NON-COMPETE" -> "NON_COMPETE"
    """

    if clause_type is None:
        return ""

    value = str(clause_type).strip().upper()

    replacements = {
        "-": "_",
        " ": "_",
        "/": "_",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    while "__" in value:
        value = value.replace("__", "_")

    return value


# ============================================================
# CONFIDENCE
# ============================================================

def clamp_confidence(value: Any) -> float:
    """
    Keep confidence safely between 0 and 1.
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, value))


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_clause_name(clause_type: str) -> str:
    """
    Convert a clause label into readable text.

    Example:
        UNCAPPED_LIABILITY -> Uncapped Liability
    """

    return clause_type.replace("_", " ").title()


def get_risk_level(score: float) -> str:
    """
    Convert a 0–100 score into LOW / MEDIUM / HIGH.
    """

    score = max(0.0, min(MAX_SCORE, float(score)))

    if score <= LOW_MAX:
        return "LOW"

    if score <= MEDIUM_MAX:
        return "MEDIUM"

    return "HIGH"


def get_risk_color(risk_level: str) -> str:
    """
    Return the frontend color for the risk level.
    """

    if risk_level == "LOW":
        return "#22c55e"

    if risk_level == "MEDIUM":
        return "#f59e0b"

    return "#ef4444"


# ============================================================
# RISK SUMMARY
# ============================================================

def generate_risk_summary(
    score: float,
    risk_level: str,
    risk_reasons: list[dict[str, Any]],
    protective_factors: list[dict[str, Any]],
) -> str:
    """
    Generate a human-readable explanation of the risk score.
    """

    if not risk_reasons and not protective_factors:
        return (
            f"The contract has a {risk_level.lower()} risk level "
            f"with a score of {score}/100. "
            "No configured risk indicators were detected."
        )

    if risk_reasons:
        top_reasons = sorted(
            risk_reasons,
            key=lambda item: item.get("points", 0),
            reverse=True,
        )[:3]

        reason_text = ", ".join(
            item["reason"].lower()
            for item in top_reasons
        )

        summary = (
            f"The contract has a {risk_level.lower()} risk level "
            f"with a score of {score}/100. "
            f"The main contributing factors are: {reason_text}."
        )
    else:
        summary = (
            f"The contract has a {risk_level.lower()} risk level "
            f"with a score of {score}/100. "
            "No major risk factors were detected."
        )

    if protective_factors:
        summary += (
            " Protective clauses were also detected and "
            "reduced the overall score."
        )

    return summary


# ============================================================
# MAIN RISK CALCULATION
# ============================================================

def calculate_risk(clauses: list[Any]) -> dict[str, Any]:
    """
    Calculate explainable contract risk.

    Expected clause format:

        {
            "clause_type": "NON_COMPETE",
            "present": True,
            "confidence": 0.85
        }

    Supported object attributes:

        clause.clause_type
        clause.present
        clause.confidence

    Returns:

        {
            "risk_score": 45,
            "score": 45,
            "risk_level": "MEDIUM",
            "risk_color": "#f59e0b",
            "contributions": [...],
            "risk_reasons": [...],
            "protective_factors": [...],
            "clause_scores": [...],
            "summary": "..."
        }
    """

    raw_score = 0.0

    contributions: list[dict[str, Any]] = []
    risk_reasons: list[dict[str, Any]] = []
    protective_factors: list[dict[str, Any]] = []
    clause_scores: list[dict[str, Any]] = []

    for clause in clauses or []:

        # ----------------------------------------------------
        # Read clause fields
        # ----------------------------------------------------

        if isinstance(clause, dict):
            present = clause.get("present", False)
            clause_type = (
                clause.get("clause_type")
                or clause.get("type")
                or clause.get("label")
                or clause.get("category")
            )
            confidence = clause.get("confidence", 0.0)
        else:
            present = getattr(clause, "present", False)
            clause_type = (
                getattr(clause, "clause_type", None)
                or getattr(clause, "type", None)
                or getattr(clause, "label", None)
                or getattr(clause, "category", None)
            )
            confidence = getattr(clause, "confidence", 0.0)

        # Ignore clauses that were not detected
        if not present:
            continue

        clause_type = normalize_clause_type(clause_type)
        confidence = clamp_confidence(confidence)

        if not clause_type:
            continue

        readable_name = format_clause_name(clause_type)

        # ----------------------------------------------------
        # High-risk clause
        # ----------------------------------------------------

        if clause_type in HIGH_RISK_CLAUSES:
            weight = HIGH_RISK_CLAUSES[clause_type]

            contribution = weight * confidence
            raw_score += contribution

            contribution_value = round(contribution, 1)

            contribution_item = {
                "clause_type": clause_type,
                "category": "HIGH_RISK",
                "confidence": round(confidence, 3),
                "weight": weight,
                "contribution": contribution_value,
            }

            contributions.append(contribution_item)

            risk_reasons.append({
                "clause": clause_type,
                "clause_type": clause_type,
                "points": contribution_value,
                "reason": f"{readable_name} clause detected",
                "risk_level": (
                    "HIGH" if contribution_value >= 20 else "MEDIUM"
                ),
            })

            clause_scores.append({
                "clause": readable_name,
                "clause_type": clause_type,
                "score": min(100, round(contribution_value / weight * 100)),
                "contribution": contribution_value,
                "risk_level": (
                    "HIGH" if contribution_value >= 20 else "MEDIUM"
                ),
            })

        # ----------------------------------------------------
        # Protective clause
        # ----------------------------------------------------

        elif clause_type in PROTECTIVE_CLAUSES:
            weight = PROTECTIVE_CLAUSES[clause_type]

            contribution = weight * confidence
            raw_score += contribution

            contribution_value = round(contribution, 1)

            contributions.append({
                "clause_type": clause_type,
                "category": "PROTECTIVE",
                "confidence": round(confidence, 3),
                "weight": weight,
                "contribution": contribution_value,
            })

            protective_factors.append({
                "clause": clause_type,
                "clause_type": clause_type,
                "points": abs(contribution_value),
                "reason": (
                    f"{readable_name} clause helps reduce contract risk"
                ),
            })

            clause_scores.append({
                "clause": readable_name,
                "clause_type": clause_type,
                "score": 0,
                "contribution": contribution_value,
                "risk_level": "LOW",
            })

        # ----------------------------------------------------
        # Neutral clause
        # ----------------------------------------------------

        else:
            contributions.append({
                "clause_type": clause_type,
                "category": "NEUTRAL",
                "confidence": round(confidence, 3),
                "weight": 0.0,
                "contribution": 0.0,
            })

    # ========================================================
    # FINAL SCORE
    # ========================================================

    # Prevent negative scores after protective clauses
    raw_score = max(0.0, raw_score)

    risk_score = min(
        MAX_SCORE,
        round(raw_score, 1),
    )

    risk_level = get_risk_level(risk_score)
    risk_color = get_risk_color(risk_level)

    summary = generate_risk_summary(
        score=risk_score,
        risk_level=risk_level,
        risk_reasons=risk_reasons,
        protective_factors=protective_factors,
    )

    return {
        # Existing fields
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "contributions": contributions,

        # New explainable fields
        "score": risk_score,
        "risk_reasons": risk_reasons,
        "protective_factors": protective_factors,
        "clause_scores": clause_scores,
        "summary": summary,
    }


# ============================================================
# SIMPLE FUNCTION FOR PIPELINE
# ============================================================

def compute_risk(clauses: list[Any]) -> tuple[float, str]:
    """
    Simple interface for pipeline.py.

    Returns:

        (risk_score, risk_level)

    Example:

        score, level = compute_risk(clauses)
    """

    result = calculate_risk(clauses)

    return (
        result["risk_score"],
        result["risk_level"],
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_clauses = [
        {
            "clause_type": "UNCAPPED_LIABILITY",
            "present": True,
            "confidence": 0.95,
        },
        {
            "clause_type": "NON_COMPETE",
            "present": True,
            "confidence": 0.85,
        },
        {
            "clause_type": "CAP_ON_LIABILITY",
            "present": True,
            "confidence": 0.90,
        },
        {
            "clause_type": "TERMINATION_FOR_CONVENIENCE",
            "present": True,
            "confidence": 0.90,
        },
        {
            "clause_type": "CONFIDENTIALITY",
            "present": True,
            "confidence": 0.95,
        },
    ]

    result = calculate_risk(test_clauses)

    print("\n========== RISK TEST ==========")
    print(f"Risk Score : {result['risk_score']} / 100")
    print(f"Risk Level : {result['risk_level']}")
    print(f"Risk Color : {result['risk_color']}")

    print("\nRisk Reasons:")
    for item in result["risk_reasons"]:
        print(item)

    print("\nProtective Factors:")
    for item in result["protective_factors"]:
        print(item)

    print("\nClause Scores:")
    for item in result["clause_scores"]:
        print(item)

    print("\nSummary:")
    print(result["summary"])