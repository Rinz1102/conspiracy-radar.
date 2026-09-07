"""
detector.py
Core analysis engine for Conspiracy Radar.

Combines two layers of analysis:
1. Rule-based linguistic pattern detection (explainable, based on
   patterns researchers commonly associate with conspiratorial rhetoric)
2. A trained TF-IDF + Logistic Regression classifier (statistical layer)

The final score blends both so the tool is not just keyword spam,
and the breakdown is human-readable ("why did it flag this?").
"""

import re
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.joblib"

# ---------------------------------------------------------------------
# Linguistic pattern lexicons
# Each category represents a rhetorical pattern that misinformation /
# conspiracy research (e.g. work on "conspiracy cues") frequently flags.
# ---------------------------------------------------------------------

PATTERNS = {
    "Hidden Agent Framing": [
        r"\bthey (don't|do not) want you to know\b",
        r"\bthe (elites|elite|shadow government|deep state|puppet(s)?)\b",
        r"\bsecretly (control|controls|controlling)\b",
        r"\bpuppets?\b",
        r"\bpeople who (really|actually) run things\b",
        r"\bwho control(s)? the outcome\b",
        r"\banswers to no one\b",
    ],
    "Us-vs-Them Framing": [
        r"\bsheeple\b",
        r"\bordinary people\b",
        r"\bthe masses\b",
        r"\bmost people (don't|do not) (even )?realize\b",
        r"\bwake up\b",
        r"\bopen your eyes\b",
    ],
    "False Certainty": [
        r"\bnothing happens by accident\b",
        r"\bnone of this is random\b",
        r"\beverything (you've been taught|is a lie)\b",
        r"\bnot a coincidence\b",
        r"\bonce you see the pattern\b",
        r"\balways\b",
        r"\bnever\b",
        r"\beveryone who\b",
    ],
    "Missing / Vague Sourcing": [
        r"\bdo your own research\b",
        r"\bthe truth is being buried\b",
        r"\bofficial (story|explanation|statistics) (is|are) (fabricated|a lie)\b",
        r"\byou'll never (know|see)\b",
        r"\bno scientist will admit\b",
        r"\bindependent researchers\b",
    ],
    "Emotional Urgency": [
        r"\bwake up\b",
        r"\bpanic\b",
        r"\bconnect the dots\b",
        r"\bask yourself\b",
        r"\bfollow the money\b",
        r"\bthat alone should tell you\b",
        r"\bif you look closely\b",
        r"\bcan't unsee\b",
    ],
}

COMPILED_PATTERNS = {
    category: [re.compile(p, re.IGNORECASE) for p in phrases]
    for category, phrases in PATTERNS.items()
}


def extract_rule_features(text: str) -> dict:
    """Return per-category match counts and the exact phrases hit."""
    results = {}
    for category, compiled in COMPILED_PATTERNS.items():
        hits = []
        for pattern in compiled:
            for match in pattern.finditer(text):
                hits.append(match.group(0))
        results[category] = hits
    return results


def rule_based_score(text: str) -> tuple[float, dict]:
    """
    Score 0-1 from rule hits alone, using a soft cap so a handful of
    strong hits saturates the score rather than requiring dozens.
    """
    hits = extract_rule_features(text)
    total_hits = sum(len(v) for v in hits.values())
    categories_hit = sum(1 for v in hits.values() if len(v) > 0)
    # Blend raw hit count with category diversity — hitting multiple
    # DIFFERENT rhetorical patterns is a stronger signal than repeating one.
    raw = 0.15 * total_hits + 0.15 * categories_hit
    score = 1 - np.exp(-raw)  # saturates smoothly toward 1.0
    return float(score), hits


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def ml_score(text: str) -> float | None:
    model = get_model()
    if model is None:
        return None
    proba = model.predict_proba([text])[0]
    # class 1 = conspiracy-style
    classes = list(model.classes_)
    idx = classes.index(1)
    return float(proba[idx])


def analyze(text: str, ml_weight: float = 0.6) -> dict:
    """
    Full analysis pipeline. Returns a dict with:
      - final_score (0-1)
      - rule_score
      - ml_score (or None if model not trained yet)
      - breakdown: {category: [matched phrases]}
      - category_scores: {category: count} for charting
    """
    text = text.strip()
    if not text:
        return {
            "final_score": 0.0,
            "rule_score": 0.0,
            "ml_score": None,
            "breakdown": {},
            "category_scores": {},
        }

    r_score, hits = rule_based_score(text)
    m_score = ml_score(text)

    if m_score is not None:
        final = ml_weight * m_score + (1 - ml_weight) * r_score
    else:
        final = r_score

    category_scores = {cat: len(v) for cat, v in hits.items()}

    return {
        "final_score": round(float(final), 3),
        "rule_score": round(r_score, 3),
        "ml_score": round(m_score, 3) if m_score is not None else None,
        "breakdown": hits,
        "category_scores": category_scores,
    }


def label_for_score(score: float) -> str:
    if score < 0.25:
        return "Low — reads like neutral / factual text"
    elif score < 0.5:
        return "Mild — a few rhetorical flags, worth a closer look"
    elif score < 0.75:
        return "Elevated — multiple conspiracy-style patterns detected"
    else:
        return "High — strong conspiracy-style rhetorical signature"
