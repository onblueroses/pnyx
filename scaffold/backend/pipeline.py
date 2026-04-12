"""Local feed-overlay scoring for the hackathon demo."""

from __future__ import annotations

import os
import re

import anthropic

DEMO_POSTS = [
    {
        "id": "post-1",
        "author": "Mira Hahn",
        "handle": "@civicwire",
        "timestamp": "2m ago",
        "text": "Officials are hiding the real numbers again. This proves the transit outage was deliberate. Share before the post disappears.",
    },
    {
        "id": "post-2",
        "author": "Jonas Keller",
        "handle": "@policynotes",
        "timestamp": "8m ago",
        "text": "Early reports suggest the outage may be linked to maintenance delays, but the public statement is still thin. According to the city update, the review is ongoing.",
    },
    {
        "id": "post-3",
        "author": "Lea Sommer",
        "handle": "@townhallfeed",
        "timestamp": "14m ago",
        "text": "I get why people are frustrated, but blaming one group without evidence makes this harder to solve. What information would actually help here?",
    },
    {
        "id": "post-4",
        "author": "Niko Brandt",
        "handle": "@hottakesdaily",
        "timestamp": "27m ago",
        "text": "Anyone who still believes the official line is naive. Everyone knows this was coordinated and the media will never admit it.",
    },
    {
        "id": "post-5",
        "author": "Sara Demir",
        "handle": "@briefingroom",
        "timestamp": "42m ago",
        "text": "The latest report cites two independent inspections and still leaves open questions. We should compare the claims before drawing conclusions.",
    },
]

HIGH_CLAIM_RISK = [
    "proves",
    "everyone knows",
    "nobody can deny",
    "cover-up",
    "coordinated",
    "deliberate",
    "they are hiding",
    "officials are hiding",
    "before the post disappears",
    "share before",
    "the media will never admit it",
]

SWEEPING_LANGUAGE = [
    "always",
    "never",
    "everyone",
    "nobody",
    "all of them",
    "anyone who",
]
GROUNDING_MARKERS = [
    "according to",
    "report",
    "data",
    "evidence",
    "inspection",
    "source",
    "sources",
    "study",
    "document",
    "independent",
    "update",
]
QUALIFIERS = [
    "may",
    "might",
    "could",
    "appears",
    "suggest",
    "unclear",
    "ongoing",
    "open questions",
]
REASONING_MARKERS = [
    "because",
    "according to",
    "for example",
    "evidence",
    "report",
    "data",
    "suggest",
    "therefore",
    "since",
    "compare",
]
LISTENING_MARKERS = [
    "i get why",
    "i see why",
    "what information",
    "what do you think",
    "we should compare",
    "help me understand",
    "open questions",
]
DISMISSIVE_MARKERS = [
    "naive",
    "idiot",
    "delusional",
    "you people",
    "obviously",
    "anyone who",
    "wake up",
    "never admit it",
]


def _get_claude_client() -> anthropic.Anthropic | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _count_matches(text: str, phrases: list[str]) -> int:
    matches = 0
    for phrase in phrases:
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
        if re.search(pattern, text):
            matches += 1
    return matches


def _clamp(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, value))


def _label_for_claim_risk(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 45:
        return "medium"
    return "low"


def _label_for_argument_quality(score: int) -> str:
    if score >= 70:
        return "strong"
    if score >= 45:
        return "mixed"
    return "weak"


def _label_for_engagement_quality(score: int) -> str:
    if score >= 70:
        return "constructive"
    if score >= 45:
        return "tense"
    return "reactive"


def _score_claim_risk(normalized_text: str) -> int:
    score = 35
    negated_grounding = _count_matches(
        normalized_text,
        [
            "no evidence",
            "without evidence",
            "no source",
            "no sources",
            "any source",
            "any sources",
            "no data",
            "without data",
            "lack of evidence",
            "lack of data",
        ],
    )
    score += 18 * _count_matches(normalized_text, HIGH_CLAIM_RISK)
    score += 8 * _count_matches(normalized_text, SWEEPING_LANGUAGE)
    score += 12 * negated_grounding
    score -= 10 * max(
        0, _count_matches(normalized_text, GROUNDING_MARKERS) - negated_grounding
    )
    score -= 6 * _count_matches(normalized_text, QUALIFIERS)
    return _clamp(score)


def _score_argument_quality(normalized_text: str, word_count: int) -> int:
    score = 35
    score += 10 * _count_matches(normalized_text, REASONING_MARKERS)
    score += 6 * _count_matches(normalized_text, QUALIFIERS)
    if word_count < 14:
        score -= 18
    if normalized_text.count("!") >= 2:
        score -= 10
    return _clamp(score)


def _score_engagement_quality(normalized_text: str) -> int:
    score = 52
    score += 12 * _count_matches(normalized_text, LISTENING_MARKERS)
    score -= 14 * _count_matches(normalized_text, DISMISSIVE_MARKERS)
    return _clamp(score)


def _compose_explanation(
    claim_risk: int,
    argument_quality: int,
    engagement_quality: int,
    dismissive_count: int,
    normalized_text: str,
) -> str:
    if claim_risk >= 70:
        if _count_matches(normalized_text, GROUNDING_MARKERS) == 0:
            return "High claim risk because the post makes a strong accusation without evidence or attribution."
        return "Claim risk is elevated because the post uses certainty and urgency faster than it provides support."
    if argument_quality < 45:
        return "Argument quality is weak because the post states a conclusion without showing how it follows."
    if engagement_quality < 45 and dismissive_count:
        return "Engagement quality is low because the wording dismisses disagreement instead of inviting a response."
    return "This post qualifies its claims, leaves room for uncertainty, and keeps the conversation open."


def _tighten_with_claude(explanation: str, analysis: dict) -> str:
    client = _get_claude_client()
    if not client:
        return explanation

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=60,
            timeout=3.0,
            system=(
                "Rewrite the explanation as one plain sentence. Stay under 18 words. "
                "Do not give a truth verdict. Do not use moderation language."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Explanation: {explanation}\n"
                        f"Scores: {analysis['claim_risk']['score']}, {analysis['argument_quality']['score']}, {analysis['engagement_quality']['score']}"
                    ),
                }
            ],
        )
        block = response.content[0]
        text = getattr(block, "text", None)
        return text.strip() if text else explanation
    except anthropic.APIError:
        return explanation


def _cta_for_analysis(
    claim_risk: int, argument_quality: int, engagement_quality: int
) -> str:
    if engagement_quality < 45:
        return "Listen first"
    if claim_risk >= 70:
        return "Ask for evidence"
    if argument_quality < 45:
        return "Add a reason"
    return "Keep the exchange open"


def analyze_text(text: str, use_claude: bool = False) -> dict:
    normalized_text = _normalize(text)
    word_count = len(normalized_text.split())
    dismissive_count = _count_matches(normalized_text, DISMISSIVE_MARKERS)

    claim_risk = _score_claim_risk(normalized_text)
    argument_quality = _score_argument_quality(normalized_text, word_count)
    engagement_quality = _score_engagement_quality(normalized_text)
    explanation = _compose_explanation(
        claim_risk,
        argument_quality,
        engagement_quality,
        dismissive_count,
        normalized_text,
    )

    analysis = {
        "claim_risk": {
            "score": claim_risk,
            "label": _label_for_claim_risk(claim_risk),
        },
        "argument_quality": {
            "score": argument_quality,
            "label": _label_for_argument_quality(argument_quality),
        },
        "engagement_quality": {
            "score": engagement_quality,
            "label": _label_for_engagement_quality(engagement_quality),
        },
        "explanation": explanation,
        "cta": _cta_for_analysis(claim_risk, argument_quality, engagement_quality),
    }
    if use_claude:
        analysis["explanation"] = _tighten_with_claude(explanation, analysis)
    return analysis


def get_demo_feed() -> list[dict]:
    return [
        {
            **post,
            "analysis": analyze_text(post["text"]),
        }
        for post in DEMO_POSTS
    ]
