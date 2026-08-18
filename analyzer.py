"""Sentiment analysis of Instagram comments using Groq chat models.

Comments are classified as SATISFIED / UNSATISFIED / NEUTRAL with a
confidence score, following the JSON schema described in the project brief.
Comments are analyzed in small batches using a thread pool to keep latency
and token cost reasonable.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from preprocessing.comment_cleaner import CommentCleaner
from utils.helpers import normalize_sentiment, parse_json, retry

logger = logging.getLogger(__name__)

load_dotenv()

#: Default Groq model used for classification and insights.
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

#: Maximum number of comments passed to the LLM at once.
BATCH_SIZE = 10

#: Number of parallel classification workers.
MAX_WORKERS = 4

#: Token budget guard for very long comments.
MAX_COMMENT_CHARS = 1000

SENTIMENT_SYSTEM_PROMPT = """You are an expert Instagram comment sentiment analyzer.

Classify each comment into exactly one of these categories:
- SATISFIED: Positive comments, praise, recommendations, excitement, happiness
- UNSATISFIED: Negative comments, complaints, criticism, problems, disappointment
- NEUTRAL: Informational, questions, neutral statements

Return ONLY a single JSON object (no markdown, no extra text) with this schema:
{
    "sentiment": "SATISFIED" or "UNSATISFIED" or "NEUTRAL",
    "confidence": 0.0 to 1.0,
    "reason": "Brief one-sentence justification",
    "key_phrases": ["extracted", "key", "phrases"]
}
"""

INSIGHTS_SYSTEM_PROMPT = """You are a customer-experience analyst.

Given the analyzed comments of an Instagram post (user, comment text,
sentiment and reason), identify the most common themes.

Return ONLY a single JSON object (no markdown, no extra text) with this schema:
{
    "positive_themes": ["theme", "theme"],
    "negative_themes": ["theme", "theme"],
    "recommendations": ["actionable recommendation", "..."],
    "summary": "A one or two sentence executive summary of the feedback."
}

Theme names should be short (1-3 words). Recommendations must be concrete and
actionable for a social media / product team. Only include a theme if it is
supported by the comments."""


def build_llm(model: str | None = None) -> ChatGroq:
    """Create a configured Groq chat model.

    Args:
        model: Model id. Defaults to ``GROQ_MODEL`` env var or
            ``openai/gpt-oss-120b``.

    Raises:
        ValueError: If ``GROQ_API_KEY`` is missing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Create a .env file with GROQ_API_KEY=... "
            "(see .env.example)."
        )
    return ChatGroq(
        model=model or DEFAULT_MODEL,
        api_key=api_key,
        temperature=0,
        max_tokens=512,
    )


@retry(tries=3, delay=1.0, backoff=2.0, exceptions=(Exception,))
def _classify_one(llm: ChatGroq, text: str) -> dict[str, Any]:
    """Classify a single cleaned comment via the LLM."""
    response = llm.invoke(
        [
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": f'Classify this Instagram comment: "{text}"'},
        ]
    )
    parsed = parse_json(response.content)
    if not parsed:
        raise ValueError(f"Could not parse JSON from model output: {response.content!r}")
    return parsed


def classify_comment(
    comment: str, llm: ChatGroq | None = None, model: str | None = None
) -> dict[str, Any]:
    """Classify a single comment and normalize the result.

    Args:
        comment: Raw comment text.
        llm: Reusable Groq model. A new one is created when omitted.
        model: Optional model id, used when ``llm`` is not given.

    Returns:
        A dict with ``sentiment``, ``confidence``, ``reason`` and
        ``key_phrases``. Invalid/empty comments fall back to NEUTRAL.
    """
    llm = llm or build_llm(model)
    cleaned = CommentCleaner().clean(comment)[:MAX_COMMENT_CHARS]

    if not cleaned:
        return {
            "sentiment": "NEUTRAL",
            "confidence": 0.0,
            "reason": "Empty or non-textual comment.",
            "key_phrases": [],
        }

    try:
        parsed = _classify_one(llm, cleaned)
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        phrases = parsed.get("key_phrases") or []
        if not isinstance(phrases, list):
            phrases = [str(phrases)]
        return {
            "sentiment": normalize_sentiment(parsed.get("sentiment")),
            "confidence": round(confidence, 3),
            "reason": str(parsed.get("reason", ""))[:300],
            "key_phrases": [str(p) for p in phrases][:10],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Classification failed, defaulting to NEUTRAL: %s", exc)
        return {
            "sentiment": "NEUTRAL",
            "confidence": 0.0,
            "reason": "Classification failed: " + str(exc)[:200],
            "key_phrases": [],
        }


def analyze_comments(
    comments: list[dict[str, Any]],
    llm: ChatGroq | None = None,
    model: str | None = None,
    batch_size: int = BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Analyze a list of comments in parallel batches.

    Args:
        comments: List of ``{"user", "comment", "timestamp"}`` dicts.
        llm: Reusable Groq model.
        model: Optional model id used when ``llm`` is not given.
        batch_size: Number of comments processed concurrently.
        max_workers: Thread pool size.
        progress_cb: Optional ``callback(done, total)`` invoked after each
            comment is processed.

    Returns:
        The input comments enriched with ``sentiment``, ``confidence``,
        ``reason`` and ``key_phrases``.
    """
    if not comments:
        return []

    llm = llm or build_llm(model)
    results: list[dict[str, Any]] = []
    total = len(comments)
    done = 0

    for start in range(0, total, batch_size):
        chunk = comments[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(classify_comment, item.get("comment", ""), llm): item
                for item in chunk
            }
            for future in as_completed(futures):
                item = futures[future]
                classification = future.result()
                done += 1
                if progress_cb:
                    progress_cb(done, total)
                results.append({**item, **classification})

    # Preserve the original comment order (ThreadPoolExecutor ordering is
    # non-deterministic once concurrent futures complete).
    order = {id(item): idx for idx, item in enumerate(comments)}
    results.sort(key=lambda r: order.get(id(r), 0))
    return results


def compute_summary(analyzed: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate classification results into summary statistics."""
    total = len(analyzed)
    if total == 0:
        return {
            "total_analyzed": 0,
            "satisfied": 0,
            "unsatisfied": 0,
            "neutral": 0,
            "satisfaction_rate": 0.0,
            "average_confidence": 0.0,
        }

    counts = {"SATISFIED": 0, "UNSATISFIED": 0, "NEUTRAL": 0}
    confidences: list[float] = []
    for item in analyzed:
        label = normalize_sentiment(item.get("sentiment"))
        counts[label] += 1
        confidences.append(float(item.get("confidence", 0.0)))

    satisfaction_rate = (
        counts["SATISFIED"] / total * 100 if total else 0.0
    )
    return {
        "total_analyzed": total,
        "satisfied": counts["SATISFIED"],
        "unsatisfied": counts["UNSATISFIED"],
        "neutral": counts["NEUTRAL"],
        "satisfaction_rate": round(satisfaction_rate, 1),
        "average_confidence": round(mean(confidences), 3),
    }


@retry(tries=2, delay=1.0, backoff=2.0, exceptions=(Exception,))
def generate_insights(
    analyzed: list[dict[str, Any]],
    llm: ChatGroq | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate themes and recommendations from the analyzed comments.

    Args:
        analyzed: Output of :func:`analyze_comments`.
        llm: Reusable Groq model.
        model: Optional model id used when ``llm`` is not given.

    Returns:
        A dict with ``positive_themes``, ``negative_themes``,
        ``recommendations`` and ``summary``.
    """
    default_insights = {
        "positive_themes": [],
        "negative_themes": [],
        "recommendations": [],
        "summary": "Not enough comments to generate insights.",
    }
    if not analyzed:
        return default_insights

    llm = llm or build_llm(model)
    # Compact payload: user, text, sentiment, reason.
    payload = []
    for item in analyzed:
        payload.append(
            f"- {item.get('user', '?')} [{item.get('sentiment', 'NEUTRAL')}]: "
            f"{str(item.get('comment', ''))[:200]} ({item.get('reason', '')[:100]})"
        )
    context = "\n".join(payload[:300])

    try:
        response = llm.invoke(
            [
                {"role": "system", "content": INSIGHTS_SYSTEM_PROMPT},
                {"role": "user", "content": f"Comments:\n{context}"},
            ]
        )
        parsed = parse_json(response.content)
        if not parsed:
            raise ValueError(f"Could not parse insights JSON: {response.content!r}")
        return {
            "positive_themes": [str(t) for t in (parsed.get("positive_themes") or [])],
            "negative_themes": [str(t) for t in (parsed.get("negative_themes") or [])],
            "recommendations": [str(r) for r in (parsed.get("recommendations") or [])],
            "summary": str(parsed.get("summary", ""))[:500],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Insights generation failed: %s", exc)
        return default_insights
