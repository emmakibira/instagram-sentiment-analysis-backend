"""Shared utility helpers for the Instagram sentiment analyzer.

This module collects small, reusable pieces of logic used across the
application: URL handling, JSON parsing, retry logic, result reshaping
and export utilities (CSV / Excel / PDF / JSON).
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------

# Matches /p/, /reel/ and /tv/ as well as the profile-prefixed form used by
# mobile share links: instagram.com/{username}/p/{shortcode}/
SHORTCODE_RE = re.compile(
    r"instagram\.com/(?:[A-Za-z0-9_.]{1,30}/)?(?:p|reel|tv)/([A-Za-z0-9_-]+)"
)

VALID_URL_RE = re.compile(
    r"^https?://(?:www\.)?instagram\.com/(?:[A-Za-z0-9_.]{1,30}/)?"
    r"(?:p|reel|tv)/[A-Za-z0-9_-]+/?(?:\?[^#\s]*)?(?:#[^\s]*)?$"
)


def extract_shortcode(url: str) -> str | None:
    """Extract the shortcode from an Instagram post URL.

    Supports ``/p/``, ``/reel/`` and ``/tv/`` paths, with or without a
    profile prefix and trailing slash.

    Args:
        url: The Instagram post URL.

    Returns:
        The post shortcode, or ``None`` when the URL does not look like an
        Instagram post URL.
    """
    if not url:
        return None
    match = SHORTCODE_RE.search(url.strip())
    return match.group(1) if match else None


def validate_instagram_url(url: str) -> bool:
    """Return ``True`` when the URL is a well-formed Instagram post URL."""
    return bool(url and VALID_URL_RE.match(url.strip()))


# ---------------------------------------------------------------------------
# Robust JSON parsing (LLM output)
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> dict[str, Any] | None:
    """Best-effort parsing of JSON returned by an LLM.

    Handles markdown code fences and trailing prose after a JSON object.

    Args:
        text: Raw LLM output.

    Returns:
        The parsed JSON object, or ``None`` when no valid JSON object was
        found.
    """
    if not text:
        return None

    candidates: list[str] = []

    # Whole response first.
    candidates.append(text.strip())

    # Content inside markdown code fences.
    fences = _CODE_FENCE_RE.findall(text)
    candidates.extend(f.strip() for f in fences)

    # A JSON object embedded anywhere in the text.
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        candidates.append(obj_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    return None


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------


def retry(
    tries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Callable[..., Any]:
    """Retry a function with exponential backoff.

    Args:
        tries: Maximum number of attempts.
        delay: Initial delay in seconds between attempts.
        backoff: Multiplier applied to the delay after every attempt.
        exceptions: Tuple of exception types that trigger a retry.
        on_retry: Optional callback invoked before retrying, receiving the
            attempt number and the caught exception.

    Returns:
        The wrapped function.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt, current_delay = 1, delay
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= tries:
                        raise
                    if on_retry:
                        on_retry(attempt, exc)
                    logger.warning(
                        "%s failed (attempt %s/%s): %s",
                        fn.__name__,
                        attempt,
                        tries,
                        exc,
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Sentiment label normalization
# ---------------------------------------------------------------------------

SENTIMENT_ALIASES = {
    "positive": "SATISFIED",
    "satisfied": "SATISFIED",
    "happy": "SATISFIED",
    "good": "SATISFIED",
    "negative": "UNSATISFIED",
    "unsatisfied": "UNSATISFIED",
    "unhappy": "UNSATISFIED",
    "bad": "UNSATISFIED",
    "neutral": "NEUTRAL",
    "informational": "NEUTRAL",
}


def normalize_sentiment(label: Any) -> str:
    """Map a raw LLM sentiment label to a canonical SATISFIED/UNSATISFIED/NEUTRAL."""
    if isinstance(label, str):
        normalized = SENTIMENT_ALIASES.get(label.strip().lower())
        if normalized:
            return normalized
        upper = label.strip().upper()
        if upper in ("SATISFIED", "UNSATISFIED", "NEUTRAL"):
            return upper
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Result reshaping
# ---------------------------------------------------------------------------


def comments_to_dataframe(result: dict[str, Any]) -> pd.DataFrame:
    """Flatten the analysis result into a tidy pandas DataFrame.

    Args:
        result: The result dict produced by :func:`agent.analyze_post`.

    Returns:
        A DataFrame with one row per analyzed comment.
    """
    rows: list[dict[str, Any]] = []
    for item in result.get("comments_analysis", []):
        rows.append(
            {
                "user": item.get("user", ""),
                "comment": item.get("comment", ""),
                "sentiment": item.get("sentiment", "NEUTRAL"),
                "confidence": round(float(item.get("confidence", 0.0)), 3),
                "reason": item.get("reason", ""),
                "timestamp": item.get("timestamp", ""),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------


def export_csv_bytes(result: dict[str, Any]) -> bytes:
    """Render the analysis as CSV bytes (with post metadata header)."""
    buffer = io.StringIO()
    summary = result.get("sentiment_summary", {})
    buffer.write("# Instagram Sentiment Analysis\n")
    buffer.write(f"# Post: {result.get('post_info', {}).get('url', '')}\n")
    buffer.write(
        "# Satisfaction rate: {satisfaction_rate}% | Comments analyzed: {total_analyzed}\n".format(
            **{k: summary.get(k, "") for k in ("satisfaction_rate", "total_analyzed")}
        )
    )
    buffer.write("#\n")
    comments_to_dataframe(result).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def export_excel_bytes(result: dict[str, Any]) -> bytes:
    """Render the analysis as an Excel workbook (metadata + comments sheets)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary = result.get("sentiment_summary", {})
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        comments_to_dataframe(result).to_excel(writer, sheet_name="Comments", index=False)
    return buffer.getvalue()


def export_pdf_bytes(result: dict[str, Any]) -> bytes:
    """Render a readable one-page PDF report of the analysis."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story: list[Any] = []

    post_info = result.get("post_info", {})
    summary = result.get("sentiment_summary", {})
    insights = result.get("key_insights", {})

    story.append(Paragraph("Instagram Sentiment Analysis Report", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"Post: {post_info.get('url', '')}", styles["Normal"]))
    story.append(
        Paragraph(
            f"Caption: {str(post_info.get('caption', ''))[:200] or '(no caption)'}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Summary", styles["Heading2"]))
    summary_data = [
        ["Comments analyzed", str(summary.get("total_analyzed", 0))],
        ["Satisfied", str(summary.get("satisfied", 0))],
        ["Unsatisfied", str(summary.get("unsatisfied", 0))],
        ["Neutral", str(summary.get("neutral", 0))],
        ["Satisfaction rate", f"{summary.get('satisfaction_rate', 0)}%"],
        ["Average confidence", f"{summary.get('average_confidence', 0):.2f}"],
    ]
    summary_table = Table(summary_data, colWidths=[2.5 * inch, 3.5 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Key Insights", styles["Heading2"]))
    themes_text = (
        "Positive themes: "
        + ", ".join(insights.get("positive_themes", []))
        + "<br/>Negative themes: "
        + ", ".join(insights.get("negative_themes", []))
        + "<br/>Recommendations: "
        + ", ".join(insights.get("recommendations", []))
    )
    story.append(Paragraph(themes_text, styles["Normal"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Comments", styles["Heading2"]))
    comment_rows: list[list[Any]] = [["User", "Sentiment", "Confidence", "Comment"]]
    for item in result.get("comments_analysis", [])[:40]:
        comment_rows.append(
            [
                item.get("user", ""),
                item.get("sentiment", ""),
                f"{float(item.get('confidence', 0)):.2f}",
                str(item.get("comment", ""))[:80],
            ]
        )
    comments_table = Table(comment_rows, colWidths=[1.1 * inch, 1.1 * inch, 0.9 * inch, 3.4 * inch])
    comments_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(comments_table)

    doc.build(story)
    return buffer.getvalue()


def export_json_bytes(result: dict[str, Any]) -> bytes:
    """Render the analysis as pretty-printed JSON bytes."""
    return json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
