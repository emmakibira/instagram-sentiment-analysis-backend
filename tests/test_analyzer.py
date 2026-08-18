"""Tests for the sentiment analyzer (uses fake LLMs, no API calls)."""

from __future__ import annotations

import pytest

from analyzer import (
    analyze_comments,
    classify_comment,
    compute_summary,
    generate_insights,
)
from tests.conftest import FakeLLM, FakeLLMUnstructured, make_comments


class TestClassifyComment:
    """Single-comment classification."""

    def test_classifies_and_normalizes(self) -> None:
        llm = FakeLLM()
        result = classify_comment("Great product!", llm=llm)
        assert result["sentiment"] == "SATISFIED"
        assert result["confidence"] == pytest.approx(0.9)
        assert result["reason"] == "test"
        assert result["key_phrases"] == ["great", "love"]

    def test_empty_comment_is_neutral_without_llm(self) -> None:
        # Empty/short comments must never consult the LLM.
        result = classify_comment("", llm=FakeLLM())
        assert result["sentiment"] == "NEUTRAL"
        assert result["confidence"] == 0.0

    def test_unparseable_output_falls_back_to_neutral(self) -> None:
        llm = FakeLLMUnstructured()
        result = classify_comment("Whatever", llm=llm)
        assert result["sentiment"] == "NEUTRAL"
        assert result["confidence"] in (0.0, 0.5)

    def test_confidence_clamped_to_unit_range(self) -> None:
        llm = FakeLLM(
            sentiment_payload=(
                '{"sentiment": "SATISFIED", "confidence": 9.9, '
                '"reason": "x", "key_phrases": []}'
            )
        )
        result = classify_comment("ok", llm=llm)
        assert 0.0 <= result["confidence"] <= 1.0


class TestAnalyzeComments:
    """Batch analysis."""

    def test_batch_analysis(self) -> None:
        comments = make_comments(5)
        analyzed = analyze_comments(comments, llm=FakeLLM(), batch_size=2, max_workers=2)
        assert len(analyzed) == 5

    def test_order_preserved(self) -> None:
        comments = make_comments(5)
        analyzed = analyze_comments(comments, llm=FakeLLM(), batch_size=1, max_workers=1)
        assert [a["user"] for a in analyzed] == [
            c["user"] for c in comments
        ]

    def test_empty_input(self) -> None:
        assert analyze_comments([], llm=FakeLLM()) == []

    def test_progress_callback_invoked(self) -> None:
        calls: list[tuple] = []
        analyze_comments(
            make_comments(3),
            llm=FakeLLM(),
            batch_size=1,
            max_workers=1,
            progress_cb=lambda done, total: calls.append((done, total)),
        )
        assert calls == [(1, 3), (2, 3), (3, 3)]

    def test_comment_fields_enriched(self) -> None:
        analyzed = analyze_comments(make_comments(1), llm=FakeLLM())
        item = analyzed[0]
        for key in ("sentiment", "confidence", "reason", "key_phrases"):
            assert key in item


class TestComputeSummary:
    """Summary statistics."""

    def test_counts_and_rate(self) -> None:
        analyzed = [
            {"sentiment": "SATISFIED", "confidence": 0.9},
            {"sentiment": "SATISFIED", "confidence": 0.8},
            {"sentiment": "UNSATISFIED", "confidence": 0.7},
            {"sentiment": "NEUTRAL", "confidence": 0.5},
        ]
        summary = compute_summary(analyzed)
        assert summary["total_analyzed"] == 4
        assert summary["satisfied"] == 2
        assert summary["unsatisfied"] == 1
        assert summary["neutral"] == 1
        assert summary["satisfaction_rate"] == 50.0
        assert summary["average_confidence"] == pytest.approx(0.725)

    def test_empty(self) -> None:
        summary = compute_summary([])
        assert summary["total_analyzed"] == 0
        assert summary["satisfaction_rate"] == 0.0


class TestGenerateInsights:
    """Insights generation."""

    def test_generates_insights(self) -> None:
        analyzed = [
            {"user": "u", "comment": "Great", "sentiment": "SATISFIED", "reason": "r"}
        ]
        insights = generate_insights(analyzed, llm=FakeLLM())
        assert insights["positive_themes"] == ["quality"]
        assert insights["negative_themes"] == ["price"]
        assert insights["recommendations"] == ["lower price"]

    def test_empty_input_returns_defaults(self) -> None:
        insights = generate_insights([], llm=FakeLLM())
        assert insights["positive_themes"] == []
        assert insights["recommendations"] == []

    def test_unparseable_falls_back_without_raising(self) -> None:
        insights = generate_insights(
            [{"user": "u", "comment": "x", "sentiment": "NEUTRAL", "reason": ""}],
            llm=FakeLLMUnstructured(),
        )
        assert isinstance(insights["recommendations"], list)
