"""Tests for URL handling, JSON parsing and export utilities."""

from __future__ import annotations

import pytest

from utils.helpers import (
    comments_to_dataframe,
    export_csv_bytes,
    export_json_bytes,
    export_pdf_bytes,
    extract_shortcode,
    normalize_sentiment,
    parse_json,
    validate_instagram_url,
)


class TestShortcodeExtraction:
    """URL extraction logic tests."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.instagram.com/p/ABC123/", "ABC123"),
            ("https://www.instagram.com/reel/ReelCode123/", "ReelCode123"),
            ("https://www.instagram.com/tv/TvCode/_", "TvCode"),
            ("https://www.instagram.com/some.user.1/p/Mixed_code/", "Mixed_code"),
            ("http://instagram.com/p/NoWww/", "NoWww"),
            ("garbage text with https://www.instagram.com/p/Embedded/ at end", "Embedded"),
        ],
    )
    def test_extracts_from_supported_formats(self, url: str, expected: str) -> None:
        assert extract_shortcode(url) == expected

    def test_handles_query_parameters_and_fragments(self) -> None:
        # Instagram share links always carry ?utm_source=...&igsh=... query params.
        share_url = (
            "https://www.instagram.com/reel/Db3MqkCtBx1/"
            "?utm_source=ig_web_copy_link&igsh=NTc4MTIwNjQ2YQ=="
        )
        assert extract_shortcode(share_url) == "Db3MqkCtBx1"
        assert validate_instagram_url(share_url)

        with_fragment = "https://www.instagram.com/p/AbC123/#reels"
        assert extract_shortcode(with_fragment) == "AbC123"
        assert validate_instagram_url(with_fragment)

    def test_returns_none_for_non_post_urls(self) -> None:
        assert extract_shortcode("https://www.instagram.com/explore/") is None
        assert extract_shortcode("https://google.com") is None
        assert extract_shortcode("") is None
        assert extract_shortcode(None) is None

    def test_validation(self) -> None:
        assert validate_instagram_url("https://www.instagram.com/p/ABC123/")
        assert validate_instagram_url("https://www.instagram.com/reel/R123/")
        assert not validate_instagram_url("https://www.instagram.com/")
        assert not validate_instagram_url("https://google.com/")
        assert not validate_instagram_url("")

class TestParseJson:
    """Robust JSON parsing tests."""

    def test_parses_plain_json(self) -> None:
        assert parse_json('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fences(self) -> None:
        raw = "```json\n{\"a\": 1}\n```"
        assert parse_json(raw) == {"a": 1}

    def test_extracts_embedded_object_with_trailing_prose(self) -> None:
        raw = 'Here is the result: {"sentiment": "SATISFIED"} Regards, AI.'
        assert parse_json(raw) == {"sentiment": "SATISFIED"}

    def test_returns_none_for_garbage(self) -> None:
        assert parse_json("not json at all") is None
        assert parse_json("") is None
        assert parse_json(None) is None

    def test_returns_first_object_from_array(self) -> None:
        assert parse_json('[{"x": 1}, {"y": 2}]') == {"x": 1}


class TestNormalizeSentiment:
    """Sentiment label normalization tests."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("SATISFIED", "SATISFIED"),
            ("positive", "SATISFIED"),
            ("UNSATISFIED", "UNSATISFIED"),
            ("negative", "UNSATISFIED"),
            ("neutral", "NEUTRAL"),
            ("informational", "NEUTRAL"),
            ("something weird", "NEUTRAL"),
            (None, "NEUTRAL"),
            (123, "NEUTRAL"),
        ],
    )
    def test_normalizes(self, raw: object, expected: str) -> None:
        assert normalize_sentiment(raw) == expected


class TestExports:
    """Export utility tests."""

    @pytest.fixture
    def result(self) -> dict:
        return {
            "post_info": {"url": "https://www.instagram.com/p/ABC123/", "caption": "hi"},
            "sentiment_summary": {
                "total_analyzed": 2,
                "satisfied": 1,
                "unsatisfied": 1,
                "neutral": 0,
                "satisfaction_rate": 50.0,
                "average_confidence": 0.8,
            },
            "key_insights": {
                "positive_themes": ["quality"],
                "negative_themes": ["price"],
                "recommendations": ["lower price"],
                "summary": "Mixed feedback.",
            },
            "comments_analysis": [
                {
                    "user": "a",
                    "comment": "Great!",
                    "sentiment": "SATISFIED",
                    "confidence": 0.9,
                    "reason": "positive",
                    "timestamp": "2026-01-01 00:00:00",
                },
                {
                    "user": "b",
                    "comment": "Bad.",
                    "sentiment": "UNSATISFIED",
                    "confidence": 0.7,
                    "reason": "negative",
                    "timestamp": "2026-01-01 00:00:01",
                },
            ],
        }

    def test_dataframe_shape(self, result: dict) -> None:
        df = comments_to_dataframe(result)
        assert list(df.columns) == ["user", "comment", "sentiment", "confidence", "reason", "timestamp"]
        assert len(df) == 2

    def test_csv_exports(self, result: dict) -> None:
        data = export_csv_bytes(result)
        assert isinstance(data, bytes)
        assert "Great!" in data.decode("utf-8")

    def test_json_exports(self, result: dict) -> None:
        data = export_json_bytes(result)
        assert '"sentiment_summary"' in data.decode("utf-8")

    def test_pdf_exports(self, result: dict) -> None:
        data = export_pdf_bytes(result)
        assert data.startswith(b"%PDF")
