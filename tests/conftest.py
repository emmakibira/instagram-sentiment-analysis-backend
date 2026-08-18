"""Shared test fixtures (fake LLM and fake scraper)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class FakeLLM:
    """A minimal chat model stand-in returning canned content."""

    def __init__(
        self,
        sentiment_payload: str = (
            '{"sentiment": "SATISFIED", "confidence": 0.9, '
            '"reason": "test", "key_phrases": ["great", "love"]}'
        ),
        insights_payload: str = (
            '{"positive_themes": ["quality"], "negative_themes": ["price"], '
            '"recommendations": ["lower price"], "summary": "ok"}'
        ),
    ) -> None:
        self.sentiment_payload = sentiment_payload
        self.insights_payload = insights_payload
        self.calls: list[dict[str, Any]] = []

    def _content(self, messages: list[dict[str, Any]]) -> str:
        system = messages[0]["content"] if messages else ""
        if "sentiment analyzer" in system:
            return self.sentiment_payload
        return self.insights_payload

    def invoke(self, messages: list[dict[str, Any]]) -> SimpleNamespace:
        self.calls.append(messages)
        return SimpleNamespace(content=self._content(messages))


class FakeLLMUnstructured(FakeLLM):
    """An LLM that never returns parseable JSON (forces fallback paths)."""

    def _content(self, messages: list[dict[str, Any]]) -> str:
        return "Sorry, I cannot help with that."


class FakeScraper:
    """A scraper stand-in that never touches the network."""

    def __init__(
        self,
        comments: list[dict[str, Any]],
        post_info: dict[str, Any] | None = None,
    ) -> None:
        self.comments = comments
        self.post_info = post_info or {
            "url": "https://www.instagram.com/p/ABC123/",
            "shortcode": "ABC123",
            "caption": "Hello world",
            "likes": 10,
            "comments_count": len(comments),
            "date": "2026-08-18 12:00:00",
            "profile": "testuser",
            "media_type": 1,
        }

    def scrape_post(self, url: str, max_comments: int = 100) -> dict[str, Any]:
        return {
            "post_info": self.post_info,
            "comments": self.comments[:max_comments],
        }


def make_comments(n: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "user": f"user_{i}",
            "comment": f"Comment number {i}, great product and fast delivery!",
            "timestamp": "2026-08-18 12:00:00",
        }
        for i in range(n)
    ]
