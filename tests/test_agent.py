"""Tests for the agent module and end-to-end pipeline (no network/API calls)."""

from __future__ import annotations

import pytest

from agent import InstagramAnalyzerAgent, analyze_post
from tests.conftest import FakeLLM, FakeScraper, make_comments


class TestAnalyzePostValidation:
    """URL validation in the deterministic pipeline."""

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid Instagram URL"):
            analyze_post("https://google.com", llm=FakeLLM(), scraper=FakeScraper([]))


class TestAnalyzePostPipeline:
    """Full deterministic pipeline with fake dependencies."""

    def test_full_run_returns_expected_structure(self) -> None:
        scraper = FakeScraper(make_comments(3))
        result = analyze_post(
            "https://www.instagram.com/p/ABC123/",
            llm=FakeLLM(),
            scraper=scraper,
            max_comments=100,
        )

        assert "post_info" in result
        assert "sentiment_summary" in result
        assert "key_insights" in result
        assert "comments_analysis" in result

        summary = result["sentiment_summary"]
        assert summary["total_analyzed"] == 3
        assert summary["satisfied"] == 3

        assert all(
            item["sentiment"] == "SATISFIED" for item in result["comments_analysis"]
        )
        assert all(
            "confidence" in item and "reason" in item
            for item in result["comments_analysis"]
        )

    def test_no_comments_raises(self) -> None:
        scraper = FakeScraper([])
        with pytest.raises(ValueError, match="No comments"):
            analyze_post(
                "https://www.instagram.com/p/ABC123/",
                llm=FakeLLM(),
                scraper=scraper,
            )

    def test_max_comments_respected(self) -> None:
        scraper = FakeScraper(make_comments(10))
        result = analyze_post(
            "https://www.instagram.com/p/ABC123/",
            llm=FakeLLM(),
            scraper=scraper,
            max_comments=4,
        )
        assert result["sentiment_summary"]["total_analyzed"] == 4


class TestAgent:
    """LangChain agent behavior."""

    def test_agent_constructs_tools(self) -> None:
        agent = InstagramAnalyzerAgent(
            model="openai/gpt-oss-120b",
            llm=FakeLLM(),
            scraper=FakeScraper([]),
        )
        tools = agent._make_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"scrape_instagram_post", "analyze_comments", "generate_insights"}

    def test_run_agent_falls_back_to_pipeline(self) -> None:
        scraper = FakeScraper(make_comments(2))
        instance = InstagramAnalyzerAgent(
            model="openai/gpt-oss-120b",
            llm=FakeLLM(),
            scraper=scraper,
        )
        # FakeLLM cannot do tool calling, so the agent graph construction fails
        # and run_agent must degrade gracefully to the deterministic pipeline.
        result = instance.run_agent("https://www.instagram.com/p/ABC123/")
        assert result["sentiment_summary"]["total_analyzed"] == 2

    def test_deterministic_and_agent_results_agree(self) -> None:
        scraper = FakeScraper(make_comments(2))
        instance = InstagramAnalyzerAgent(llm=FakeLLM(), scraper=scraper)
        agent_result = instance.run_agent("https://www.instagram.com/p/ABC123/")
        pipeline_result = analyze_post(
            "https://www.instagram.com/p/ABC123/",
            llm=FakeLLM(),
            scraper=scraper,
        )
        assert (
            agent_result["sentiment_summary"]["satisfied"]
            == pipeline_result["sentiment_summary"]["satisfied"]
        )
