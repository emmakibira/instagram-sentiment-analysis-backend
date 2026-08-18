"""LangChain agent and deterministic pipeline for the analysis workflow.

The recommended entry point is :func:`analyze_post`, which runs a reliable,
deterministic pipeline (scrape -> classify -> summarize -> insights).

A real LangChain agent built with ``create_agent`` is also provided
(:meth:`InstagramAnalyzerAgent.run_agent`). It exposes the three tools
``scrape_instagram_post``, ``analyze_comments`` and ``generate_insights``
and lets the model orchestrate the same workflow. The app can choose between
the two modes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

from analyzer import (
    DEFAULT_MODEL,
    build_llm,
    compute_summary,
)
from analyzer import (
    analyze_comments as run_comment_analysis,
)
from analyzer import (
    generate_insights as run_insights,
)
from instagram.scraper import InstagramScraper
from utils.helpers import (
    normalize_sentiment,
    parse_json,
    validate_instagram_url,
)

logger = logging.getLogger(__name__)

load_dotenv()

AGENT_SYSTEM_PROMPT = """You are an Instagram comment sentiment analysis agent.

When given an Instagram post URL, follow this exact workflow:
1. Call `scrape_instagram_post` with the URL to get post_info and comments.
2. Call `analyze_comments` with the scraped comments to get sentiment_summary
   and comments_analysis.
3. Call `generate_insights` with the analyzed comments to get key_insights.
4. Return the complete result as a single JSON object with exactly these keys:
   {
     "post_info": {...},
     "sentiment_summary": {...},
     "key_insights": {...},
     "comments_analysis": [...]
   }

Do not summarize; return the full JSON object produced by the tools."""


class InstagramAnalyzerAgent:
    """LangChain agent that orchestrates the analysis workflow."""

    def __init__(
        self,
        model: str | None = None,
        max_comments: int = 100,
        scraper: InstagramScraper | None = None,
        llm: Any | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            model: Groq model id.
            max_comments: Default number of comments to scrape.
            scraper: Reusable scraper instance.
            llm: Reusable Groq chat model.
        """
        self.model = model or DEFAULT_MODEL
        self.max_comments = max_comments
        self.scraper = scraper or InstagramScraper()
        self.llm = llm or build_llm(self.model)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _make_tools(self) -> list[Any]:
        """Build the agent's tool list (bound to this instance)."""
        scraper = self.scraper
        llm = self.llm

        @tool
        def scrape_instagram_post(url: str, max_comments: int = 100) -> dict[str, Any]:
            """Scrape an Instagram post's metadata and comments.

            Args:
                url: Instagram post URL (/p/, /reel/ or /tv/).
                max_comments: Maximum number of comments to scrape.
            """
            return scraper.scrape_post(url, max_comments=max_comments)

        @tool
        def analyze_comments(comments: list[dict[str, Any]]) -> dict[str, Any]:
            """Analyze the sentiment of a list of Instagram comments.

            Args:
                comments: List of comment dicts with "user" and "comment" keys.
            """
            analyzed = run_comment_analysis(comments, llm=llm)
            return {
                "sentiment_summary": compute_summary(analyzed),
                "comments_analysis": analyzed,
            }

        @tool
        def generate_insights(analyzed: list[dict[str, Any]]) -> dict[str, Any]:
            """Generate themes, recommendations and a summary from analyzed comments.

            Args:
                analyzed: List of analyzed comment dicts.
            """
            return run_insights(analyzed, llm=llm)

        return [scrape_instagram_post, analyze_comments, generate_insights]

    # ------------------------------------------------------------------
    # Agent orchestration
    # ------------------------------------------------------------------

    def build_agent(self, system_prompt: str = AGENT_SYSTEM_PROMPT) -> Any:
        """Create and return the LangChain agent."""
        return create_agent(
            self.llm,
            self._make_tools(),
            system_prompt=system_prompt,
            name="instagram_sentiment_agent",
        )

    def run_agent(self, url: str, max_comments: int | None = None) -> dict[str, Any]:
        """Run the full workflow through the LangChain agent.

        The agent's final JSON answer is returned directly; when parsing fails
        or the agent errors, the deterministic :func:`analyze_post` pipeline is
        used as a fallback.

        Args:
            url: Instagram post URL.
            max_comments: Maximum comments to scrape.

        Returns:
            The analysis result dict.
        """
        max_comments = max_comments or self.max_comments
        agent = self.build_agent()

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": url}]}
            )
            final_message = result["messages"][-1]
            content = getattr(final_message, "content", "") or ""
            parsed = parse_json(content)
            if parsed and "sentiment_summary" in parsed and "comments_analysis" in parsed:
                return parsed
            logger.warning("Agent output was not a complete result JSON; falling back.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent run failed (%s); falling back to pipeline.", exc)

        return analyze_post(url, max_comments=max_comments, llm=self.llm, scraper=self.scraper)


# ---------------------------------------------------------------------------
# Deterministic pipeline
# ---------------------------------------------------------------------------


def analyze_post(
    url: str,
    max_comments: int = 100,
    llm: Any | None = None,
    scraper: InstagramScraper | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Analyze an Instagram post end-to-end (deterministic pipeline).

    Steps:
        1. Validate the URL.
        2. Scrape post metadata and comments.
        3. Classify each comment's sentiment.
        4. Compute summary statistics.
        5. Generate themes and recommendations.

    Args:
        url: Instagram post URL.
        max_comments: Maximum number of comments to scrape/analyze.
        llm: Reusable Groq chat model.
        scraper: Reusable scraper instance.
        progress_cb: Optional ``callback(done, total)`` for progress reporting.
        model: Optional model id (used when ``llm`` is not given).

    Returns:
        The result dict matching the project's sample output format.

    Raises:
        ValueError: When the URL is invalid or no comments could be scraped.
    """
    if not validate_instagram_url(url):
        raise ValueError(
            "Invalid Instagram URL. Use a link like "
            "https://www.instagram.com/p/XXXXX/ (supports /p/, /reel/ and /tv/)."
        )

    llm = llm or build_llm(model)
    scraper = scraper or InstagramScraper()

    scraped = scraper.scrape_post(url, max_comments=max_comments)
    post_info = scraped["post_info"]
    comments = scraped["comments"]

    if not comments:
        raise ValueError(
            "No comments were scraped from this post. It may have no comments, "
            "or Instagram may be restricting access."
        )

    analyzed = run_comment_analysis(comments, llm=llm, progress_cb=progress_cb)
    sentiment_summary = compute_summary(analyzed)
    key_insights = run_insights(analyzed, llm=llm)

    comments_analysis = [
        {
            "user": item.get("user", ""),
            "comment": item.get("comment", ""),
            "timestamp": item.get("timestamp", ""),
            "sentiment": normalize_sentiment(item.get("sentiment")),
            "confidence": round(float(item.get("confidence", 0.0)), 3),
            "reason": item.get("reason", ""),
            "key_phrases": item.get("key_phrases", []),
        }
        for item in analyzed
    ]

    return {
        "post_info": post_info,
        "sentiment_summary": sentiment_summary,
        "key_insights": key_insights,
        "comments_analysis": comments_analysis,
    }
