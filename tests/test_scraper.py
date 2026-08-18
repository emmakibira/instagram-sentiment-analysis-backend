"""Tests for the scraper module (no network access)."""

from __future__ import annotations

import instaloader
import pytest

from instagram.scraper import (
    InstagramScraper,
    InstagramScraperError,
    InvalidPostURLError,
    LoginRequiredError,
    PostNotFoundError,
    RateLimitError,
    TwoFactorRequiredError,
)
from utils.helpers import extract_shortcode


class TestInvalidURLs:
    """InstaloaderError-sharing class for URL/network error mapping."""

    def test_scrape_with_invalid_url_raises(self) -> None:
        scraper = InstagramScraper(sleep_ratio=0)
        with pytest.raises(InvalidPostURLError):
            scraper.scrape_post("https://google.com/not-an-instagram-post")

    def test_scrape_with_empty_url_raises(self) -> None:
        scraper = InstagramScraper(sleep_ratio=0)
        with pytest.raises(InvalidPostURLError):
            scraper.scrape_post("")

    def test_max_comments_is_clamped(self) -> None:
        scraper = InstagramScraper(sleep_ratio=0)

        class _Post:
            shortcode = "X"

            date_utc = None
            owner_username = "u"

            @property
            def caption(self):
                return ""

            @property
            def likes(self):
                return 1

            @property
            def comments(self):
                return 1

        # Monkey-patch the loader to avoid network calls.
        class _Ctx:
            pass

        scraper._get_post = lambda url: _Post()

        def fake_comments(post, max_comments):
            return []

        scraper._fetch_comments = fake_comments
        result = scraper.scrape_post("https://www.instagram.com/p/XXX/", max_comments=0)
        assert result["post_info"]["shortcode"] == "X"

    def test_shortcode_helper_matches_scraper_rules(self) -> None:
        assert extract_shortcode("https://www.instagram.com/p/AbC-123_/") == "AbC-123_"
        # Legacy instance method must stay in sync with the helper.
        scraper = InstagramScraper(sleep_ratio=0)
        assert scraper.extract_shortcode("https://www.instagram.com/reel/xyz/") == "xyz"


class TestErrorMapping:
    """Map Instaloader exceptions to friendly domain errors."""

    def test_error_classes_are_distinct(self) -> None:
        assert issubclass(InvalidPostURLError, InstagramScraperError)
        assert issubclass(PostNotFoundError, InstagramScraperError)
        assert issubclass(LoginRequiredError, InstagramScraperError)
        assert issubclass(RateLimitError, InstagramScraperError)


class TestConnect:
    """Explicit Instagram login via the Connect flow."""

    def _scraper(self) -> InstagramScraper:
        return InstagramScraper(autologin=False)

    def test_connect_success_saves_session(self) -> None:
        from unittest.mock import patch

        scraper = self._scraper()

        with patch.object(scraper.loader, "login"), patch.object(
            scraper.loader, "save_session_to_file"
        ):
            ok, msg = scraper.connect("testuser", "secret")

        assert ok is True
        assert "Connected as @testuser" in msg
        assert scraper.authenticated is True
        assert scraper.username == "testuser"

    def test_connect_bad_credentials(self) -> None:
        from unittest.mock import patch

        scraper = self._scraper()
        with patch.object(
            scraper.loader, "login", side_effect=instaloader.BadCredentialsException("nope")
        ):
            ok, msg = scraper.connect("testuser", "wrong")
        assert ok is False
        assert "Invalid username or password" in msg
        assert scraper.authenticated is False

    def test_connect_requires_two_factor(self) -> None:
        from unittest.mock import patch

        scraper = self._scraper()
        with patch.object(
            scraper.loader,
            "login",
            side_effect=instaloader.TwoFactorAuthRequiredException("2fa"),
        ), pytest.raises(TwoFactorRequiredError):
            scraper.connect("testuser", "secret")

    def test_complete_two_factor_success(self) -> None:
        from unittest.mock import patch

        scraper = self._scraper()
        with patch.object(scraper.loader, "two_factor_login"), patch.object(
            scraper.loader, "save_session_to_file"
        ):
            ok, msg = scraper.complete_two_factor("123456", "testuser")
        assert ok is True
        assert msg.startswith("Connected as @testuser")
        assert scraper.authenticated is True

    def test_complete_two_factor_bad_code(self) -> None:
        from unittest.mock import patch

        scraper = self._scraper()
        with patch.object(
            scraper.loader, "two_factor_login", side_effect=instaloader.BadCredentialsException("x")
        ):
            ok, msg = scraper.complete_two_factor("000000", "testuser")
        assert ok is False
        assert "Invalid two-factor" in msg

    def test_resend_two_factor_raises_after_login(self) -> None:
        from unittest.mock import patch

        scraper = self._scraper()
        with patch.object(
            scraper.loader,
            "login",
            side_effect=instaloader.TwoFactorAuthRequiredException("2fa"),
        ), pytest.raises(TwoFactorRequiredError) as excinfo:
            scraper.resend_two_factor_code("testuser", "secret")
        assert "fresh two-factor code" in str(excinfo.value)

    def test_resend_two_factor_success_when_not_needed(self) -> None:
        from unittest.mock import patch

        scraper = self._scraper()
        with patch.object(scraper.loader, "login"), patch.object(
            scraper.loader, "save_session_to_file"
        ):
            ok, msg = scraper.resend_two_factor_code("testuser", "secret")
        assert ok is True
        assert "no code needed" in msg

    def test_resend_two_factor_login_error(self) -> None:
        from unittest.mock import patch

        scraper = self._scraper()
        with patch.object(
            scraper.loader,
            "login",
            side_effect=instaloader.LoginException("checkpoint_challenge_required"),
        ):
            ok, msg = scraper.resend_two_factor_code("testuser", "secret")
        assert ok is False
        assert "security checkpoint" in msg

    def test_connect_checkpoint_uses_friendly_error(self) -> None:
        from unittest.mock import patch

        scraper = self._scraper()
        with patch.object(
            scraper.loader,
            "login",
            side_effect=instaloader.LoginException("checkpoint_challenge_required"),
        ):
            ok, msg = scraper.connect("testuser", "secret")
        assert ok is False
        assert "security checkpoint" in msg
        assert "checkpoint_challenge_required" in msg

    def test_connect_blocked_uses_friendly_error(self) -> None:
        from unittest.mock import patch

        scraper = self._scraper()
        with patch.object(
            scraper.loader,
            "login",
            side_effect=instaloader.LoginException("Your account has been blocked"),
        ):
            ok, msg = scraper.connect("testuser", "secret")
        assert ok is False
        assert "blocked" in msg.lower()


class TestAnonymousFirst:
    """Anonymous-by-default behavior and block retries."""

    def test_anonymous_by_default(self) -> None:
        scraper = InstagramScraper(sleep_ratio=0)
        assert scraper.authenticated is False
        assert scraper.username is None

    def test_anonymous_block_raises_helpful_error_after_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import patch

        scraper = InstagramScraper(sleep_ratio=0)
        monkeypatch.setattr("time.sleep", lambda secs: None)

        with patch.object(
            __import__("instaloader").Post,
            "from_shortcode",
            side_effect=instaloader.BadResponseException("blocked"),
        ), pytest.raises(LoginRequiredError) as excinfo:
            scraper._get_post("https://www.instagram.com/p/ABC123/")
        assert "create_session.py" in str(excinfo.value)

    def test_transient_block_then_success_retries_anonymously(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import patch

        scraper = InstagramScraper(sleep_ratio=0)
        monkeypatch.setattr("time.sleep", lambda secs: None)

        calls = {"n": 0}

        class _Post:
            shortcode = "ABC123"
            date_utc = None
            owner_username = "u"

        def _from_shortcode(context, shortcode):
            calls["n"] += 1
            if calls["n"] == 1:
                raise instaloader.BadResponseException("transient")
            return _Post()

        with patch.object(__import__("instaloader").Post, "from_shortcode", side_effect=_from_shortcode):
            post = scraper._get_post("https://www.instagram.com/p/ABC123/")
        assert post.shortcode == "ABC123"
        assert calls["n"] == 2
