"""Instagram post scraping using Instaloader.

The :class:`InstagramScraper` downloads the metadata of a public Instagram
post together with a bounded number of its comments. It works anonymously
for public posts and can optionally authenticate with username/password or
a saved session file when Instagram demands a login (private content, some
rate-limited endpoints).

Only ``instaloader`` is required; no Instagram API tokens are needed.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import instaloader
from dotenv import load_dotenv

from utils.helpers import extract_shortcode, retry

logger = logging.getLogger(__name__)

load_dotenv()

#: Default number of comments scraped per post.
DEFAULT_MAX_COMMENTS = 100

#: Multiplier applied to Instaloader's internal rate-limit sleeps. Lower is
#: faster but more likely to hit Instagram rate limits.
SLEEP_RATIO = 0.1

SCRAPE_RETRIES = 3


def _username_from_session_file(session_file: str) -> str | None:
    """Derive the username from a default Instaloader session filename."""
    name = os.path.basename(session_file)
    if name.startswith("session-"):
        return name[len("session-") :]
    return None


def _friendly_login_error(exc: Exception) -> str:
    """Turn an Instaloader ``LoginException`` into user-facing guidance."""
    message = str(exc)
    lowered = message.lower()
    if "checkpoint" in lowered:
        return (
            "Instagram is asking for a security checkpoint. Open the link below "
            "in a browser (while logged into the Instagram app) and approve the "
            "login, then retry. "
            f"{message}"
        )
    if "blocked" in lowered or "too many" in lowered:
        return (
            "Instagram rejected the login attempt (possible IP block or too many "
            "attempts). Wait a few minutes and try again. " + message
        )
    return f"Login failed: {message}"


def _rate_controller_factory(sleep_ratio: float) -> Any:
    """Build a rate controller that scales Instaloader's built-in sleeps.

    Instaloader 4.15+ expects a factory ``Callable[[InstaloaderContext],
    RateController]``. The returned controller scales every wait by
    ``sleep_ratio`` so callers can trade speed against rate-limit safety.
    """

    class _ScaledRateController(instaloader.RateController):
        def sleep(self, secs: float) -> None:
            super().sleep(max(secs * sleep_ratio, 0.0))

    return _ScaledRateController


class InstagramScraperError(Exception):
    """Base class for scraping errors."""


class InvalidPostURLError(InstagramScraperError):
    """Raised when the supplied URL is not a valid Instagram post URL."""


class PostNotFoundError(InstagramScraperError):
    """Raised when the post does not exist or was deleted."""


class LoginRequiredError(InstagramScraperError):
    """Raised when the post is private or requires authentication."""


class TwoFactorRequiredError(InstagramScraperError):
    """Raised when the account requires a two-factor authentication code."""


class RateLimitError(InstagramScraperError):
    """Raised when Instagram throttles the scraping."""


class InstagramScraper:
    """Scrape Instagram post metadata and comments via Instaloader."""

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        session_file: str | None = None,
        sleep_ratio: float = SLEEP_RATIO,
        autologin: bool = True,
    ) -> None:
        """Initialize the scraper.

        Credentials may be passed explicitly or read from the environment
        variables ``INSTAGRAM_USERNAME``, ``INSTAGRAM_PASSWORD`` and
        ``INSTAGRAM_SESSION_FILE``.

        Args:
            username: Optional Instagram username.
            password: Optional Instagram password.
            session_file: Path to an Instaloader session file.
            sleep_ratio: Rate-limiting sleep ratio.
            autologin: When False, skip the automatic login attempt in the
                constructor (used by the app's "Connect to Instagram" flow).
        """
        self.username = username or os.getenv("INSTAGRAM_USERNAME")
        self.password = password or os.getenv("INSTAGRAM_PASSWORD")
        self.session_file = session_file or os.getenv("INSTAGRAM_SESSION_FILE")

        self.loader = instaloader.Instaloader(
            rate_controller=_rate_controller_factory(sleep_ratio), quiet=True
        )
        self.authenticated = False
        if autologin:
            self._login_if_possible()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _login_if_possible(self) -> None:
        """Attempt authentication using a session file, then credentials."""
        try:
            if self.session_file and os.path.exists(self.session_file):
                # Username can be omitted: derive it from the default session
                # filename pattern ("session-<username>") when possible.
                username = self.username or _username_from_session_file(self.session_file)
                self.loader.load_session_from_file(username, self.session_file)
                self.authenticated = True
                logger.info("Loaded Instagram session from %s", self.session_file)
                return

            if self.username and self.password:
                self.loader.login(self.username, self.password)
                self.authenticated = True
                logger.info("Logged in to Instagram as %s", self.username)
                return
        except instaloader.BadCredentialsException as exc:
            logger.warning("Invalid Instagram credentials: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Instagram login failed (continuing anonymously): %s", exc)

    def connect(self, username: str, password: str) -> tuple[bool, str]:
        """Explicitly log in to Instagram and save a session file.

        Intended for the app's "Connect to Instagram" button. The session file
        is persisted as ``session-<username>`` so the login survives app
        restarts. Anonymous scraping is unaffected when this is not called.

        Args:
            username: Instagram username.
            password: Instagram password.

        Returns:
            ``(True, message)`` on success.

        Raises:
            TwoFactorRequiredError: When the account needs a 2FA code — call
                :meth:`complete_two_factor` with the code.
        """
        try:
            self.loader.login(username, password)
        except instaloader.TwoFactorAuthRequiredException as exc:
            raise TwoFactorRequiredError(
                "Two-factor authentication is enabled on this account."
            ) from exc
        except instaloader.BadCredentialsException:
            return False, "Invalid username or password."
        except instaloader.ConnectionException as exc:
            return False, f"Could not reach Instagram: {exc}"
        except instaloader.LoginException as exc:
            return False, _friendly_login_error(exc)

        return self._finalize_login(username)

    def resend_two_factor_code(self, username: str, password: str) -> tuple[bool, str]:
        """Re-trigger Instagram's 2FA code delivery.

        Instagram only sends a code after a login attempt, so the login call
        is repeated. Re-runs should deliver a fresh code to the account's
        2FA method (WhatsApp, SMS text message, authenticator app, or the
        Instagram app) — which one depends on how the account is configured.

        Raises:
            TwoFactorRequiredError: When Instagram asks for the (new) code.
        """
        try:
            self.loader.login(username, password)
        except instaloader.TwoFactorAuthRequiredException as exc:
            raise TwoFactorRequiredError(
                "A fresh two-factor code has been requested. Check WhatsApp, "
                "SMS, your authenticator app, or the Instagram app for the "
                "new code."
            ) from exc
        except instaloader.LoginException as exc:
            return False, _friendly_login_error(exc)
        except instaloader.ConnectionException as exc:
            return False, f"Could not reach Instagram: {exc}"
        return True, "Login succeeded — no code needed."

    def complete_two_factor(self, code: str, username: str) -> tuple[bool, str]:
        """Finish a two-factor-authenticated login and save the session file."""
        try:
            self.loader.two_factor_login(code)
        except instaloader.BadCredentialsException:
            return False, "Invalid two-factor authentication code."
        except instaloader.ConnectionException as exc:
            return False, f"Could not reach Instagram: {exc}"
        return self._finalize_login(username)

    def _finalize_login(self, username: str) -> tuple[bool, str]:
        """Mark the loader as authenticated and persist the session file."""
        self.authenticated = True
        self.username = username
        session_path = f"session-{username}"
        try:
            self.loader.save_session_to_file(session_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save session file: %s", exc)
            return True, f"Connected as @{username} (session could not be saved)."
        logger.info("Saved Instagram session to %s", session_path)
        return True, f"Connected as @{username}. Session saved to {session_path}."

    # ------------------------------------------------------------------
    # Post fetching
    # ------------------------------------------------------------------

    def _get_post(self, url: str) -> instaloader.Post:
        """Resolve a post shortcode to an Instaloader Post object.

        Anonymous access is always attempted first. Transient block responses
        (``BadResponseException`` / ``QueryReturnedForbiddenException``) are
        retried with backoff before the final error is raised, which makes
        anonymous scraping succeed in more cases.
        """
        shortcode = extract_shortcode(url)
        if not shortcode:
            raise InvalidPostURLError(
                "Invalid Instagram post URL. Expected a /p/, /reel/ or /tv/ link."
            )

        # Retry transient anonymous blocks; a single attempt when authenticated.
        attempts = 3 if not self.authenticated else 1
        last_block: BaseException | None = None

        for attempt in range(attempts):
            try:
                return instaloader.Post.from_shortcode(self.loader.context, shortcode)
            except (instaloader.BadResponseException, instaloader.QueryReturnedForbiddenException) as exc:
                last_block = exc
                if attempt < attempts - 1:
                    wait = 2.0 + attempt * 2.0
                    logger.info(
                        "Anonymous block on attempt %s/%s, retrying in %.1fs",
                        attempt + 1,
                        attempts,
                        wait,
                    )
                    time.sleep(wait)
            except instaloader.InvalidArgumentException as exc:
                raise InvalidPostURLError(str(exc)) from exc
            except instaloader.ProfileNotExistsException as exc:
                raise PostNotFoundError("Instagram profile not found.") from exc
            except instaloader.QueryReturnedNotFoundException as exc:
                raise PostNotFoundError("Post not found or was deleted.") from exc
            except instaloader.LoginRequiredException as exc:
                raise LoginRequiredError(
                    "Instagram requires login for this post. Add credentials in the sidebar "
                    "or set up a session file (see README: 'Creating an Instagram session')."
                ) from exc
            except instaloader.TooManyRequestsException as exc:
                raise RateLimitError(
                    "Instagram is rate limiting requests. Wait a few minutes and retry."
                ) from exc
            except instaloader.ConnectionException as exc:
                raise InstagramScraperError(
                    f"Network error while contacting Instagram: {exc}"
                ) from exc

        raise LoginRequiredError(
            "Instagram refused anonymous access to this post. Add your Instagram "
            "credentials in the sidebar (or INSTAGRAM_USERNAME / INSTAGRAM_PASSWORD "
            "in .env), or create a session file with `python create_session.py`, "
            "and retry."
        ) from last_block

    def _post_info(self, post: instaloader.Post) -> dict[str, Any]:
        """Build a serializable dict describing the post."""
        try:
            caption = post.caption or ""
        except Exception:  # noqa: BLE001
            caption = ""
        return {
            "url": f"https://www.instagram.com/p/{post.shortcode}/",
            "shortcode": post.shortcode,
            "caption": caption[:2000],
            "likes": getattr(post, "likes", 0) or 0,
            "comments_count": getattr(post, "comments", 0) or 0,
            "date": post.date_utc.strftime("%Y-%m-%d %H:%M:%S") if post.date_utc else "",
            "profile": post.owner_username or "",
            "media_type": getattr(post, "media_type", None),
        }

    # ------------------------------------------------------------------
    # Comments
    # ------------------------------------------------------------------

    def _fetch_comments(self, post: instaloader.Post, max_comments: int) -> list[dict[str, Any]]:
        """Fetch up to ``max_comments`` comments from the post."""
        comments: list[dict[str, Any]] = []
        try:
            for comment in post.get_comments():
                if len(comments) >= max_comments:
                    break
                comments.append(
                    {
                        "user": getattr(comment.owner, "username", "") or "unknown",
                        "comment": comment.text,
                        "timestamp": datetime.fromtimestamp(
                            comment.created_at_utc, tz=timezone.utc
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        if comment.created_at_utc
                        else "",
                        "likes": getattr(comment, "likes", 0) or 0,
                    }
                )
        except instaloader.TooManyRequestsException as exc:
            raise RateLimitError(
                "Instagram is rate limiting requests while loading comments."
            ) from exc
        except instaloader.LoginRequiredException as exc:
            raise LoginRequiredError(
                "Instagram requires login to read comments for this post. Add "
                "credentials in the sidebar or create a session file with "
                "`python create_session.py --username YOUR_USERNAME`."
            ) from exc
        except instaloader.QueryReturnedNotFoundException as exc:
            logger.warning("Could not load all comments: %s", exc)
        except instaloader.ConnectionException as exc:
            logger.warning("Comment fetch interrupted by network error: %s", exc)
        except (instaloader.BadResponseException, instaloader.QueryReturnedForbiddenException) as exc:
            raise LoginRequiredError(
                "Instagram refused to serve comments for this post. Add your "
                "Instagram credentials and retry."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error while loading comments: %s", exc)

        return comments

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @retry(tries=SCRAPE_RETRIES, delay=2.0, exceptions=(instaloader.ConnectionException,))
    def scrape_post(self, url: str, max_comments: int = DEFAULT_MAX_COMMENTS) -> dict[str, Any]:
        """Scrape a post and its comments.

        Args:
            url: Instagram post URL.
            max_comments: Maximum number of comments to collect (latest ones
                are prioritized).

        Returns:
            A dict with ``post_info`` and a list of ``comments``.

        Raises:
            InvalidPostURLError: The URL is not a valid Instagram post URL.
            PostNotFoundError: The post does not exist.
            LoginRequiredError: The post requires authentication.
            RateLimitError: Instagram is throttling the requests.
            InstagramScraperError: Any other scraping failure.
        """
        if not isinstance(url, str) or not url.strip():
            raise InvalidPostURLError("A post URL is required.")
        url = url.strip()
        max_comments = max(1, min(int(max_comments), 500))

        logger.info("Scraping post %s (max %s comments)", url, max_comments)
        post = self._get_post(url)
        post_info = self._post_info(post)
        comments = self._fetch_comments(post, max_comments)
        logger.info("Scraped %s comments from %s", len(comments), post_info["shortcode"])

        return {
            "post_info": post_info,
            "comments": comments,
        }

    # ------------------------------------------------------------------
    # Backwards-compatible helpers (legacy backend)
    # ------------------------------------------------------------------

    def extract_shortcode(self, url: str) -> str | None:
        """Extract a post shortcode from a URL.

        Backwards-compatible wrapper around :func:`utils.helpers.extract_shortcode`.
        """
        return extract_shortcode(url)

    def fetch_comments(self, post_url: str, max_comments: int = DEFAULT_MAX_COMMENTS) -> list[dict[str, Any]]:
        """Fetch comments in the legacy ``[{"text", "timestamp"}, ...]`` shape.

        Provided for compatibility with the older Flask/Django backend.
        New code should use :meth:`scrape_post` instead.
        """
        data = self.scrape_post(post_url, max_comments=max_comments)
        return [{"text": c["comment"], "timestamp": c["timestamp"]} for c in data["comments"]]
