"""Streamlit UI for the Instagram Sentiment Analyzer.

Run with::

    streamlit run app.py

The app scrapes an Instagram post, classifies the sentiment of its comments
using a Groq-powered LangChain model and presents a dashboard with
visualizations, insights and export options.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from wordcloud import WordCloud

from agent import InstagramAnalyzerAgent
from analyzer import (
    DEFAULT_MODEL,
    analyze_comments,
    build_llm,
    compute_summary,
    generate_insights,
)
from database.history_store import HistoryStore
from instagram.scraper import (
    InstagramScraper,
    LoginRequiredError,
    RateLimitError,
    TwoFactorRequiredError,
)
from preprocessing.comment_cleaner import CommentCleaner
from utils.helpers import (
    export_csv_bytes,
    export_excel_bytes,
    export_json_bytes,
    export_pdf_bytes,
    extract_shortcode,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Instagram Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #f7f7f9; border-radius: 12px; padding: 16px;
        border: 1px solid #e6e6eb;
    }
    .tag {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        margin: 2px; font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_llm(model: str) -> Any:
    """Return a cached Groq chat model."""
    return build_llm(model)


@st.cache_data(show_spinner=False, ttl=30 * 60)
def scrape_cached(
    url: str,
    max_comments: int,
    username: str,
    password: str,
    session_file: str,
) -> dict[str, Any]:
    """Scrape a post, caching the result so tabs/history don't re-hit Instagram."""
    scraper = InstagramScraper(
        username=username or None,
        password=password or None,
        session_file=session_file or None,
    )
    return scraper.scrape_post(url, max_comments=max_comments)


def _run_pipeline(
    url: str,
    max_comments: int,
    model: str,
    use_agent: bool,
    username: str = "",
    password: str = "",
    session_file: str = "",
    progress_cb: Any | None = None,
) -> dict[str, Any]:
    """Run the analysis pipeline (agent or deterministic) for one URL."""
    if use_agent:
        agent = InstagramAnalyzerAgent(
            model=model,
            max_comments=max_comments,
            scraper=InstagramScraper(
                username=username or None,
                password=password or None,
                session_file=session_file or None,
            ),
            llm=get_llm(model),
        )
        return agent.run_agent(url, max_comments=max_comments)

    scraped = scrape_cached(url, max_comments, username, password, session_file)
    post_info = scraped["post_info"]
    comments = scraped["comments"]
    if not comments:
        raise ValueError(
            "No comments were scraped from this post. It may have no comments, "
            "or Instagram may be restricting access."
        )

    llm = get_llm(model)
    analyzed = analyze_comments(comments, llm=llm, progress_cb=progress_cb)
    summary = compute_summary(analyzed)
    insights = generate_insights(analyzed, llm=llm)

    comments_analysis = []
    for item in analyzed:
        comments_analysis.append(
            {
                "user": item.get("user", ""),
                "comment": item.get("comment", ""),
                "timestamp": item.get("timestamp", ""),
                "sentiment": item.get("sentiment", "NEUTRAL"),
                "confidence": round(float(item.get("confidence", 0.0)), 3),
                "reason": item.get("reason", ""),
                "key_phrases": item.get("key_phrases", []),
            }
        )

    return {
        "post_info": post_info,
        "sentiment_summary": summary,
        "key_insights": insights,
        "comments_analysis": comments_analysis,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def sentiment_pie(result: dict[str, Any]) -> go.Figure:
    """Pie chart of the sentiment distribution."""
    summary = result["sentiment_summary"]
    fig = go.Figure(
        go.Pie(
            labels=["Satisfied", "Unsatisfied", "Neutral"],
            values=[
                summary["satisfied"],
                summary["unsatisfied"],
                summary["neutral"],
            ],
            hole=0.45,
            marker={"colors": ["#2ecc71", "#e74c3c", "#95a5a6"]},
        )
    )
    fig.update_layout(title="Sentiment Distribution", height=380)
    return fig


def confidence_gauge(avg_confidence: float) -> go.Figure:
    """Radial gauge showing the average confidence."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_confidence,
            number={"suffix": "", "valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#5b8def"},
                "steps": [
                    {"range": [0, 0.5], "color": "#fdeaea"},
                    {"range": [0.5, 0.8], "color": "#fdf3e7"},
                    {"range": [0.8, 1.0], "color": "#e8f6ec"},
                ],
            },
            title={"text": "Average Confidence"},
        )
    )
    fig.update_layout(height=320)
    return fig


def phrases_bar(result: dict[str, Any], top_n: int = 10) -> go.Figure:
    """Bar chart of the most frequent key phrases across comments."""
    counter: Counter = Counter()
    for item in result["comments_analysis"]:
        for phrase in item.get("key_phrases", []):
            phrase = str(phrase).strip().lower()
            if phrase:
                counter[phrase] += 1
    if not counter:
        return px.bar(title="No key phrases detected")

    top = counter.most_common(top_n)
    df = pd.DataFrame(top, columns=["phrase", "count"]).sort_values("count")
    fig = px.bar(
        df,
        x="count",
        y="phrase",
        orientation="h",
        title=f"Top {len(top)} Key Phrases",
        color="count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=400)
    return fig


def wordcloud_fig(result: dict[str, Any]) -> None:
    """Render a word cloud of the comment text."""
    cleaner = CommentCleaner()
    texts = [
        cleaner.clean(item.get("comment", ""))
        for item in result["comments_analysis"]
    ]
    text = " ".join(t for t in texts if t) or "no comments"

    fig, ax = plt.subplots(figsize=(10, 5))
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=150,
        collocations=False,
        colormap="viridis",
    ).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def sentiment_timeline(result: dict[str, Any]) -> go.Figure:
    """Sentiment trend over the comment timeline."""
    score_map = {"SATISFIED": 1, "NEUTRAL": 0, "UNSATISFIED": -1}
    points = []
    for item in result["comments_analysis"]:
        ts = item.get("timestamp", "")
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            dt = None
        points.append(
            {
                "x": dt,
                "y": score_map.get(item.get("sentiment", "NEUTRAL"), 0),
                "sentiment": item.get("sentiment", "NEUTRAL"),
                "comment": item.get("comment", "")[:60],
            }
        )

    points.sort(key=lambda p: (p["x"] is None, p["x"]))
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    has_dates = any(x is not None for x in xs)

    if has_dates:
        xs = [x if x is not None else datetime(1970, 1, 1, tzinfo=timezone.utc) for x in xs]
        xs_str = [x.strftime("%Y-%m-%d %H:%M") for x in xs]
    else:
        xs_str = [str(i + 1) for i in range(len(points))]

    df = pd.DataFrame(
        {"time": xs_str, "score": ys, "sentiment": [p["sentiment"] for p in points]}
    )

    if len(df) >= 3:
        df["rolling"] = df["score"].rolling(window=3, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["time"],
            y=df["score"],
            marker_color=["#2ecc71" if v == 1 else "#95a5a6" if v == 0 else "#e74c3c" for v in df["score"]],
            name="Comment sentiment",
        )
    )
    if "rolling" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["rolling"],
                mode="lines",
                name="Trend (rolling avg)",
                line={"color": "#5b8def", "width": 2},
            )
        )
    fig.update_layout(title="Sentiment Trend", xaxis_title="Time", yaxis_title="Score", height=380)
    return fig


def top_commenters(result: dict[str, Any], top_n: int = 5) -> pd.DataFrame:
    """Table of the most active commenters and their sentiment profile."""
    rows: dict[str, dict[str, Any]] = {}
    for item in result["comments_analysis"]:
        user = item.get("user", "unknown")
        entry = rows.setdefault(
            user, {"user": user, "comments": 0, "satisfied": 0, "unsatisfied": 0}
        )
        entry["comments"] += 1
        entry["satisfied"] += item.get("sentiment") == "SATISFIED"
        entry["unsatisfied"] += item.get("sentiment") == "UNSATISFIED"
    df = pd.DataFrame(list(rows.values())).sort_values("comments", ascending=False)
    return df.head(top_n)


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------


def _render_dashboard(result: dict[str, Any]) -> None:
    summary = result["sentiment_summary"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    rate = summary["satisfaction_rate"]
    c1.metric("🎯 Satisfaction", f"{rate}%", delta=f"{rate - 50:.1f} pts vs neutral")
    c2.metric("💬 Comments Analyzed", summary["total_analyzed"])
    c3.metric("✅ Satisfied", summary["satisfied"])
    c4.metric("❌ Unsatisfied", summary["unsatisfied"])
    c5.metric("➖ Neutral", summary["neutral"])
    c6.metric("🧠 Avg Confidence", f"{summary['average_confidence']:.2f}")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(sentiment_pie(result), use_container_width=True)
    with right:
        st.plotly_chart(confidence_gauge(summary["average_confidence"]), use_container_width=True)

    st.subheader("👑 Top Commenters")
    st.dataframe(
        top_commenters(result),
        use_container_width=True,
        hide_index=True,
        column_config={
            "user": "User",
            "comments": "Comments",
            "satisfied": "Satisfied",
            "unsatisfied": "Unsatisfied",
        },
    )


def _render_insights(result: dict[str, Any]) -> None:
    insights = result["key_insights"]
    st.markdown(f"**Summary:** {insights.get('summary', '') or '—'}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### ✅ Positive Themes")
        for theme in insights.get("positive_themes", []) or ["—"]:
            st.markdown(f'<span class="tag" style="background:#e8f6ec;">{theme}</span>', unsafe_allow_html=True)
    with c2:
        st.markdown("#### ❌ Negative Themes")
        for theme in insights.get("negative_themes", []) or ["—"]:
            st.markdown(f'<span class="tag" style="background:#fdeaea;">{theme}</span>', unsafe_allow_html=True)
    with c3:
        st.markdown("#### 💡 Recommendations")
        for rec in insights.get("recommendations", []) or ["—"]:
            st.markdown(f"- {rec}")

    hashtags = Counter(
        tag.lower()
        for item in result["comments_analysis"]
        for tag in re.findall(r"#([A-Za-z0-9_]+)", str(item.get("comment", "")))
    )
    if hashtags:
        st.markdown("#### #️⃣ Trending Hashtags")
        st.markdown(
            " ".join(
                f'<span class="tag" style="background:#e8f0fd;">#{tag} ({count})</span>'
                for tag, count in hashtags.most_common(10)
            ),
            unsafe_allow_html=True,
        )


def _render_visualizations(result: dict[str, Any]) -> None:
    tab_pie, tab_bar, tab_wc, tab_timeline = st.tabs(
        ["Pie Chart", "Bar Chart", "Word Cloud", "Timeline"]
    )
    with tab_pie:
        st.plotly_chart(sentiment_pie(result), use_container_width=True)
    with tab_bar:
        st.plotly_chart(phrases_bar(result), use_container_width=True)
    with tab_wc:
        wordcloud_fig(result)
    with tab_timeline:
        st.plotly_chart(sentiment_timeline(result), use_container_width=True)


def _render_comments(result: dict[str, Any]) -> None:
    df = pd.DataFrame(result["comments_analysis"])
    if df.empty:
        st.info("No comments to display.")
        return

    filter_map = {
        "All": None,
        "Satisfied": "SATISFIED",
        "Unsatisfied": "UNSATISFIED",
        "Neutral": "NEUTRAL",
    }
    choice = st.radio("Filter:", list(filter_map.keys()), horizontal=True)
    filtered = df if filter_map[choice] is None else df[df["sentiment"] == filter_map[choice]]

    st.caption(f"Showing {len(filtered)} of {len(df)} comments")

    detail_view = st.toggle("Show details (reason & phrases)", value=False)
    if detail_view:
        for _, row in filtered.iterrows():
            emoji = (
                "✅" if row["sentiment"] == "SATISFIED"
                else "❌" if row["sentiment"] == "UNSATISFIED"
                else "➖"
            )
            with st.expander(f"{emoji} {row['user']} · {row['sentiment']} · {row['confidence']:.2f}"):
                st.write(row["comment"])
                st.caption(f"Reason: {row['reason']}")
                if row.get("key_phrases"):
                    st.caption("Phrases: " + ", ".join(str(p) for p in row["key_phrases"]))
    else:
        display = filtered[["user", "comment", "sentiment", "confidence", "timestamp"]].copy()
        display.columns = ["User", "Comment", "Sentiment", "Confidence", "Timestamp"]
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="%.2f"
                )
            },
        )


def _render_post_info(result: dict[str, Any]) -> None:
    info = result["post_info"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("❤️ Likes", info.get("likes", 0))
    c2.metric("💬 Comments", info.get("comments_count", 0))
    c3.metric("📅 Date", info.get("date", "—"))
    c4.metric("👤 Profile", info.get("profile", "—"))

    caption = info.get("caption")
    st.markdown("**Caption:**")
    st.write(caption or "*(no caption)*")
    st.markdown(
        f"[Open on Instagram]({info.get('url', '')}) "
        f"· Shortcode: `{info.get('shortcode', '')}` · Type: {info.get('media_type', '—')}"
    )


def _render_result(result: dict[str, Any]) -> None:
    st.markdown("---")
    tab_dash, tab_insights, tab_viz, tab_comments, tab_info = st.tabs(
        ["📈 Dashboard", "🎯 Key Insights", "📊 Visualizations", "📝 Comments", "ℹ️ Post Info"]
    )
    with tab_dash:
        _render_dashboard(result)
    with tab_insights:
        _render_insights(result)
    with tab_viz:
        _render_visualizations(result)
    with tab_comments:
        _render_comments(result)
    with tab_info:
        _render_post_info(result)


def _render_export_buttons(result: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "⬇️ CSV", export_csv_bytes(result), "analysis.csv", "text/csv", use_container_width=True
        )
    with c2:
        st.download_button(
            "⬇️ Excel",
            export_excel_bytes(result),
            "analysis.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "⬇️ PDF", export_pdf_bytes(result), "analysis.pdf", "application/pdf", use_container_width=True
        )
    with c4:
        st.download_button(
            "⬇️ JSON", export_json_bytes(result), "analysis.json", "application/json", use_container_width=True
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _find_session_files() -> list[str]:
    """Find Instaloader session files (``session-*``) in the working directory."""
    try:
        return sorted(glob.glob("session-*"))
    except Exception:  # noqa: BLE001
        return []


def _render_instagram_connect() -> tuple[str, str, str]:
    """Render the sidebar Instagram connection widget.

    Lets the user log straight into Instagram with username/password (2FA
    supported), which persists a session file automatically. Falls back to
    picking an existing saved session file or entering one manually.

    Returns:
        ``(username, password, session_file)`` to use for scraping.
    """
    if st.session_state.get("ig_connected"):
        session_path = st.session_state.get("ig_session_file", "")
        st.success(f"✅ Connected as @{st.session_state.get('ig_username', '')}")
        if session_path:
            st.caption(f"Session: `{session_path}`")
        if st.button("Disconnect", key="ig_disconnect", use_container_width=True):
            for key in (
                "ig_connected",
                "ig_username",
                "ig_session_file",
                "ig_pending_2fa",
                "ig_pending_user",
                "ig_pending_password",
                "ig_scraper",
            ):
                st.session_state.pop(key, None)
            st.rerun()
        return (
            st.session_state.get("ig_username", ""),
            "",
            session_path,
        )

    st.caption(
        "Optional — anonymous scraping is tried first. Connect only if "
        "Instagram blocks anonymous access or the post is private."
    )
    col_u, col_p = st.columns(2)
    username = col_u.text_input("Username", key="ig_username_input")
    password = col_p.text_input("Password", type="password", key="ig_password_input")

    if st.session_state.get("ig_pending_2fa"):
        st.info(
            "**Two-factor authentication required.** Instagram just sent a code. It "
            "arrives as a **WhatsApp** or **text (SMS)** message to the phone number "
            "on the account, a rotating code in your **authenticator app** (Google "
            "Authenticator / Authy), or an **approval prompt in the Instagram app**. "
            "If nothing arrives, tap **🔄 Resend code** below."
        )
        code = st.text_input("Two-factor authentication code", key="ig_2fa_code")
        c1, c2 = st.columns(2)
        verified = c1.button("✓ Verify code", key="ig_verify", disabled=not code.strip())
        resend_clicked = c2.button("🔄 Resend code", key="ig_resend")

        if resend_clicked:
            scraper = st.session_state.get("ig_scraper")
            pending_user = st.session_state.get("ig_pending_user", username)
            pending_password = st.session_state.get("ig_pending_password", "")
            try:
                _, msg = scraper.resend_two_factor_code(pending_user, pending_password)
                st.success(msg)
            except TwoFactorRequiredError as exc:
                st.success(str(exc))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to resend 2FA code")
                st.error(f"Could not resend the code: {exc}")
        elif verified:
            scraper = st.session_state.get("ig_scraper")
            pending_user = st.session_state.get("ig_pending_user", username)
            ok, msg = scraper.complete_two_factor(code.strip(), pending_user)
            if ok:
                for key in ("ig_pending_2fa", "ig_scraper", "ig_pending_user", "ig_pending_password"):
                    st.session_state.pop(key, None)
                st.session_state["ig_connected"] = True
                st.session_state["ig_username"] = pending_user
                st.session_state["ig_session_file"] = f"session-{pending_user}"
                st.rerun()
            else:
                st.error(f"{msg} The code may have expired — use **🔄 Resend code**.")
    else:
        if st.button(
            "🔗 Connect to Instagram",
            key="ig_connect",
            disabled=not (username.strip() and password),
            use_container_width=True,
        ):
            scraper = InstagramScraper(autologin=False)
            try:
                ok, msg = scraper.connect(username.strip(), password)
            except TwoFactorRequiredError:
                st.session_state["ig_pending_2fa"] = True
                st.session_state["ig_scraper"] = scraper
                st.session_state["ig_pending_user"] = username.strip()
                st.session_state["ig_pending_password"] = password
                st.rerun()
            if ok:
                st.session_state["ig_connected"] = True
                st.session_state["ig_username"] = username.strip()
                st.session_state["ig_session_file"] = f"session-{username.strip()}"
                st.rerun()
            else:
                st.error(msg)

    saved = _find_session_files()
    if saved:
        session_file = st.selectbox(
            "Saved session files (on disk)",
            [""] + saved,
            key="ig_saved_session",
            help="Select a previously saved Instaloader session to reuse it.",
        )
    else:
        session_file = st.text_input(
            "Session file path",
            value=os.getenv("INSTAGRAM_SESSION_FILE", ""),
            key="ig_session_input",
        )

    # Credentials typed here are only used after a successful Connect; the
    # scraper still picks up INSTAGRAM_USERNAME / INSTAGRAM_PASSWORD from .env.
    return "", "", session_file


def _render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("⚙️ Settings")

        model = st.selectbox(
            "Groq model",
            options=[
                DEFAULT_MODEL,
                "groq/compound",
                "openai/gpt-oss-20b",
                "qwen/qwen3.6-27b",
            ],
            help="Model used for sentiment classification and insights.",
        )
        max_comments = st.slider(
            "Max comments", min_value=10, max_value=200, value=100, step=10
        )
        use_agent = st.toggle(
            "Use LangChain agent",
            value=False,
            help="Orchestrate with the LangChain agent instead of the deterministic pipeline.",
        )

        st.markdown("### 🔗 Instagram connection")
        username, password, session_file = _render_instagram_connect()

        if st.button("🗑️ Clear scrape cache", use_container_width=True):
            scrape_cached.clear()
            st.success("Cache cleared")

        st.markdown("---")
        st.subheader("🗂️ History")
        store = HistoryStore()
        history = store.list(limit=10)
        if not history:
            st.caption("No past analyses yet.")
        else:
            for item in history:
                label = f"{item['post_url'][:42]}… · {item['satisfaction_rate']}%"
                with st.container():
                    b1, b2 = st.columns([4, 1])
                    with b1:
                        if st.button(label, key=f"load_{item['id']}", use_container_width=True):
                            st.session_state["result"] = store.get(item["id"])
                            st.rerun()
                    with b2:
                        if st.button("🗑", key=f"del_{item['id']}"):
                            store.delete(item["id"])
                            st.rerun()

    return {
        "model": model,
        "max_comments": max_comments,
        "use_agent": use_agent,
        "username": username,
        "password": password,
        "session_file": session_file,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _analyze(
    url: str,
    settings: dict[str, Any],
    store: HistoryStore,
    progress_bar: Any,
    status: Any,
) -> dict[str, Any] | None:
    shortcode = extract_shortcode(url)
    if not shortcode:
        st.error(
            "Invalid Instagram URL. Use a link like "
            "https://www.instagram.com/p/XXXXX/ (supports /p/, /reel/ and /tv/)."
        )
        return None

    total_comments = settings["max_comments"]
    processed = {"n": 0}

    def progress_cb(done: int, total: int) -> None:
        status.write(f"Analyzing sentiment… {done}/{total} comments")
        progress_bar.progress(int(done / max(total, 1) * 100))

    try:
        with st.spinner("Analyzing post…"):
            result = _run_pipeline(
                url,
                total_comments,
                settings["model"],
                settings["use_agent"],
                settings["username"],
                settings["password"],
                settings["session_file"],
                progress_cb=progress_cb,
            )
        processed["n"] = result.get("sentiment_summary", {}).get("total_analyzed", 0)
    except LoginRequiredError as exc:
        logger.warning("Login required for %s: %s", url, exc)
        progress_bar.empty()
        status.empty()
        st.error(str(exc))
        with st.expander("🔑 How to fix login-required posts"):
            st.markdown(
                """
1. Open **Settings → Instagram credentials** in the sidebar.
2. Either enter your **username + password**, or generate a session file:
   ```bash
   python create_session.py --username YOUR_USERNAME
   ```
   and paste the resulting file path (e.g. `session-YOUR_USERNAME`) into the
   **Session file path** box.
3. Click **🚀 Analyze** again. The cached anonymous failure is skipped
   automatically when credentials change.
"""
            )
        return None
    except RateLimitError as exc:
        logger.warning("Rate limited on %s: %s", url, exc)
        progress_bar.empty()
        status.empty()
        st.error(str(exc))
        return None
    except Exception as exc:
        logger.exception("Analysis failed for %s", url)
        progress_bar.empty()
        st.error(f"Analysis failed: {exc}")
        return None

    progress_bar.progress(100)
    status.write(f"Done. Analyzed {processed['n']} comments.")
    store.save(result)
    return result


def _compare_posts(settings: dict[str, Any], store: HistoryStore) -> None:
    st.markdown("### 🔀 Compare Multiple Posts")
    raw = st.text_area(
        "Enter Instagram post URLs (one per line)",
        height=120,
        key="compare_input",
        placeholder="https://www.instagram.com/p/AAAA/\nhttps://www.instagram.com/reel/BBBB/",
    )
    urls = [line.strip() for line in raw.splitlines() if line.strip()]
    if st.button("Analyze all posts", key="compare_button", disabled=not urls):
        progress = st.progress(0.0)
        status = st.empty()
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for idx, url in enumerate(urls):
            status.write(f"[{idx + 1}/{len(urls)}] Analyzing {url}")
            try:
                result = _run_pipeline(
                    url,
                    settings["max_comments"],
                    settings["model"],
                    settings["use_agent"],
                    settings["username"],
                    settings["password"],
                    settings["session_file"],
                )
                results.append(result)
                store.save(result)
            except Exception as exc:
                logger.exception("Comparison analysis failed for %s", url)
                errors.append(f"{url}: {exc}")
            progress.progress(int((idx + 1) / len(urls) * 100))
        progress.empty()

        if errors:
            for err in errors:
                st.error(err)

        if results:
            rows = []
            for r in results:
                summary = r["sentiment_summary"]
                rows.append(
                    {
                        "Post": extract_shortcode(r["post_info"].get("url", "")) or r["post_info"].get("url", ""),
                        "Satisfaction %": summary["satisfaction_rate"],
                        "Analyzed": summary["total_analyzed"],
                        "Satisfied": summary["satisfied"],
                        "Unsatisfied": summary["unsatisfied"],
                        "Neutral": summary["neutral"],
                    }
                )
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            fig = px.bar(
                df,
                x="Post",
                y="Satisfaction %",
                title="Satisfaction Rate by Post",
                color="Satisfaction %",
                color_continuous_scale="RdYlGn",
                text_auto=".1f",
            )
            st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("📊 Instagram Sentiment Analyzer")
    st.caption("Powered by LangChain · Groq · Instaloader — analyze public post comments in seconds")

    settings = _render_sidebar()
    store = HistoryStore()

    with st.expander("🔀 Compare multiple posts", expanded=False):
        _compare_posts(settings, store)

    st.markdown("### 🔍 Analyze a single post")
    url = st.text_input(
        "Instagram post URL",
        placeholder="https://www.instagram.com/p/XXXXX/",
        key="post_url",
    )
    analyze_clicked = st.button("🚀 Analyze", type="primary", disabled=not url.strip())

    if analyze_clicked:
        progress_bar = st.progress(0)
        status = st.empty()
        result = _analyze(url.strip(), settings, store, progress_bar, status)
        if result:
            st.session_state["result"] = result
            st.session_state["last_url"] = url.strip()

    if st.session_state.get("result"):
        result = st.session_state["result"]
        _render_export_buttons(result)
        _render_result(result)


if __name__ == "__main__":
    main()
