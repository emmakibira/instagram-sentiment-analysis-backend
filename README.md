# Instagram Sentiment Analyzer

Analyze the sentiment of Instagram post comments in seconds. Paste a post
URL, and the app scrapes the post with **Instaloader**, classifies every
comment as **SATISFIED / UNSATISFIED / NEUTRAL** using a **Groq** LLM routed
through a **LangChain** agent, and renders an interactive **Streamlit**
dashboard with charts, insights and export options.

> Powered by LangChain · Groq · Instaloader · Streamlit · Plotly

---

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Creating an Instagram session](#creating-an-instagram-session)
- [Usage](#usage)
- [Output format](#output-format)
- [How it works](#how-it-works)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

- **Single input** — paste an Instagram post URL (supports `/p/`, `/reel/`,
  `/tv/`), click **Analyze**.
- **Real scraping** — Instaloader pulls the caption, like/comment counts,
  post date and the latest comments (10–200, default 100) without needing an
  Instagram API token.
- **LangChain agent** — three tools (`scrape_instagram_post`,
  `analyze_comments`, `generate_insights`) orchestrated either by a
  `create_agent` graph or a deterministic pipeline (toggle in the sidebar).
- **Sentiment classification** — per-comment label + confidence + reason +
  key phrases, using a Groq chat model with a strict JSON schema.
- **Aggregated stats** — satisfaction %, satisfied/unsatisfied/neutral
  counts, average confidence.
- **Insights** — positive & negative themes, actionable recommendations,
  trending hashtags and top commenters.
- **Visualizations** — Plotly pie chart, key-phrase bar chart, word cloud,
  confidence gauge and a sentiment-over-time timeline.
- **Multiple-post comparison** — analyze several URLs at once.
- **Exports** — download results as CSV, Excel, PDF or JSON.
- **History** — past analyses are stored in SQLite and reloadable from the
  sidebar.
- **Caching** — scraped posts are cached (30 min) to avoid re-hitting
  Instagram; the LLM is reused across runs.
- **Robust error handling** — friendly messages for invalid URLs, private
  posts, rate limits and network failures, with retry logic for flaky calls.

## Demo

```
+------------------------------------------+
|  📊 Instagram Sentiment Analyzer          |
|  Powered by LangChain                     |
+------------------------------------------+
|  [https://www.instagram.com/p/XXXXX/] 🚀 |
+------------------------------------------+
|  🎯 Satisfaction: 78%   ✅ 35  ❌ 10    |
|  ➖ 5   💬 50   🧠 conf 0.85            |
+------------------------------------------+
|  📈 Dashboard | 🎯 Insights | 📊 Charts  |
|  📝 Comments  | ℹ️ Post Info             |
+------------------------------------------+
```

## Project Structure

```
instagram-sentiment-analysis-backend/
├── app.py                     # Streamlit UI
├── agent.py                   # LangChain agent + deterministic pipeline
├── analyzer.py                # Sentiment classification & insights (Groq)
├── create_session.py          # Generate an Instagram session file
├── requirements.txt           # Python dependencies
├── .env                       # Secrets (GROQ_API_KEY, …)
├── .env.example               # Template for environment variables
├── instagram/
│   └── scraper.py             # Instaloader-based scraping (anonymous first)
├── preprocessing/
│   └── comment_cleaner.py     # Text cleaning (emojis, mentions, …)
├── database/
│   ├── history_store.py       # SQLite history for analyses
│   └── db_manager.py          # (legacy backend module)
├── utils/
│   ├── helpers.py             # URLs, JSON parsing, retries, exports
│   └── validators.py          # URL validation
├── api/ config/ manage.py     # (legacy Django scaffold — unused by the app)
└── tests/                     # pytest suite
```

## Installation

Requires **Python 3.10+**.

```bash
git clone <repository-url>
cd instagram-sentiment-analysis-backend

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Or, if you use [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your Groq API key (get one at
<https://console.groq.com>):

```bash
cp .env.example .env
```

```dotenv
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional: model override (must support the OpenAI-compatible chat endpoint)
GROQ_MODEL=openai/gpt-oss-120b

# Optional: Instagram login (recommended — see below)
INSTAGRAM_USERNAME=
INSTAGRAM_PASSWORD=
INSTAGRAM_SESSION_FILE=

# History database location (optional)
HISTORY_DB_PATH=analysis_history.db
```

### Does it need an Instagram login?

No. The app **always tries anonymous scraping first** — on most home/office
networks it works with zero setup. Login is only required when Instagram
blocks anonymous access (common on datacenter/cloud IPs, and for private
posts). When that happens you'll get a clear message in the app telling you
how to add credentials.

### Creating an Instagram session

The easiest way is right inside the app: in the sidebar under **🔗 Instagram
connection**, enter your username and password and click **Connect to
Instagram**. The app logs in (two-factor authentication is supported — it
prompts for the code), saves a session file automatically, and uses it from
then on. You can disconnect anytime.

Or, from the command line:

```bash
python create_session.py --username your_username
```

It will prompt for your password (two-factor authentication is supported),
then write a `session-<username>` file in the current directory.

Then either:

```dotenv
# .env
INSTAGRAM_USERNAME=your_username
INSTAGRAM_SESSION_FILE=session-your_username
```

or pick the saved session from the sidebar's **Saved session files** dropdown.

## Usage

```bash
streamlit run app.py
```

Open the printed URL (default `http://localhost:8501`).

1. Paste an Instagram post URL, e.g. `https://www.instagram.com/p/XXXXX/`.
2. Click **🚀 Analyze**. Watch the progress bar as comments are classified.
3. Explore the tabs:
   - **📈 Dashboard** — headline metrics, sentiment pie, confidence gauge,
     top commenters.
   - **🎯 Key Insights** — positive/negative themes, recommendations,
     hashtags.
   - **📊 Visualizations** — pie, phrase bar chart, word cloud, timeline.
   - **📝 Comments** — filter by sentiment; switch to *details* to see each
     classifier's reasoning.
   - **ℹ️ Post Info** — caption, likes, comment count, date, profile.
4. Download the report from the export bar (CSV / Excel / PDF / JSON).
5. Use the sidebar to pick a different model, change the comment limit,
   toggle the **LangChain agent**, or reload a past analysis from history.

### Agent vs pipeline mode

| Mode | Description |
| ---- | ----------- |
| **Deterministic pipeline** (default) | Scrape → classify → summarize → insights. Fast, predictable, recommended. |
| **LangChain agent** | The same three tools are driven by an LLM agent loop (`create_agent`). Falls back to the pipeline if the agent misbehaves. |

## Output format

`analyze_post()` (in `agent.py`) returns:

```json
{
    "post_info": {
        "url": "https://www.instagram.com/p/XXXXX",
        "shortcode": "XXXXX",
        "caption": "Sample caption...",
        "likes": 1000,
        "comments_count": 150,
        "date": "2024-01-01 12:00:00",
        "profile": "username",
        "media_type": 1
    },
    "sentiment_summary": {
        "total_analyzed": 50,
        "satisfied": 35,
        "unsatisfied": 10,
        "neutral": 5,
        "satisfaction_rate": 70.0,
        "average_confidence": 0.85
    },
    "key_insights": {
        "positive_themes": ["quality", "service"],
        "negative_themes": ["price", "shipping delay"],
        "recommendations": ["Improve packaging", "Offer faster shipping"],
        "summary": "Customers love the quality but complain about shipping."
    },
    "comments_analysis": [
        {
            "user": "username",
            "comment": "Great product!",
            "timestamp": "2024-01-01 10:00:00",
            "sentiment": "SATISFIED",
            "confidence": 0.95,
            "reason": "Positive language used",
            "key_phrases": ["great", "product"]
        }
    ]
}
```

## How it works

```
URL ──► InstagramScraper (Instaloader)
          └─► post_info + comments (latest N)
                  └─► analyzer.analyze_comments()   (parallel batches, Groq)
                          └─► per-comment {sentiment, confidence, reason, phrases}
                                  └─► compute_summary() + generate_insights()
                                          └─► Streamlit dashboard + exports
```

- **Batch classification** — comments are processed in parallel batches
  (default 10 at a time, 4 workers) with a progress callback.
- **Rate limiting** — Instaloader's internal rate controller is kept (scaled
  by `SLEEP_RATIO`); `TooManyRequestsException` is surfaced with guidance.
- **Retries** — network calls and LLM invocations retry with backoff.
- **JSON safety** — the LLM is prompted for strict JSON; output is parsed
  robustly (fenced/embedded) and validated before use.

## Testing

```bash
python -m pytest tests/ -q
```

The suite (52 tests) covers URL extraction, JSON parsing, sentiment
normalization, classification (with a fake LLM — no API calls), summary
statistics, exports, scraper error mapping and the end-to-end pipeline.

## Troubleshooting

| Symptom | Cause / Fix |
| ------- | ----------- |
| `Instagram refused anonymous access` | Instagram blocks some networks (especially cloud IPs). Run `python create_session.py --username YOUR_USERNAME` and set the session path in the sidebar, or provide username/password (see [Creating an Instagram session](#creating-an-instagram-session)). |
| `Instagram is rate limiting requests` | You hit Instagram's throttle. Wait a few minutes, lower the comment limit, and retry. |
| `Invalid Instagram URL` | Only `/p/`, `/reel/` and `/tv/` links are supported. |
| `GROQ_API_KEY is not set` | Add your key to `.env` and restart Streamlit. |
| `Analysis failed: No comments…` | The post has no comments, or comments are restricted. |
| Model returns garbage JSON | Switch to `openai/gpt-oss-120b` in the sidebar (default). Some Groq models do not support tool calling. |
| Private / restricted post | Requires a logged-in session that follows the account. |

## Roadmap

- [ ] Live streaming of comments for active campaigns
- [ ] Competitor benchmark comparisons
- [ ] Response-template generator for common complaints
- [ ] Multi-language sentiment support
- [ ] Docker deployment recipe

## License

MIT — see [LICENSE](LICENSE).
