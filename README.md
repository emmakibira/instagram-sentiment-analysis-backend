# Instagram Sentiment Analysis System

A professional-grade sentiment analysis application that analyzes Instagram post comments and provides customer satisfaction insights using AI/ML.

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Backend Setup](#backend-setup)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Frontend Setup](#frontend-setup)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Features

### Backend
- **Sentiment Classification**: Analyzes comments as positive, neutral, or negative using state-of-the-art transformer models
- **Comment Preprocessing**: Automatically removes emojis, hashtags, mentions, URLs, and special characters
- **Instagram Integration**: Fetches comments from Instagram posts (via API or mock data for demo)
- **REST API**: Fully documented endpoints for analysis and history retrieval
- **Database Storage**: SQLite database to store analysis results and comments
- **Scalable Architecture**: Built with Flask and supports production deployment with Gunicorn

### Frontend (React Native + TypeScript)
- **Mobile Interface**: Beautiful, responsive UI for iOS and Android
- **Dashboard**: Real-time sentiment visualization with charts
- **History Management**: View past analyses with filtering
- **TypeScript Support**: Fully typed components and API responses
- **Error Handling**: Comprehensive error messages and user guidance

## Tech Stack

### Backend
- **Framework**: Flask 3.0.0 with CORS support
- **ML/AI**: 
  - Transformers (DistilBERT model)
  - TensorFlow 2.13.0
  - PyTorch 2.1.0
  - Scikit-learn 1.3.0
- **Database**: SQLite3
- **Server**: Gunicorn (production)
- **Other Libraries**: 
  - Requests, BeautifulSoup4 (web scraping)
  - Emoji, Regex (text processing)
  - Python-dotenv (environment variables)

### Frontend
- **Framework**: React Native with TypeScript
- **Navigation**: React Navigation
- **Charts**: React Native Chart Kit
- **HTTP Client**: Axios
- **State Management**: Redux Toolkit
- **UI Components**: React Native Paper or Native Base

## Installation

### Prerequisites
- Python 3.10+
- Node.js 16+ (for frontend)
- npm or yarn (for frontend)
- Git

### Backend Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd instagram-sentiment-analysis/backend
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
# Or with Poetry
poetry install
```

4. **Download ML Models** (first run):
```bash
python
>>> from transformers import pipeline
>>> pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
>>> exit()
```

5. **Run the development server**:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory**:
```bash
cd instagram-sentiment-analysis/frontend
```

2. **Install dependencies**:
```bash
npm install
# or
yarn install
```

3. **Configure API endpoint** in `.env`:
```
REACT_APP_API_URL=http://localhost:5000/api
```

4. **Run development server**:
```bash
npm start
# For iOS
npm run ios
# For Android
npm run android
```

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```bash
# Server Configuration
FLASK_ENV=development
FLASK_DEBUG=True
HOST=0.0.0.0
PORT=5000

# Database
DATABASE_PATH=sentiment_analysis.db

# Instagram API (optional)
INSTAGRAM_API_TOKEN=your_token_here

# Model Configuration
MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
CONFIDENCE_THRESHOLD=0.6
```

See `.env.example` for all available options.

## API Endpoints

### 1. Analyze Instagram Post
**POST** `/api/analyze`

Request:
```json
{
  "post_url": "https://www.instagram.com/p/ABC123DEF456/"
}
```

Response:
```json
{
  "analysis_id": "uuid",
  "post_url": "https://www.instagram.com/p/ABC123DEF456/",
  "total_comments": 100,
  "satisfaction_score": 75.5,
  "sentiment_breakdown": {
    "positive": 75.5,
    "neutral": 15.3,
    "negative": 9.2
  },
  "comments": [
    {
      "text": "Amazing product!",
      "cleaned_text": "amazing product",
      "sentiment": "positive",
      "confidence": 0.99
    }
  ],
  "analyzed_at": "2024-04-21T10:30:00"
}
```

### 2. Get Analysis History
**GET** `/api/history`

Response:
```json
[
  {
    "analysis_id": "uuid",
    "post_url": "...",
    "total_comments": 100,
    "satisfaction_score": 75.5,
    "sentiment_breakdown": {...},
    "analyzed_at": "2024-04-21T10:30:00"
  }
]
```

### 3. Get Specific Analysis
**GET** `/api/history/<analysis_id>`

Response: Same as analyze endpoint

### 4. Health Check
**GET** `/api/health`

Response:
```json
{
  "status": "healthy"
}
```

## Usage Examples

### Using cURL

```bash
# Analyze a post
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"post_url": "https://www.instagram.com/p/ABC123DEF456/"}'

# Get history
curl http://localhost:5000/api/history

# Get specific analysis
curl http://localhost:5000/api/history/uuid-here
```

### Using Python

```python
import requests

api_url = "http://localhost:5000/api"

# Analyze post
response = requests.post(
    f"{api_url}/analyze",
    json={"post_url": "https://www.instagram.com/p/ABC123DEF456/"}
)
result = response.json()
print(f"Satisfaction Score: {result['satisfaction_score']}%")

# Get history
history = requests.get(f"{api_url}/history").json()
print(f"Total analyses: {len(history)}")
```

### Using JavaScript/TypeScript

```typescript
const API_URL = "http://localhost:5000/api";

// Analyze post
async function analyzePost(postUrl: string) {
  const response = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ post_url: postUrl })
  });
  return response.json();
}

// Get history
async function getHistory() {
  const response = await fetch(`${API_URL}/history`);
  return response.json();
}
```

## Project Structure

```
instagram-sentiment-analysis/
├── backend/
│   ├── app.py                    # Flask application & API endpoints
│   ├── requirements.txt           # Python dependencies
│   ├── pyproject.toml             # Poetry configuration
│   ├── README.md                  # Documentation
│   ├── .env.example               # Environment variables template
│   ├── .gitignore                 # Git ignore rules
│   ├── sentiment_analysis.db      # SQLite database (auto-created)
│   ├── models/
│   │   └── sentiment_model.py     # Sentiment analysis model
│   ├── database/
│   │   └── db_manager.py          # Database operations
│   ├── instagram/
│   │   └── scraper.py             # Instagram comment fetching
│   ├── preprocessing/
│   │   └── comment_cleaner.py     # Text preprocessing
│   └── utils/
│       └── validators.py          # URL validation
├── frontend/
│   ├── src/
│   │   ├── components/            # React Native components (.tsx)
│   │   ├── screens/               # App screens
│   │   ├── services/              # API services
│   │   ├── redux/                 # State management
│   │   ├── types/                 # TypeScript types
│   │   └── App.tsx                # Main app component
│   ├── package.json               # Frontend dependencies
│   └── tsconfig.json              # TypeScript config
├── docker-compose.yml             # Docker configuration
└── README.md                       # Main documentation
```

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t instagram-sentiment-analysis .
docker run -p 5000:5000 instagram-sentiment-analysis
```

### Deployment Platforms
- **Backend**: Heroku, Railway, Render, AWS ECS, DigitalOcean
- **Frontend**: Expo, Google Play Store, Apple App Store
- **Database**: Cloud-based PostgreSQL for scalability

## API Rate Limiting

Implement rate limiting in production:

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(app, key_func=get_remote_address)

@app.route('/api/analyze', methods=['POST'])
@limiter.limit("30 per hour")
def analyze_post():
    # endpoint code
```

## Testing

### Backend Tests

```bash
pytest tests/

# With coverage
pytest --cov=. tests/
```

### Frontend Tests

```bash
npm test
npm run test:e2e
```

## Performance Optimization

1. **Model Caching**: Sentiment model is loaded once and reused
2. **Batch Processing**: Process multiple comments efficiently
3. **Database Indexing**: Indexed analysis_id for fast lookups
4. **Async Operations**: Use async/await for non-blocking operations
5. **Frontend Optimization**: 
   - Code splitting
   - Lazy loading screens
   - Memoization of expensive components

## Security Considerations

1. **HTTPS**: Always use HTTPS in production
2. **CORS**: Configured to allow frontend origin
3. **Input Validation**: All URLs and inputs validated
4. **Error Messages**: Don't expose sensitive information
5. **Environment Variables**: Sensitive data in .env files
6. **Authentication**: Add JWT authentication for protected endpoints
7. **Rate Limiting**: Prevent abuse with rate limits

## Troubleshooting

### Issue: Model download fails
**Solution**: Download manually:
```bash
python -c "from transformers import pipeline; pipeline('sentiment-analysis')"
```

### Issue: Database locked error
**Solution**: Close all connections and restart the server

### Issue: CORS errors on frontend
**Solution**: Ensure flask-cors is installed and CORS(app) is called in app.py

### Issue: Long analysis times
**Solution**: Consider using a lighter model or GPU acceleration with CUDA

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Real Instagram API integration
- [ ] Multi-language sentiment analysis
- [ ] Advanced visualization dashboard
- [ ] Export analysis reports (PDF, CSV)
- [ ] Email notifications
- [ ] Custom model training
- [ ] Real-time comment monitoring
- [ ] Competitor analysis
- [ ] Sentiment trend over time

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: support@example.com
- Discord: [Community Server](https://discord.gg/example)

---

**Last Updated**: April 2024
**Version**: 0.1.0
