from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import uuid
import json
from datetime import datetime

from models.sentiment_model import SentimentAnalyzer
from preprocessing.comment_cleaner import CommentCleaner
from instagram.scraper import InstagramScraper
from utils.validators import validate_instagram_url
from database.db_manager import DatabaseManager

sentiment_analyzer = SentimentAnalyzer()
comment_cleaner = CommentCleaner()
instagram_scraper = InstagramScraper()
db_manager = DatabaseManager()

class AnalyzePostView(APIView):
    def post(self, request):
        try:
            data = request.data
            post_url = data.get('post_url')
            
            if not post_url:
                return Response({'error': 'Post URL is required'}, status=status.HTTP_400_BAD_REQUEST)
            
            if not validate_instagram_url(post_url):
                return Response({'error': 'Invalid Instagram URL'}, status=status.HTTP_400_BAD_REQUEST)
            
            comments = instagram_scraper.fetch_comments(post_url)
            
            if not comments:
                return Response({'error': 'No comments found'}, status=status.HTTP_404_NOT_FOUND)
            
            results = []
            for comment in comments:
                cleaned_text = comment_cleaner.clean(comment['text'])
                sentiment = sentiment_analyzer.analyze(cleaned_text)
                results.append({
                    'text': comment['text'],
                    'cleaned_text': cleaned_text,
                    'sentiment': sentiment['label'],
                    'confidence': sentiment['confidence']
                })
            
            total_comments = len(results)
            sentiment_counts = {
                'positive': sum(1 for r in results if r['sentiment'] == 'positive'),
                'neutral': sum(1 for r in results if r['sentiment'] == 'neutral'),
                'negative': sum(1 for r in results if r['sentiment'] == 'negative')
            }
            
            satisfaction_score = (sentiment_counts['positive'] / total_comments) * 100 if total_comments > 0 else 0
            
            analysis_id = str(uuid.uuid4())
            response = {
                'analysis_id': analysis_id,
                'post_url': post_url,
                'total_comments': total_comments,
                'satisfaction_score': round(satisfaction_score, 2),
                'sentiment_breakdown': {
                    'positive': round((sentiment_counts['positive'] / total_comments) * 100, 2),
                    'neutral': round((sentiment_counts['neutral'] / total_comments) * 100, 2),
                    'negative': round((sentiment_counts['negative'] / total_comments) * 100, 2)
                },
                'comments': results,
                'analyzed_at': datetime.now().isoformat()
            }
            
            db_manager.save_analysis(response)
            
            return Response(response, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class HistoryView(APIView):
    def get(self, request):
        try:
            history = db_manager.get_all_analyses()
            return Response(history, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AnalysisDetailView(APIView):
    def get(self, request, analysis_id):
        try:
            analysis = db_manager.get_analysis(analysis_id)
            if analysis:
                return Response(analysis, status=status.HTTP_200_OK)
            return Response({'error': 'Analysis not found'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class HealthCheckView(APIView):
    def get(self, request):
        return Response({'status': 'healthy'}, status=status.HTTP_200_OK)
