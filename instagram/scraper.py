import requests
from bs4 import BeautifulSoup
import re
import time

class InstagramScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_comments(self, post_url):
        """Fetch comments from Instagram post (simplified version)"""
        # Note: Instagram API requires authentication
        # This is a simplified example using public endpoints
        # In production, use Instagram Graph API or Instagram Basic Display API
        
        try:
            # Extract shortcode from URL
            shortcode = self.extract_shortcode(post_url)
            if not shortcode:
                return []
            
            # For demo purposes, return mock comments
            # In production, you would use Instagram's official API
            mock_comments = self.get_mock_comments()
            return mock_comments
            
        except Exception as e:
            print(f"Error fetching comments: {e}")
            return []
    
    def extract_shortcode(self, url):
        """Extract post shortcode from Instagram URL"""
        patterns = [
            r'instagram\.com/p/([A-Za-z0-9_-]+)',
            r'instagram\.com/reel/([A-Za-z0-9_-]+)',
            r'instagram\.com/tv/([A-Za-z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_mock_comments(self):
        """Return mock comments for demonstration"""
        return [
            {'text': 'This product is amazing! Love it so much ❤️', 'timestamp': '2024-01-01'},
            {'text': 'Best purchase ever made! #happycustomer', 'timestamp': '2024-01-02'},
            {'text': 'Not worth the money, disappointed 😞', 'timestamp': '2024-01-03'},
            {'text': 'Good quality, fast shipping @brand', 'timestamp': '2024-01-04'},
            {'text': 'Average product, nothing special', 'timestamp': '2024-01-05'},
            {'text': 'Excellent customer service! Will buy again', 'timestamp': '2024-01-06'},
            {'text': 'Terrible experience, product broke after 2 days', 'timestamp': '2024-01-07'},
            {'text': 'Pretty good for the price 👍', 'timestamp': '2024-01-08'},
            {'text': 'Highly recommend to everyone!', 'timestamp': '2024-01-09'},
            {'text': 'Shipping took forever, but product is decent', 'timestamp': '2024-01-10'},
        ]