import re

def validate_instagram_url(url):
    """Validate Instagram post URL"""
    patterns = [
        r'^https?://(www\.)?instagram\.com/p/[A-Za-z0-9_-]+/?',
        r'^https?://(www\.)?instagram\.com/reel/[A-Za-z0-9_-]+/?',
        r'^https?://(www\.)?instagram\.com/tv/[A-Za-z0-9_-]+/?'
    ]
    
    for pattern in patterns:
        if re.match(pattern, url):
            return True
    return False