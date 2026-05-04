import re
import emoji

class CommentCleaner:
    def __init__(self):
        # Patterns for cleaning
        self.mention_pattern = r'@\w+'
        self.hashtag_pattern = r'#\w+'
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+'
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
    def clean(self, text):
        """Clean Instagram comment text"""
        if not text:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')
        
        # Remove mentions
        text = re.sub(self.mention_pattern, '', text)
        
        # Remove hashtags (keep the word without #)
        text = re.sub(self.hashtag_pattern, '', text)
        
        # Remove URLs
        text = re.sub(self.url_pattern, '', text)
        
        # Remove emails
        text = re.sub(self.email_pattern, '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep letters, numbers, basic punctuation)
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        return text