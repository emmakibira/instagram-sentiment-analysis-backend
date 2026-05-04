#!/usr/bin/env python
"""
Test script to verify all backend components work correctly
"""

import sys
import json

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    try:
        import flask
        print("✓ Flask")
        import flask_cors
        print("✓ Flask-CORS")
        from transformers import pipeline
        print("✓ Transformers")
        import torch
        print("✓ PyTorch")
        import emoji
        print("✓ Emoji")
        import requests
        print("✓ Requests")
        import sqlite3
        print("✓ SQLite3")
        print("\n✓ All imports successful!\n")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_components():
    """Test core components"""
    print("Testing core components...")
    
    try:
        # Test validators
        from utils.validators import validate_instagram_url
        assert validate_instagram_url("https://www.instagram.com/p/ABC123DEF456/"), "Valid URL test failed"
        assert not validate_instagram_url("https://google.com"), "Invalid URL test failed"
        print("✓ URL Validator")
        
        # Test comment cleaner
        from preprocessing.comment_cleaner import CommentCleaner
        cleaner = CommentCleaner()
        cleaned = cleaner.clean("Amazing product! @brand #love ❤️ https://example.com")
        assert len(cleaned) > 0, "Comment cleaner test failed"
        print(f"✓ Comment Cleaner (example: '{cleaned}')")
        
        # Test scraper
        from instagram.scraper import InstagramScraper
        scraper = InstagramScraper()
        shortcode = scraper.extract_shortcode("https://www.instagram.com/p/ABC123DEF456/")
        assert shortcode == "ABC123DEF456", "Shortcode extraction test failed"
        print("✓ Instagram Scraper")
        
        # Test database
        from database.db_manager import DatabaseManager
        db = DatabaseManager(":memory:")  # Use in-memory database for testing
        print("✓ Database Manager")
        
        # Test sentiment analyzer
        print("Loading sentiment model (this may take a moment)...")
        from models.sentiment_model import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("This product is amazing!")
        assert "label" in result, "Sentiment analyzer test failed"
        print(f"✓ Sentiment Analyzer (test: 'This product is amazing!' → {result['label']})")
        
        print("\n✓ All component tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sentiment_analysis():
    """Test sentiment analysis with various inputs"""
    print("Testing sentiment analysis...")
    try:
        from models.sentiment_model import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        
        test_cases = [
            "This is amazing! Love it!",
            "It's okay, nothing special",
            "Terrible product, very disappointed",
            "",  # Empty comment
        ]
        
        for text in test_cases:
            result = analyzer.analyze(text)
            display_text = text if text else "(empty)"
            print(f"  '{display_text}' → {result['label']} ({result['confidence']:.2f})")
        
        print("\n✓ Sentiment analysis tests passed!\n")
        return True
    except Exception as e:
        print(f"✗ Sentiment analysis test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Instagram Sentiment Analysis Backend - Test Suite")
    print("=" * 60 + "\n")
    
    results = {
        "imports": test_imports(),
        "components": test_components(),
        "sentiment_analysis": test_sentiment_analysis(),
    }
    
    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! Backend is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
