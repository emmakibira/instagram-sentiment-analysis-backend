import sqlite3
import json
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='sentiment_analysis.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT UNIQUE,
                post_url TEXT,
                total_comments INTEGER,
                satisfaction_score REAL,
                sentiment_breakdown TEXT,
                analyzed_at TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT,
                comment_text TEXT,
                sentiment TEXT,
                confidence REAL,
                FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, analysis_data):
        """Save analysis results to database"""
        import uuid
        analysis_id = analysis_data.get('analysis_id', str(uuid.uuid4()))
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO analyses 
            (analysis_id, post_url, total_comments, satisfaction_score, sentiment_breakdown, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            analysis_id,
            analysis_data['post_url'],
            analysis_data['total_comments'],
            analysis_data['satisfaction_score'],
            json.dumps(analysis_data['sentiment_breakdown']),
            analysis_data['analyzed_at']
        ))
        
        # Save individual comments
        for comment in analysis_data['comments']:
            cursor.execute('''
                INSERT INTO comments (analysis_id, comment_text, sentiment, confidence)
                VALUES (?, ?, ?, ?)
            ''', (
                analysis_id,
                comment['text'],
                comment['sentiment'],
                comment['confidence']
            ))
        
        conn.commit()
        conn.close()
        
        return analysis_id
    
    def get_all_analyses(self):
        """Retrieve all analyses"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT analysis_id, post_url, total_comments, satisfaction_score, sentiment_breakdown, analyzed_at FROM analyses ORDER BY analyzed_at DESC')
        rows = cursor.fetchall()
        
        analyses = []
        for row in rows:
            analyses.append({
                'analysis_id': row[0],
                'post_url': row[1],
                'total_comments': row[2],
                'satisfaction_score': row[3],
                'sentiment_breakdown': json.loads(row[4]),
                'analyzed_at': row[5]
            })
        
        conn.close()
        return analyses
    
    def get_analysis(self, analysis_id):
        """Retrieve specific analysis with comments"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM analyses WHERE analysis_id = ?', (analysis_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        cursor.execute('SELECT comment_text, sentiment, confidence FROM comments WHERE analysis_id = ?', (analysis_id,))
        comments = [{'text': c[0], 'sentiment': c[1], 'confidence': c[2]} for c in cursor.fetchall()]
        
        analysis = {
            'analysis_id': row[1],
            'post_url': row[2],
            'total_comments': row[3],
            'satisfaction_score': row[4],
            'sentiment_breakdown': json.loads(row[5]),
            'analyzed_at': row[6],
            'comments': comments
        }
        
        conn.close()
        return analysis