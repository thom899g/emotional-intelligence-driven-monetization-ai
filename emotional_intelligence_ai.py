from textblob import TextBlob
from typing import Optional, Dict
import logging

class SentimentAnalyzer:
    def __init__(self):
        self.model = TextBlob()
        logging.basicConfig(level=logging.INFO)
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of given text and return emotion scores."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Use basic emotion detection for now
            emotion = self._detect_emotion(polarity, subjectivity)
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'emotion': emotion
            }
        except Exception as e:
            logging.error(f"Sentiment analysis failed: {str(e)}")
            # Fallback to basic positive/negative classification
            if polarity > 0:
                return {'emotion': 'positive'}
            else:
                return {'emotion': 'negative'}

    def _detect_emotion(self, polarity: float, subjectivity: float) -> str:
        """Simplified emotion detection based on polarity and subjectivity."""
        if polarity > 0.3:
            return 'positive'
        elif polarity < -0.3:
            return 'negative'
        else:
            if subjectivity > 0.7:
                return 'confused'
            else:
                return 'neutral'

    def handle_error(self, error: Exception) -> None:
        """Log and handle errors during sentiment analysis."""
        logging.error(f"Error in sentiment analysis: {str(error)}")