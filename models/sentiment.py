from transformers import pipeline
import requests

class SentimentAnalyzer:
    def __init__(self, config):
        # Using a specialized financial sentiment model
        self.sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        self.news_api_key = config.get('news_api_key')

    def get_market_sentiment(self, symbol="BTC"):
        """
        Fetches latest headlines for the symbol and returns an average score.
        """
        # Example using a generic News API (replace with CryptoPanic or similar)
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={self.news_api_key}"
        
        try:
            response = requests.get(url).json()
            articles = [a['title'] for a in response.get('articles', [])[:10]]
            
            if not articles:
                return 0.0 # Neutral if no news
            
            results = self.sentiment_pipe(articles)
            
            # Convert labels to scores: positive = 1, negative = -1, neutral = 0
            scores = []
            for res in results:
                if res['label'] == 'positive':
                    scores.append(res['score'])
                elif res['label'] == 'negative':
                    scores.append(-res['score'])
                else:
                    scores.append(0)
                    
            return sum(scores) / len(scores)
            
        except Exception:
            return 0.0
