import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup

nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment

def fetch_news_sentiment(symbol):
    # This is a simple example. In a real-world scenario, you'd use a more robust news API.
    url = f"https://finance.yahoo.com/quote/{symbol}/news"
    
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = soup.find_all('h3', class_='Mb(5px)')
        
        sentiments = []
        for item in news_items[:5]:  # Analyze sentiment for top 5 news headlines
            headline = item.text
            sentiment = analyze_sentiment(headline)
            sentiments.append({
                'headline': headline,
                'sentiment': sentiment
            })
        
        return sentiments
    except Exception as e:
        print(f"Error fetching news for {symbol}: {str(e)}")
        return []