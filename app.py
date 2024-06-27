import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(news_summary):
    sentiment_scores = sia.polarity_scores(news_summary)
    
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, sentiment_scores

def main():
    st.title("Sentiment Analysis")
    st.write("Summary to determine its sentiment.")

    news_summary = st.text_area("Enter here:")

    if st.button("Analyze Sentiment"):
        if news_summary:
            sentiment, scores = analyze_sentiment(news_summary)
            st.write(f"Sentiment: **{sentiment}**")
            st.write("Sentiment Scores:")
            st.json(scores)
        else:
            st.write("Please enter a news summary to analyze.")

if __name__ == "__main__":
    main()
