import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os

# Download VADER lexicon for sentiment analysis
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def load_and_process_data(filepath):
    """
    Loads student feedback data, concatenates feedback columns,
    and generates sentiment labels.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    
    # List of columns containing feedback text
    feedback_cols = [
        "What did you like most about this course and why?",
        "Which topics were most useful for your understanding or future career?",
        "Which topics were difficult or unclear? Please mention briefly.",
        "How effective was the teaching method used in this course?",
        "Were the lectures and study materials helpful? Explain shortly.",
        "How can this course be improved for future students?",
        "Did this course meet your expectations? Why or why not?",
        "How was the pace of teaching (too fast, slow, or balanced)? Explain briefly.",
        "What practical skills or knowledge did you gain from this course?",
        "Any other suggestions or comments?"
    ]
    
    # Concatenate all feedback columns into one 'combined_text'
    # We add a space between columns to avoid merging words
    df['combined_text'] = df[feedback_cols].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    
    # Use explicit Ground Truth Label if available, else fallback (e.g. for inference)
    if 'Ground_Truth_Label' in df.columns:
        df['sentiment_label'] = df['Ground_Truth_Label']
        print("Using Ground Truth Labels.")
    else:
        # Fallback to VADER only if ground truth is missing (e.g. user upload)
        print("Ground Truth not found. Using VADER for labelling.")
        sia = SentimentIntensityAnalyzer()
        def get_sentiment(text):
            score = sia.polarity_scores(text)
            return 1 if score['compound'] >= 0.05 else 0
        df['sentiment_label'] = df['combined_text'].apply(get_sentiment)
    
    # Select only necessary columns for the model
    processed_df = df[['Name', 'Roll No', 'Course', 'combined_text', 'sentiment_label']]
    
    print(f"Processed {len(processed_df)} records.")
    print("Sentiment distribution:")
    print(processed_df['sentiment_label'].value_counts())
    
    return processed_df

if __name__ == "__main__":
    # Test the processor
    path = 'd:/final_year_project/final_year_project/student_feedback.csv'
    df = load_and_process_data(path)
    df.to_csv('d:/final_year_project/final_year_project/processed_feedback.csv', index=False)
    print("Saved processed data to processed_feedback.csv")
