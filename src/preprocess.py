import pandas as pd
import re
import nltk
import string

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
data = pd.read_csv("../archive/tweet_emotions.csv")

# =============================
# Stopwords (keep negations)
# =============================
stop_words = set(stopwords.words('english'))
negation_words = {"not", "no", "never", "n't"}
stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()

# =============================
# Text Cleaning Functions
# =============================

def normalize_repeated_letters(word):
    # soooo -> soo
    return re.sub(r'(.)\1{2,}', r'\1\1', word)

def clean_tweet(text):
    if pd.isna(text):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove Mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtag symbol only
    text = re.sub(r'#', '', text)

    # Remove HTML entities
    text = re.sub(r'&\w+;', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_text(text):
    words = text.split()

    processed_words = []
    for w in words:
        if w not in stop_words:
            w = normalize_repeated_letters(w)
            w = lemmatizer.lemmatize(w)
            processed_words.append(w)

    return " ".join(processed_words)

# =============================
# Apply Cleaning
# =============================
data['clean_text'] = data['content'].apply(clean_tweet)
data['Content'] = data['clean_text'].apply(preprocess_text)

# Remove empty rows
data = data[data['Content'].str.strip() != ""]
data.dropna(subset=['Content'], inplace=True)

# =============================
# Sentiment Grouping
# =============================

def group_sentiments(sentiment):
    positives = ['happiness', 'love', 'surprise', 'fun', 'enthusiasm', 'relief']
    negatives = ['worry', 'sadness', 'hate', 'anger', 'boredom']
    neutral = ['neutral', 'empty']

    if sentiment in positives:
        return 'positive'
    elif sentiment in negatives:
        return 'negative'
    elif sentiment in neutral:
        return 'neutral'
    else:
        return 'neutral'

data['sentiment'] = data['sentiment'].apply(group_sentiments)

# =============================
# Save Output
# =============================
print(data[['Content','sentiment']].head(), "<<< Sample Cleaned Data")

data[['Content','sentiment']].to_csv('cleaned_data.csv', index=False)

print("\nClass Distribution:")
print(data['sentiment'].value_counts(),"<<<Counting Sentiments")
