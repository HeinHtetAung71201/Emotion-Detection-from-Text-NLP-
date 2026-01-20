import pandas as pd
import re
import nltk, string
nltk.download('stopwords')

from nltk.corpus import stopwords

data= pd.read_csv("../archive/tweet_emotions.csv")
# print(data.head())
stop_words=set(stopwords.words('english'))

def clean_tweet(text):
    #Lowercase
    text = text.lower()
    #Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    #Remove Mentions (@user)
    text = re.sub(r'@\w+', '', text)
    #Remove Hashtag symbol (keep the word)
    text = re.sub(r'#', '', text)
    #Remove HTML special entities (like &amp;, &quot;)
    text = re.sub(r'&\w+;', '', text)
    #remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    #Remove extra whitespace
    text = " ".join(text.split())

    return text

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

data['clean_text'] = data['content'].apply(clean_tweet)
data['Content']= data['clean_text'].apply(remove_stopwords)
data['sentiment']= data['sentiment'].replace('empty','neutral')
# print(data['clean_text'].head(),"Cleaned Text <<<<")
print(data[['Content','sentiment']].head(),"Cleaned SW <<<<")
data= data[data['Content'].str.strip() != ""]
data.dropna(subset=['Content'], inplace=True)
print(data[['Content','sentiment']],"Complete Cleaning<<<")
data[['Content','sentiment']].to_csv('cleaned_data.csv', index=False)


