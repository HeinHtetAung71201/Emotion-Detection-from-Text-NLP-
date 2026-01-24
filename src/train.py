import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the data you already cleaned
data = pd.read_csv('cleaned_data.csv')
print(data['sentiment'].value_counts(),"<<<counting sentiments")
data.dropna(inplace=True) # Ensure no empty rows sneaked in

# 2. TF-IDF Phase
# tfidf = TfidfVectorizer(
#     max_features=3000, 
#     ngram_range=(1,2), 
#     min_df=5, 
#     sublinear_tf=True
# )
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    stop_words=None
)
X = tfidf.fit_transform(data['Content'])
y = data['sentiment']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Feature matrix shape: {X_train.shape}")

# 4. Training Phase
# model = MultinomialNB()
# model.fit(X_train, y_train)
# # print(model)

# print(f"New Accuracy: {model.score(X_test, y_test):.2%}")
model = ComplementNB(alpha=0.5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# 5. Check Results
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions):.2%}")

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))