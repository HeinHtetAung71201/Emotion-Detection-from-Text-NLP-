import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import ComplementNB # Changed to ComplementNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Negation Handling ---
def mark_negation(text):
    if not isinstance(text, str): return ""
    neg_words = {"not", "no", "never", "dont", "cant", "isnt", "wasnt"}
    words = text.lower().split()
    transformed = []
    neg_mode = False
    for word in words:
        clean = re.sub(r'[^\w]', '', word)
        if neg_mode:
            transformed.append(clean + "_NEG")
            neg_mode = False 
        else:
            transformed.append(clean)
        if clean in neg_words or clean.endswith("nt"):
            neg_mode = True
        if any(p in word for p in [".", "!", "?", ","]):
            neg_mode = False
    return " ".join(transformed)

# --- Load and Preprocess ---
data = pd.read_csv('cleaned_data.csv')
data.dropna(subset=['Content', 'sentiment'], inplace=True)

# --- NEW: BOOSTER DATA ---
# If your model fails 'not sad', we manually feed it examples to correct its bias
booster = pd.DataFrame({
    'Content': ['i am not sad', 'i am not angry', 'im not upset', 'not feeling bad', 'not sad'],
    'sentiment': ['positive', 'positive', 'positive', 'positive', 'positive']
})
data = pd.concat([data, booster], ignore_index=True)

# Apply negation marking
data['Content_Processed'] = data['Content'].apply(mark_negation)

# --- STEP 3: TF-IDF Phase ---
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3), # capturing "i_am_not", "am_not_sad", etc.
    min_df=2,
    sublinear_tf=True,
    stop_words=None 
)
X = tfidf.fit_transform(data['Content_Processed'])
y = data['sentiment']

# --- STEP 4: N-Fold Cross-Validation ---
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
# Using ComplementNB instead of MultinomialNB
model = ComplementNB(alpha=0.1) 

print(f"--- Running {n_folds}-Fold Cross-Validation ---")
cv_scores = cross_val_score(model, X, y, cv=skf)
print("5-fold Validation Accuracy: ",cv_scores)
print(f"Average Accuracy: {cv_scores.mean():.2%}\n")

# --- STEP 5: Final Train ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("--- Accuracy Evaluation ---")
print(f"Dataset (Training) Accuracy: {accuracy_score(y_train, train_preds):.2%}")
print(f"Test Set Accuracy: {accuracy_score(y_test, test_preds):.2%}")

print("\n--- Confusion Matrix  ---")
# Define the labels in the correct order
labels = ['negative', 'neutral', 'positive']

# Create the raw confusion matrix
cm = confusion_matrix(y_test, test_preds, labels=labels)

# Convert to a DataFrame for pretty-printing in the prompt
cm_df = pd.DataFrame(
    cm, 
    index=[f"Actual_{l}" for l in labels], 
    columns=[f"Pred_{l}" for l in labels]
)

print(cm_df)

# Show the detailed report as well
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, test_preds, target_names=labels))