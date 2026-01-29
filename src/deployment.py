# import preprocess as pre
# import train 
# # new_tweet = ["I'm haappy and glad to see you"]
# new_tweet = input ("Enter new tweet : ")
# cleaned_new = pre.clean_tweet(new_tweet)
# mark_neg= train.mark_negation(cleaned_new)
# vectorized_new = train.tfidf.transform([mark_neg])
# prediction = train.model.predict(vectorized_new)

# print(f"The predicted emotion is: {prediction[0]}")

import preprocess as pre
import train 

# User input sentence
new_tweet = input("Enter new tweet: ")

# Optional: true label for validation
true_label = input("Enter true emotion (optional, press Enter to skip): ").strip()

# Clean
cleaned_new = pre.clean_tweet(new_tweet)
mark_neg= train.mark_negation(cleaned_new)


# Vectorize
vectorized_new = train.tfidf.transform([mark_neg])

# Predict
prediction = train.model.predict(vectorized_new)
probabilities = train.model.predict_proba(vectorized_new)

predicted_label = prediction[0]
confidence = max(probabilities[0]) * 100

print("\n========== PREDICTION RESULT ==========")
print("Cleaned Text :", cleaned_new)
print("Predicted Emotion :", predicted_label)
print(f"Confidence : {confidence:.2f}%")

# Validation if true label provided
if true_label != "":
    if true_label.lower() == predicted_label.lower():
        print("✅ Prediction is CORRECT")
        print("Sentence Accuracy: 100%")
    else:
        print("❌ Prediction is WRONG")
        print("Sentence Accuracy: 0%")


