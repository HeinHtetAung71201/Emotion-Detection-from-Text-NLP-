import preprocess as pre
import train 
new_tweet = ["you're a bad girl but actually nice attitude"]
cleaned_new = pre.clean_tweet(new_tweet[0]) # Use your function from step 1
vectorized_new = train.tfidf.transform([cleaned_new])
prediction = train.model.predict(vectorized_new)

print(f"The predicted emotion is: {prediction[0]}")