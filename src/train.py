import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df= pd.read_csv('../archive/cleaned_data.csv')
df.dropna(inplace=True)
print(df.head())
# Encode the sentiment labels
le= LabelEncoder()
y= le.fit_transform(df['sentiment'])
print(y,"<<<")