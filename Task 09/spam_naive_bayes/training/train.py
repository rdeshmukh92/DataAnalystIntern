import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

#Load dataset
df = pd.read_csv("E:/MS Ritika/Projects/DataAnalystIntern/Task 09/spam_naive_bayes/data/naive_bayes_spam_dataset.csv")

#Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

df["clean_text"] = df["email_text"].apply(clean_text)

#Features
vectorizer = TfidfVectorizer(stop_words= 'english', max_features= 5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

#Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.2, random_state= 42
)

#Model
model = MultinomialNB()
model.fit(X_train, y_train)

#Evalution
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#Save model
joblib.dump(model, "E:/MS Ritika/Projects/DataAnalystIntern/Task 09/spam_naive_bayes/models/model.pkl")
joblib.dump(vectorizer, "E:/MS Ritika/Projects/DataAnalystIntern/Task 09/spam_naive_bayes/models/vectorizer.pkl")

print("Model saved successfully!")