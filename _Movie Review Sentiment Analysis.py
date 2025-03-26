#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")  # Ensure the dataset file is in the correct path

# Preprocessing function
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

# Apply preprocessing
df["cleaned_review"] = df["review"].apply(preprocess_text)

df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})  # Convert labels to binary

# Convert text to numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_review"]).toarray()
y = df["sentiment"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Sentiment Prediction Function
def predict_sentiment(review):
    cleaned_review = preprocess_text(review)
    review_vector = vectorizer.transform([cleaned_review]).toarray()
    prediction = model.predict(review_vector)[0]
    return "Positive" if prediction == 1 else "Negative"

# Save model & vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tokenizer.pkl")

print("Model and vectorizer saved successfully!")


# In[31]:


import tkinter as tk
from tkinter import messagebox
import joblib

# Load trained model and vectorizer
def load_model():
    global model, vectorizer
    try:
        model = joblib.load("sentiment_model.pkl")  # Load model
        vectorizer = joblib.load("tokenizer.pkl")  # Load vectorizer
    except FileNotFoundError:
        messagebox.showerror("Error", "Model files not found!")
        return

# Predict sentiment
def predict_sentiment():
    review = entry_review.get("1.0", "end-1c")  # Get user input
    if not review.strip():
        messagebox.showwarning("Warning", "Please enter a review!")
        return

    transformed_review = vectorizer.transform([review])  # Transform text
    prediction = model.predict(transformed_review)[0]
    result = "Positive 😊" if prediction == 1 else "Negative 😢"

    # Update label with result
    label_result.config(text=f"Sentiment: {result}", fg="green" if prediction == 1 else "red")

# Exit Application
def exit_app():
    if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
        root.destroy()

# GUI Setup
root = tk.Tk()
root.title("Movie Review Sentiment Analysis")
root.geometry("500x400")
root.configure(bg="#282c34")

# Title Label
label_title = tk.Label(root, text="Movie Sentiment Analyzer", font=("Arial", 18, "bold"), fg="white", bg="#282c34")
label_title.pack(pady=10)

# Review Input
entry_review = tk.Text(root, height=5, width=50, font=("Arial", 12))
entry_review.pack(pady=10)

# Predict Button
btn_predict = tk.Button(root, text="Analyze Sentiment", font=("Arial", 12, "bold"), bg="#61afef", fg="white",
                        command=predict_sentiment)
btn_predict.pack(pady=10)

# Result Label
label_result = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#282c34")
label_result.pack(pady=10)

# Exit Button
btn_exit = tk.Button(root, text="Exit", font=("Arial", 12, "bold"), bg="red", fg="white", command=exit_app)
btn_exit.pack(pady=10)

# Load Model
load_model()

# Run Application
root.mainloop()


# In[ ]:




