# Movie Review Sentiment Analysis
## Overview
Sentiment analysis of movie reviews using Logistic Regression with a Tkinter GUI.

### Features<br>
• Preprocesses text (lowercasing, tokenization, stopword removal, TF-IDF vectorization).<br>
• Trains and evaluates a Logistic Regression model.<br>
• GUI for real-time sentiment prediction.

#### Requirements
Install dependencies:
```pip install pandas numpy nltk scikit-learn joblib tkinter```

##### Usage
1. Ensure ```IMDB Dataset.csv``` is available(You can download it from Kaggle).
2. Run:
   ```python sentiment_analysis.py```<br>
3.Enter a review in the GUI and click Analyze Sentiment.

**Output**<br>
Displays sentiment (Positive or Negative).<br>
Trained model (sentiment_model.pkl) and vectorizer (tokenizer.pkl) are saved.<br>

**License**<br>
This project is licensed under the MIT License.



