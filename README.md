# Sentiment Analysis on Amazon Reviews

This project focuses on **sentiment analysis** of Amazon reviews by comparing traditional machine learning models and deep learning approaches. We have explored multiple models ranging from **Naive Bayes** and **Logistic Regression** to more advanced **LSTM** and **pre-trained BERT models**. The goal is to identify which model performs best in classifying **positive** and **negative** reviews.

## Table of Contents

- [Project Overview](#project-overview)
- [Models Used](#models-used)
- [Data Preprocessing](#data-preprocessing)
- [License](#license)

## Project Overview

In this project, we compare the performance of various models for sentiment analysis on Amazon product reviews. The dataset includes customer reviews with corresponding sentiment labels (positive or negative). We apply both traditional machine learning and deep learning models to perform this task, showcasing the results and insights derived from each approach.

## Models Used

1. **Naive Bayes**: A probabilistic model commonly used for text classification.
2. **Logistic Regression**: A simple and interpretable linear model used for binary classification.
3. **LSTM (Long Short-Term Memory)**: A type of recurrent neural network that captures sequential dependencies in text data.
4. **Pre-trained BERT**: A transformer-based model, fine-tuned for sentiment classification using the Hugging Face `transformers` library.

## Data Preprocessing

To prepare the text data for model training, we performed the following preprocessing steps using **NLTK**, **SpaCy**, and **TextBlob** libraries:

- **Lowercasing**: Convert all text to lowercase for uniformity.
- **Removing Punctuation**: Eliminate unnecessary punctuation.
- **Spell Check**: Correct spelling mistakes using the **TextBlob** library.
- **Removing Stopwords**: Remove common but insignificant words (e.g., "the", "is") using **NLTK**.
- **Tokenization**: Split text into individual words/tokens using the **SpaCy** library.
- **Lemmatization**: Reduce words to their base or root form.
  
For converting text into numerical vectors, we tried multiple techniques:

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Captures the importance of words across documents.
- **Bag of Words (BoW)**: Simple word frequency representation.
- **n-Grams**: Capture contiguous sequences of words (bi-grams, tri-grams).

## License

This project is licensed under the MIT License.
