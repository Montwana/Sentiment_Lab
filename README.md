# Sentiment_Lab

## Why it is important to remove HTML tags and punctuation before modelling.

HTML tags are not meaningful words, so including them adds noise 
to your text features.

Punctuation can also add noise for models like Bag-of-Words or TF-IDF, unless explicitly used for sentiment (like emoticons).

Benefit: Cleaning ensures your model focuses on actual words conveying meaning, improving accuracy and reducing dimensionality.

## How emoticons contribute to sentiment analysis.

Emoticons directly express emotion, which is highly relevant for sentiment.

Including emoticons as tokens helps the model capture sentiment signals that words alone may miss.

## Comparison of logistic regression performance using different vectorizers.

### CountVectorizer

Simple Bag-of-Words (word counts)

Baseline performance, may be biased toward frequent words

### TF-IDF Vectorizer

Weights words by importance (inverse document frequency)

Usually improves performance by reducing influence of common words like "the", "and"

### HashingVectorizer

Memory-efficient and streaming-friendly

Works well for large datasets but slightly lower accuracy than TF-IDF sometimes because feature indices are fixed

## Problem out-of-core learning solve

Problem: Large datasets (e.g. IMDB 50k reviews) may not fit into memory.

Solution: Out-of-core learning (SGDClassifier + HashingVectorizer) trains the model in batches directly from disk.

Benefit: Allows training on very large datasets without using too much RAM.

## What LDA Topics represent

LDA topics: Groups of words that often appear together, representing “topics.”

Each topic: A distribution of words; each document can have multiple topics.

Example: Topic 1: “love, amazing, fantastic, best, great”, likely positive sentiment.

Purpose: Helps discover hidden themes in reviews beyond just sentiment.


## Summary

## IMDB Movie Review Sentiment Analysis

### Project Goal:
The main goal of this project was to build a system that can automatically analyze movie reviews and classify them as positive or negative. Additionally, we explored patterns and topics within the reviews to uncover common themes.

### Dataset Description:
We used the IMDB Large Movie Review Dataset, containing 50,000 reviews split evenly between training and testing sets, with equal numbers of positive and negative reviews. Each review is raw text, some containing HTML tags, punctuation, and emoticons.

### Preprocessing Steps:
Removed HTML tags to clean unnecessary text.
Converted text to lowercase and removed most punctuation.
Extracted and retained emoticons as they convey sentiment.
Tokenized text and optionally removed stopwords.

### Methods Used:
TF-IDF Vectorizer: Converts text into weighted numeric features, emphasizing informative words.
Logistic Regression: Used with GridSearchCV to find the best hyperparameters, achieving strong sentiment classification.
SGDClassifier + HashingVectorizer: Enabled out-of-core learning, allowing batch-wise training on the full dataset without exhausting memory.
Latent Dirichlet Allocation (LDA): Identified hidden topics in reviews by grouping frequently co-occurring words.

### Key Findings:
Logistic Regression with TF-IDF achieved 86% accuracy on the training set.
Top LDA topics revealed themes like positive feelings, movie plot, acting quality, humour, and drama.
Emoticons contributed noticeably to detecting sentiment that words alone sometimes missed.

### Observations / Insights:
Cleaning and preprocessing text is crucial for good model performance.
TF-IDF generally outperforms simple count-based features by reducing the impact of very common words.
Out-of-core learning is practical for large datasets and prevents memory issues.
Topic modeling adds value by uncovering hidden themes, providing insights beyond sentiment classification.

### Conclusion:
This project demonstrates how combining text preprocessing, feature extraction, supervised learning, and topic modeling can effectively analyze large-scale text data. The workflow can be extended to other sentiment-based or text-mining applications.


