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
