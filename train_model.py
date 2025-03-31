#importing necessary libraries
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow import keras
import nltk
import os
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Dense, LSTM, SimpleRNN, Bidirectional
from tensorflow.keras.models import Model
from gensim.models import FastText
#import torch
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


#loading dataset
data = pd.read_csv("SENTIMENT_ANALYSIS.csv",  encoding="latin1")
data.head()

# Drop unnecessary columns
data = data.drop(columns=["Email", "Address", "Privacy_policy", "Name", "Page_URL", "Official_website", "App_Name"])

# Fill missing values
data["Year"] = data["Year"].fillna(data["Year"].mode()[0])  # Fill with mode (most common year)
data["Helpful"] = data["Helpful"].fillna(0)  # Assume missing means no helpful votes
data["Developer_Reply"] = data["Developer_Reply"].fillna("No Reply")  # Replace missing replies with "No Reply"

# Drop rows where 'Star_rating' is missing (only 6 rows)
data = data.dropna(subset=["Star_rating"])

# Verify missing values
print(data.isnull().sum())


#data preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags (#topic)
    text = re.sub(r'#\w+', '', text)

    # Remove newline characters
    text = re.sub(r'\n', ' ', text)

    # Remove single quotes
    text = re.sub(r"'", '', text)

    # Remove hyperlinks
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & map symbols
        u"\U0001F700-\U0001F77F"  # Alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric shapes
        u"\U0001F800-\U0001F8FF"  # Supplemental arrows
        u"\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and pictographs
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # Enclosed characters
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    return text

#Apply Preprocessing to the Dataset
data['Comments'] = data['Comments'].apply(preprocess_text)

# Tokenize comments
tokenized_comments = [comment.split() for comment in data['Comments']]

# Train FastText model
fasttext_model = FastText(sentences=tokenized_comments, vector_size=100, window=5, min_count=1, sg=1, epochs=10)

# Function to compute FastText embeddings
def fasttext_embedding(tokens):
    return np.mean([fasttext_model.wv[word] for word in tokens if word in fasttext_model.wv], axis=0)

# Compute embeddings for all comments using list comprehension
data['Embeddings'] = [fasttext_embedding(tokens) for tokens in tokenized_comments]

# Drop rows where there is null values
data = data.dropna(subset=["Embeddings"])

# Download VADER lexicon
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize VADER Sentiment Analyzer

# Function to analyze sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to determine sentiment
def get_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
data['Sentiment_Score'] = data['Comments'].apply(lambda x: sid.polarity_scores(x)['compound'])
data['sentiment'] = data['Sentiment_Score'].apply(get_sentiment)

# Initialize LabelEncoder
from sklearn.preprocessing import LabelEncoder
encoder_1 = LabelEncoder()

# Fit and transform the sentiment column (only the labels)
data['sentiment'] = encoder_1.fit_transform(data['sentiment'])

print(data['sentiment'].value_counts())

# FastText embeddings as input features
X = np.array(data['Embeddings'].tolist())  
y = np.array(data['sentiment'])  # Sentiment labels

# Check unique labels before and after splitting the data
print("Unique classes in entire dataset:", np.unique(y))

from sklearn.model_selection import train_test_split

# Split data: 80% training, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"Balanced Training set: {X_train_bal.shape}, {y_train_bal.shape}")

# Reshape function
def reshape_for_rnn(X):
    return X.reshape((X.shape[0], 1, X.shape[1]))  # (samples, 1 timestep, features)

# Reshape training and validation sets
X_train_rnn = reshape_for_rnn(X_train_bal)
X_val_rnn = reshape_for_rnn(X_val)

print(f"RNN-ready X_train shape: {X_train_rnn.shape}")
print(f"RNN-ready X_val shape: {X_val_rnn.shape}")

from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
# Build BiLSTM model
bilstm_model = Sequential()
bilstm_model.add(Bidirectional(LSTM(128, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]))))
bilstm_model.add(Dropout(0.5))
bilstm_model.add(Dense(64, activation='relu'))
bilstm_model.add(Dropout(0.5))
bilstm_model.add(Dense(len(np.unique(y_train_bal)), activation='softmax'))

# Compile the model
bilstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
bilstm_history = bilstm_model.fit(
    X_train_rnn, y_train_bal,
    epochs=20,
    batch_size=64,
    validation_data=(X_val_rnn, y_val),
    verbose=1
)
# Saves the model as an HDF5 file
bilstm_model.save("sentiment_model.h5")  