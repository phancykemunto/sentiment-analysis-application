import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences


from tensorflow.keras.models import load_model

from keras.models import load_model
bilstm_model = load_model("sentiment_model.h5", compile=False)

df = pd.read_csv("SENTIMENT_ANALYSIS.csv",  encoding="latin1")

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

# Download VADER lexicon
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    if not text:
        return None, None
    scores = sia.polarity_scores(text)
    sentiment = "Positive" if scores["compound"] > 0 else "Negative" if scores["compound"] < 0 else "Neutral"
    return sentiment, scores["compound"]


# Option to choose input method
option = st.radio("Choose Input Method:", ("Enter Text", "Upload Dataset"))

if option == "Enter Text":
    user_input = st.text_area("Enter your text:")

    if st.button("Analyze Sentiment"):
        if user_input:
            sentiment, score = analyze_sentiment(user_input)

            # Display sentiment result
            st.subheader("Sentiment Result:")
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Polarity Score:** {score:.4f}")

            # Generate Word Cloud
            from wordcloud import WordCloud
            st.subheader("Word Cloud")
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(user_input)

            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("Please enter some text.")

elif option == "Upload Dataset":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding='latin')

        # Check for required text column
        text_column = st.selectbox("Select text column:", df.columns)
        
        if st.button("Analyze Dataset"):
            # Apply sentiment analysis
            df["Sentiment"], df["Polarity Score"] = zip(*df[text_column].apply(analyze_sentiment))

            # Display results
            st.subheader("Sample Results:")
            st.dataframe(df.head())

            # Sentiment distribution
            st.subheader("Sentiment Score Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df["Polarity Score"], bins=20, kde=True, ax=ax, color="blue")
            ax.set_xlabel("Sentiment Score (-1 = Negative, 1 = Positive)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            
            # Set the style for plots
            sns.set(style="whitegrid")

# Sentiment Distribution
            st.subheader("Sentiment Distribution")

# Create the plot for sentiment distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(x='Sentiment', data=df, palette='Set2', ax=ax)

# Add value labels to the bars
            for p in ax.patches:
                 ax.annotate(f'{int(p.get_height())}',
                             (p.get_x() + p.get_width() / 2, p.get_height()),
                             ha='center', va='bottom', fontsize=12, color='black')

# Set titles and labels for the sentiment distribution plot
            ax.set_title('Distribution of Sentiments')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')

# Display the sentiment distribution plot
            st.pyplot(fig)


            # Word Cloud
            from wordcloud import WordCloud
            st.subheader("Word Cloud")
            text_data = " ".join(df[text_column].dropna())
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)

            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # Save word cloud image as a PNG file
            wordcloud_image_path = "wordcloud_image.png"
            wordcloud.to_file(wordcloud_image_path)

# Provide a download button for the word cloud image
            with open(wordcloud_image_path, "rb") as file:
                st.download_button(
                    label="Download Word Cloud Image",
                    data=file,
                    file_name="wordcloud_image.png",
                    mime="image/png"
    )

            # Download results
            st.subheader("Download Results")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="sentiment_analysis_results.csv", mime="text/csv")