
# Introduction

We've all been there — trying to transfer money in a hurry, only for the app to crash. Or waiting endlessly for a simple transaction to go through, dealing with poor customer support, or struggling with a confusing user interface. As a regular customer, it’s easy to feel ignored, disappointed, and even angry.
When that frustration builds up, what do most users think about?
- Closing the bank account altogether
- Switching to a more reliable bank
- Venting on social media or review platforms
These emotional reactions are more than just complaints — they’re valuable data.
For this project, I’m stepping into the role of a data analyst working with a digital banking provider. My goal is to analyze customer sentiment using real user feedback to uncover:
- What are customers complaining about most?
- Are there trends in negative feedback that could predict churn?
- Are there insights that could help improve the mobile banking experience?

By applying **Natural Language Processing (NLP)** techniques and sentiment analysis, this project aims to turn raw customer voices into **actionable business insights** that can help financial institutions improve **user experience**, **reduce churn**, and build stronger digital trust.

---
## 📊 Dataset

The dataset I’m using is sourced from Google play store and contains 8,798 rows and 14 columns. Each row represents user feedback on a mobile banking application, and each entry is tied to a specific banking app name — allowing for a comparison of user experiences across different financial institutions.

The columns in the dataset include a mix of textual data (like comments), ratings, developer reply and categorical variables such as the app name, review date, and version. These data points are ideal for sentiment analysis, topic modeling, and uncovering patterns in customer satisfaction or dissatisfaction.

For this project, I chose to use Python and key libraries like **NLTK**, **Tensorflow**, and **scikit-learn** for text preprocessing and analysis. Before diving into the modeling, I had to go through a rigorous data cleaning phase — handling missing values, dropping unecessary columns, coverting text to lower case, removing hashtags and normalizing the text for better sentiment interpretation.

This dataset gives me a solid foundation to explore what users are saying, how they feel about their banking experience, and where mobile apps might be falling short.

---
### 💡 Insights & Recommendations
### Key Insights
1. Customer Sentiment Skews Negative
Out of 8,798 reviews:
- 4,653 were negative
- 3,444 were positive
- Only 695 were neutral
This shows that many users are having a frustrating experience with their banking apps, and they’re not holding back.

2. Frequent Issues Highlighted in Negative Reviews
- The negative sentiment word cloud highlights common frustrations:
- Words like **“can’t,”** **“update,”** **“login,”** **“error,”** **“time,”** **“problem,”** and **“frustrating”** appear frequently.
- Users are mainly struggling with:
  - App crashes or failed updates
  - Login and authentication issues
  - Delayed or failed transactions
  - Poor customer service and response times
  - Phrases like “still not working,” “can’t access,” and “help please” suggest a lack of technical reliability and user support.

3. Positive Feedback Centers Around Functionality and Convenience
- The positive sentiment word cloud shows terms like:
  - **“good,”** **“excellent,”** **“easy,”** **“convenient,”** **“secure,”** **“service,”** **“transaction,”** and **“efficient”**
- Positive reviewers appreciate:
  - Ease of use
  - Transaction reliability
  - Security and account management features
- Notably, users also express gratitude with words like “thank,” “love,” and “awesome,” indicating that when the app works well, it builds strong loyalty.
---
### ✅ Recommendations
Based on these insights, here are a few ways mobile banking apps can improve the user experience:
1. Prioritize Stability and Performance
- Regularly test and monitor app updates to avoid crashes and bugs.
- Invest in better load handling and bug fixing before release.
2. Improve Login and Security UX
- Make authentication more reliable and seamless.
- Offer clear error messages and easy ways to reset or recover login credentials.
3. Enhance Customer Support Integration
- Integrate in-app support or chatbots to respond to user issues faster.
- Respond to reviews — especially negative ones — to show customers their voices matter.
4. Leverage Positive Feedback for Feature Development
- Double down on what users love: convenience, transaction reliability, and security.
- Use positive reviews as a guide to what should be emphasized or retained in future updates.
5. Prioritize real time Monitoring of Sentiments Continuously
- Set up sentiment tracking tools to stay ahead of user frustration.
- Identify patterns early and act before issues snowball into churn.

---
## Methodology
You might be wondering — how did I come up with these insights?

Well, let me walk you through the process I followed to clean the data, analyze the sentiments, and visualize the results. Step by step, I was able to turn thousands of raw reviews into something meaningful.

### Step 1: **Data Cleaning – Making Sense of the Mess**
Curious how I got started? Well, before diving into any analysis, I had to roll up my sleeves and clean the data. And let’s be honest — real-world data is never clean!

One of the first things I did was check for missing values:
```Python
# Check null values
data.isnull().sum()

This gave me the following result below:
App_Name               0  
Company_Name           0  
Page_URL               0  
Official_website       0  
Email                  0  
Address             7659  
Privacy_policy         0  
Name                   0  
Time                   0  
Year                 569  
Star_rating            6  
Helpful             4904  
Comments               0  
Developer_Reply     3099  

```
Handling Missing Values
After identifying which columns had missing values, the next step was figuring out what to do with them. Not all missing data needs to be deleted — sometimes, we can make smart assumptions to fill in the blanks. Here's how I handled it:
```Python

df["Year"] = df["Year"].fillna(df["Year"].mode()[0])  
df["Helpful"] = df["Helpful"].fillna(0)  # Assume missing means no helpful votes
df["Developer_Reply"] = df["Developer_Reply"].fillna("No Reply")  # Replace missing replies with "No Reply"

# Drop rows where 'Star_rating' is missing 
df = df.dropna(subset=["Star_rating"])

# Verify missing values
print(df.isnull().sum())

This was the output
App_Name           0
Company_Name       0
Time               0
Year               0
Star_rating        0
Helpful            0
Comments           0
Developer_Reply    0
dtype: int64

```
Checking for Duplicates – One Voice per User

After cleaning up missing values, the next thing I wanted to make sure of was that each review counted only once. Duplicate reviews can skew the analysis — especially if they contain strong sentiments (positive or negative) and show up multiple times.
```
#check duplicates
duplicates = df.duplicated()  # Returns a Boolean Series
print(f"Number of duplicate rows: {duplicates.sum()}")
```
---
**Text Cleaning – Tidying Up the Reviews**

Before I could analyze user reviews I had to clean up the comments. Raw user reviews can be messy — filled with emojis, hashtags, mentions, links, and extra spaces. So I created a custom preprocess_text() function to tidy things up.

Here’s what the function did:

- Converted all text to lowercase (for consistency)
- Removed user mentions, hashtags, URLs, punctuation, emojis, and stopwords
- Stripped away extra spaces and newline characters

This step made sure that the text was clean, simple, and ready for sentiment analysis — without all the noisy distractions. The code is shown below.

```
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
```
---

### Exploratory Data Analysis (EDA) – Getting to Know the Data
Once the data was clean, I spent some time just getting to know it. Think of EDA like the “first date” with your dataset — you're asking questions, spotting trends, and seeing what stands out.
Here’s what I explored:
- **Star Rating Distribution:**
I looked at how users rated the apps from 1 to 5 stars. The distribution showed a strong polarity — lots of 1-star and 5-star reviews, with fewer in the middle. This hinted at extreme user experiences: people either loved or hated the apps.
![App Rating Distribution](images/rating_distribution.png)


### 🧪 Code Snippet

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_label(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
text = "This app is amazing and super user-friendly!"
score = analyzer.polarity_scores(text)
print(score)
# Output: {'neg': 0.0, 'neu': 0.254, 'pos': 0.746, 'compound': 0.8519} 
---

``` 

## ⚙️ Tools & Technologies

- Python (Jupyter Notebook, VS Code)
- TensorFlow / Keras
- NLTK, spaCy
- Gensim, GloVe, FastText
- Matplotlib / Seaborn for visualization
- Google Colab (for model training)
- Git & GitHub (version control)

---

## 📈 Results & Key Findings

- **Bi-LSTM** performed best with an accuracy of **78%** using **GloVe embeddings**.
- **FastText** offered better performance on low-frequency words.
- VADER scores correlated highly with Bi-LSTM predictions, validating model output consistency.
- User concerns centered around **app crashes**, **transaction failures**, and **user interface** issues.

---

## 📸 Visualizations

### ☁️ Word Cloud of Most Frequent Words
A WordCloud was generated to visualize the most common terms in the user reviews. This helps identify key themes and sentiments expressed by users.

![WordCloud](wordcloud.png)

---

### 📈 Sentiment Distribution
This plot shows the distribution of sentiments (Positive, Negative, Neutral) generated using VADER, which were used to label data for training the models.

![Sentiment Distribution](sentimentdistribution.png)

---

### 🧠 BiLSTM Model Output
Below is a sample output from the **BiLSTM model**, which had the highest performance among the models evaluated. The model successfully predicts the sentiment label based on the review input.

![BiLSTM Output](bilstm_model.PNG)

---

### 🌐 Web Application Interface
The trained model was deployed using **Streamlit** with an interactive web interface. Banks can upload data and get sentiment classifications.

![Web Interface](webinterface.jpg)

  












