# 📦 Imports
import streamlit as st
import pandas as pd
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

# ⬇️ Download NLTK Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# 🎨 Page Setup
st.set_page_config(page_title="🧠 Smart Feedback Analyzer", layout="wide")
st.title("🧠 Smart Feedback Analyzer")
st.markdown("Analyze user feedback, detect sentiments, and discover trending topics 🔍")

# 🧹 Clean Text Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# 📊 Sentiment Detector
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# 🔤 Top Keywords Extractor
def extract_keywords(texts, top_n=5):
    vec = CountVectorizer(stop_words='english')
    matrix = vec.fit_transform(texts)
    sum_words = matrix.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]

# 🌥️ WordCloud Generator
def generate_wordcloud(text_list):
    text = " ".join(text_list)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wc

# 📥 Input Area
st.subheader("✍️ Enter Feedbacks (one per line)")
user_input = st.text_area("Paste all feedbacks here", height=200, placeholder="Example:\nI love this app!\nWorst experience ever.\n...")

if st.button("🔍 Analyze Feedbacks"):
    if user_input.strip() == "":
        st.warning("Please enter some feedbacks to analyze.")
    else:
        feedbacks = user_input.strip().split("\n")
        df = pd.DataFrame(feedbacks, columns=["Original Feedback"])
        df["Cleaned"] = df["Original Feedback"].apply(clean_text)
        df["Sentiment"] = df["Cleaned"].apply(get_sentiment)
        df["Polarity"] = df["Cleaned"].apply(lambda x: TextBlob(x).sentiment.polarity)

        # 📌 Summary
        st.subheader("📈 Sentiment Summary")
        st.dataframe(df["Sentiment"].value_counts().rename_axis('Sentiment').reset_index(name='Count'))

        # 🟢 Top 3 Positive
        st.subheader("🌟 Top 3 Positive Feedbacks")
        top_pos = df[df["Sentiment"] == "Positive"].sort_values(by="Polarity", ascending=False).head(3)
        for i, row in top_pos.iterrows():
            st.success(row["Original Feedback"])

        # 🔴 Top 3 Negative
        st.subheader("⚠️ Top 3 Negative Feedbacks")
        top_neg = df[df["Sentiment"] == "Negative"].sort_values(by="Polarity").head(3)
        for i, row in top_neg.iterrows():
            st.error(row["Original Feedback"])

        # 🧠 Trending Topics
        st.subheader("📌 Trending Topics (Top 5 Words)")
        top_keywords = extract_keywords(df["Cleaned"])
        for word, freq in top_keywords:
            st.write(f"🔹 **{word}** – {freq} times")

        # ☁️ Word Cloud
        st.subheader("☁️ Word Cloud of Feedbacks")
        wordcloud = generate_wordcloud(df["Cleaned"].tolist())
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # 📋 Full Table
        st.subheader("📋 Full Feedback Analysis Table")
        st.dataframe(df[["Original Feedback", "Sentiment", "Polarity"]])

