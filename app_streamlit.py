import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(layout="wide")
st.title("📊 NLP Data Explorer Dashboard")

# ===============================
# CARGAR CSV LIMPIO
# ===============================

@st.cache_data
def load_data():
    return pd.read_csv("dataset/tweetsDisaster/train_clean.csv")

df = load_data()

st.subheader("Vista previa del dataset")
st.dataframe(df.head())

# ===============================
# DISTRIBUCIÓN DE TARGET
# ===============================

st.subheader("🎯 Distribución de Target")

fig_target = px.histogram(df, x="target", title="Target Distribution")
st.plotly_chart(fig_target, use_container_width=True)

# ===============================
# NÚMERO DE HASHTAGS
# ===============================

st.subheader("📊 Número de Hashtags")

fig_hash = px.histogram(
    df,
    x="hashtag_count",
    nbins=20,
    title="Distribution of Hashtag Count"
)

st.plotly_chart(fig_hash, use_container_width=True)

# ===============================
# HASHTAGS vs TARGET
# ===============================

st.subheader("📈 Hashtag Count vs Target")

fig_hash_target = px.box(
    df,
    x="target",
    y="hashtag_count",
    title="Hashtag Count by Target"
)

st.plotly_chart(fig_hash_target, use_container_width=True)

# ===============================
# LOCATION DISTRIBUTION
# ===============================

st.subheader("📍 Top Locations")

top_locations = df["location"].value_counts().head(20).reset_index()
top_locations.columns = ["location", "count"]

fig_loc = px.bar(
    top_locations,
    x="count",
    y="location",
    orientation="h",
    title="Top 20 Locations"
)

st.plotly_chart(fig_loc, use_container_width=True)

# ===============================
# WORDCLOUD DINÁMICO
# ===============================

st.subheader("☁️ WordCloud")

target_filter = st.selectbox(
    "Selecciona target para WordCloud",
    ["Todos", 0, 1]
)

if target_filter == "Todos":
    text_data = " ".join(df["text"].astype(str))
else:
    text_data = " ".join(df[df["target"] == target_filter]["text"].astype(str))

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text_data)

fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
ax_wc.imshow(wordcloud, interpolation="bilinear")
ax_wc.axis("off")

st.pyplot(fig_wc)

# ===============================
# N-GRAMAS
# ===============================

st.subheader("🔠 N-gram Analysis")

ngram_size = st.selectbox("Selecciona N-grama", [1, 2])
top_k = st.slider("Top N palabras", 5, 30, 15)

if target_filter == "Todos":
    corpus = df["text"].astype(str)
else:
    corpus = df[df["target"] == target_filter]["text"].astype(str)

vectorizer = CountVectorizer(
    ngram_range=(ngram_size, ngram_size),
    stop_words="english"
)

X_counts = vectorizer.fit_transform(corpus)
sum_words = X_counts.sum(axis=0)

words_freq = [
    (word, sum_words[0, idx])
    for word, idx in vectorizer.vocabulary_.items()
]

words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]

ngram_df = pd.DataFrame(words_freq, columns=["Ngram", "Frequency"])

fig_ngram = px.bar(
    ngram_df,
    x="Frequency",
    y="Ngram",
    orientation="h",
    title=f"Top {top_k} {'Unigrams' if ngram_size==1 else 'Bigrams'}"
)

st.plotly_chart(fig_ngram, use_container_width=True)

# ===============================
# CLUSTER ANALYSIS
# ===============================

if "cluster" in df.columns:
    st.subheader("🧩 Cluster Distribution")

    fig_cluster = px.histogram(
        df,
        x="cluster",
        title="Cluster Distribution"
    )

    st.plotly_chart(fig_cluster, use_container_width=True)

# ===============================
# SENTIMENT ANALYSIS
# ===============================

st.subheader("😊 Sentiment Distribution")

fig_sent = px.histogram(
    df,
    x="sentiment_compound",
    nbins=30,
    title="Sentiment Compound Distribution"
)

st.plotly_chart(fig_sent, use_container_width=True)

# ===============================
# HASHTAG ESPECÍFICOS
# ===============================

st.subheader("🔎 Hashtag Feature Explorer")

hashtag_cols = [col for col in df.columns if col.startswith("hashtag_")]

selected_hashtag = st.selectbox("Selecciona hashtag feature", hashtag_cols)

fig_hash_specific = px.histogram(
    df,
    x=selected_hashtag,
    color="target",
    title=f"Distribution of {selected_hashtag}"
)

st.plotly_chart(fig_hash_specific, use_container_width=True)