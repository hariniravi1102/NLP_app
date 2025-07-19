# ai_paper_explorer_streamlit.py

import streamlit as st
import feedparser
import urllib.parse
import torch
import requests
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Load Hugging Face embedding model
@st.cache_resource

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

def get_embedding(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.pooler_output.mean(dim=0)

def search_arxiv(query, max_results=15):
    base_url = "http://export.arxiv.org/api/query"
    encoded_query = urllib.parse.quote(query)
    search_url = f"{base_url}?search_query=all:{encoded_query}&start=0&max_results={max_results}"
    feed = feedparser.parse(search_url)
    return feed.entries

def plot_similarity_graph(papers, sim_matrix):
    G = nx.Graph()
    titles = [p['title'] for p in papers]
    for i, title in enumerate(titles):
        G.add_node(title)
        for j in range(i+1, len(titles)):
            if sim_matrix[i, j] > 0.7:
                G.add_edge(title, titles[j], weight=sim_matrix[i, j])

    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(12, 6))
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=7, ax=ax)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# --- UI ---
st.set_page_config(page_title="Research Paper Explorer", layout="wide")
st.title("📚 Research Paper Explorer")

query = st.text_input("Enter your research topic:", "graph neural networks for drug discovery")
max_results = st.slider("Number of papers to fetch:", 5, 50, 15)
show_graph = st.checkbox("Show similarity graph", value=True)

if st.button("Search"):
    with st.spinner("Fetching and ranking papers..."):
        tokenizer, model = load_model()
        papers = search_arxiv(query, max_results=max_results)

        abstracts = [p.summary.replace('\n', ' ') for p in papers]
        embeddings = torch.stack([get_embedding(a, tokenizer, model) for a in abstracts])

        query_vec = get_embedding(query, tokenizer, model).unsqueeze(0)
        sims = cosine_similarity(query_vec, embeddings)[0]

        # Sort papers by similarity
        ranked = sorted(zip(sims, papers), key=lambda x: x[0], reverse=True)

        st.subheader("📄 Top Relevant Papers")
        for score, paper in ranked:
            st.markdown(f"### [{paper.title}]({paper.link})")
            st.markdown(f"**Authors:** {paper.author}")
            st.markdown(f"**Published:** {paper.published.split('T')[0]}")
            st.markdown(f"**Relevance Score:** {score:.2f}")
            st.markdown(paper.summary[:500] + "...")
            st.markdown("---")

        if show_graph:
            st.subheader("📊 Similarity Graph Between Papers")
            sim_matrix = cosine_similarity(embeddings)
            buf = plot_similarity_graph(papers, sim_matrix)
            st.image(buf, use_column_width=True)
