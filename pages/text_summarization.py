import streamlit as st
from transformers import pipeline

# Load the summarization model
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# Streamlit UI
st.title("Text Summarization App")
st.write("Summarize long text")

# User input
input_text = st.text_area("Enter text to summarize", height=250)

# Summarize when button is clicked
if st.button("Summarize"):
    if input_text.strip():
        with st.spinner("Summarizing..."):
            summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
            st.subheader("Summary")
            st.success(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text.")

if st.button("⬅ back to main page"):
    # Your redirect or logic here
    st.switch_page("main.py")