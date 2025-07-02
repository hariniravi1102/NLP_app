import streamlit as st
import textblob
from textblob import TextBlob

st.title("📝 Simple NLP App with Streamlit")

text_input = st.text_area("Enter some text", "Streamlit is awesome!")

if st.button("Analyze"):
    blob = TextBlob(text_input)
    sentiment = blob.sentiment
    st.write("**Polarity:**", sentiment.polarity)
    st.write("**Subjectivity:**", sentiment.subjectivity)