import os
import nltk

import shutil
import streamlit as st
# ✅ Ensure correct 'punkt' tokenizer is present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    with st.spinner("Downloading NLTK 'punkt' tokenizer..."):
        nltk.download("punkt")