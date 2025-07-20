import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
hf_token = st.secrets["HF_TOKEN"]
login(token=hf_token)
# Load Mistral model and tokenizer (public, no gated access required)
@st.cache_resource
def load_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, use_auth_token=hf_token)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

generator = load_model()

# Streamlit UI
st.set_page_config(page_title="AI Article Generator", layout="wide")

st.title("AI Article Generator")
st.write("Generate high-quality, SEO-friendly articles")

topic = st.text_input("Enter an article topic:", placeholder="e.g., The Rise of Green Technology")

# Text input for word count
word_count_str = st.text_input("Enter desired word count:", value="800")

if st.button("Generate Article"):
    if not topic:
        st.warning("⚠️ Please enter a topic to generate an article.")
    else:
        try:
            word_count = int(word_count_str)
            if word_count <= 0:
                raise ValueError
        except ValueError:
            st.error("❌ Word count must be a positive integer.")
        else:
            with st.spinner("Generating article..."):
                prompt = f"""You are an expert content writer. Write a well-researched, SEO-friendly article about: "{topic}".
The article should be approximately {word_count} words, include an engaging introduction, multiple H2 subheadings, and a clear conclusion.
Avoid repetition. Focus on clarity, structure, and real-world examples."""

                output = generator(
                    prompt,
                    max_new_tokens=int(word_count * 1.3),
                    do_sample=True,
                    temperature=0.7
                )[0]['generated_text']

            st.success("✅ Article generated!")
            st.markdown("---")
            st.markdown(output)
            st.download_button("📥 Download as .txt", output, file_name="article.txt")

if st.button("⬅ back to main page"):
    # Your redirect or logic here
    st.switch_page("main.py")
