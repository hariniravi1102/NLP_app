import os
import json
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score

# Load Mistral model and tokenizer with GPU support
@st.cache_resource
def load_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    hf_token = os.getenv("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")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=hf_token
    )

    return tokenizer, model

# Function to generate text using GPU
def generate_text(prompt, tokenizer, model, max_tokens=1024, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Feedback logging
def log_feedback(prompt, output, feedback):
    score = 1 if feedback == "👍 Good" else 0
    with open("feedback_data.jsonl", "a") as f:
        json.dump({"prompt": prompt, "output": output, "score": score}, f)
        f.write("\n")

def log_human_evaluation(prompt, output, scores):
    data = {
        "prompt": prompt,
        "output": output,
        "fluency": scores["fluency"],
        "coherence": scores["coherence"],
        "factuality": scores["factuality"],
        "relevance": scores["relevance"]
    }
    with open("human_feedback.jsonl", "a") as f:
        json.dump(data, f)
        f.write("\n")

# Evaluation Metrics
def evaluate_metrics(reference, generated):
    metrics = {}
    try:
        metrics['BLEU'] = sentence_bleu([reference.split()], generated.split())
    except:
        metrics['BLEU'] = 0.0
    try:
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = rouge.score(reference, generated)
        metrics['ROUGE-1'] = scores['rouge1'].fmeasure
        metrics['ROUGE-L'] = scores['rougeL'].fmeasure
    except:
        metrics['ROUGE-1'] = metrics['ROUGE-L'] = 0.0
    try:
        P, R, F1 = score([generated], [reference], lang="en", verbose=False)
        metrics['BERTScore F1'] = F1[0].item()
    except:
        metrics['BERTScore F1'] = 0.0
    return metrics

# UI setup
st.set_page_config(page_title="AI Article Generator with Mistral", layout="wide")
st.title("🧠 Mistral-Powered AI Article Generator")

if torch.cuda.is_available():
    st.success(f"✅ Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    st.warning("⚠️ Running on CPU — generation may be slower.")

tokenizer, model = load_model()

topic = st.text_input("Enter an article topic:", placeholder="e.g., The Rise of Green Technology")
word_count_str = st.text_input("Enter desired word count:", value="800")
reference = st.text_area("Optional Reference Output (for BLEU, ROUGE, BERTScore)", "")

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
                prompt = f"""You are an expert content writer. Write a well-researched, SEO-friendly article about: \"{topic}\".
The article should be approximately {word_count} words, include an engaging introduction, multiple H2 subheadings, and a clear conclusion.
Avoid repetition. Focus on clarity, structure, and real-world examples.
Format the article using Markdown with proper headings (##), paragraphs, and bullet points where helpful."""
                output = generate_text(prompt, tokenizer, model, max_tokens=int(word_count * 1.5))

            st.success("✅ Article generated!")
            st.markdown("---")
            st.markdown(output)
            st.download_button("📥 Download as .txt", output, file_name="article.txt")


            if reference:
                st.subheader("📊 Evaluation Metrics")
                metrics = evaluate_metrics(reference, output)
                for k, v in metrics.items():
                    st.write(f"**{k}**: {v:.4f}")

            st.subheader("👥 Human Rating")

            # Human Feedback and Rating Form
            with st.form("feedback_and_rating_form"):
                st.subheader("📣 Feedback and Human Rating")

                # Set defaults
                feedback = st.radio(
                    "How would you rate this article?",
                    ["👍 Good", "👎 Bad"]
                )
                fluency = st.slider("Fluency", 0, 5, 3)
                coherence = st.slider("Coherence", 0, 5, 3)
                factuality = st.slider("Factuality", 0, 5, 3)
                relevance = st.slider("Relevance", 0, 5, 3)

                submitted = st.form_submit_button("Submit Feedback & Ratings")
                if submitted:
                    scores = {
                        "fluency": fluency,
                        "coherence": coherence,
                        "factuality": factuality,
                        "relevance": relevance
                    }
                    log_feedback(prompt, output, feedback, scores)
                    st.success("✅ Feedback and ratings submitted!")
                    st.json(scores)
