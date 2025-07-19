# Streamlit UI for RLHF Article Generator
import os
import json
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score

# ✅ Load HF token safely from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this in your environment


@st.cache_resource
def load_model(model_version="base"):
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

    model_path = model_name if model_version == "base" else "ppo_model"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=HF_TOKEN if model_version == "base" else None
    )
    return tokenizer, model


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


# ✅ Unified logging function
def log_feedback(prompt, output, feedback=None, scores=None):
    print("📝 Logging feedback function called")
    data = {
        "prompt": prompt,
        "output": output,
    }
    if feedback:
        data["feedback"] = feedback
        data["score"] = 1 if feedback == "👍 Good" else 0
    if scores:
        data.update(scores)

    try:
        with open("feedback_data.jsonl", "a", encoding="utf-8") as f:
            json.dump(data, f)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        print("✅ Feedback saved:", data)
    except Exception as e:
        print("❌ Error saving feedback:", e)
        st.error(f"Error saving feedback: {e}")

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


# 🖼️ Streamlit UI Setup
st.set_page_config(page_title="AI Article Generator with Mistral RLHF", layout="wide")
st.title("🧠 Mistral-Powered AI Article Generator with RLHF")

if torch.cuda.is_available():
    st.success(f"✅ Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    st.warning("⚠️ Running on CPU — generation may be slower.")

model_choice = st.radio("Choose Model Version", ["Base", "Fine-Tuned (RLHF)"])
tokenizer, model = load_model("base" if model_choice == "Base" else "rlhf")

topic = st.text_input("Enter an article topic:", placeholder="e.g., The Rise of Green Technology")
word_count_str = st.text_input("Enter desired word count:", value="800")
reference = st.text_area("Optional Reference Output (for BLEU, ROUGE, BERTScore)", "")
st.write("Current working directory:", os.getcwd())
st.write("File exists:", os.path.exists("feedback_data.jsonl"))
st.write("Writable:", os.access("feedback_data.jsonl", os.W_OK))

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
                max_tokens = min(int(word_count * 1.5), 2048)
                output = generate_text(prompt, tokenizer, model, max_tokens=max_tokens)

            st.success("✅ Article generated!")
            st.markdown("---")
            st.markdown(output)
            st.download_button("📥 Download as .txt", output, file_name="article.txt")
            if reference.strip():
                st.subheader("📊 Evaluation Metrics")
                try:
                    metrics = evaluate_metrics(reference, output)
                    if metrics:
                        for k, v in metrics.items():
                            st.write(f"**{k}**: {v:.4f}")
                    else:
                        st.write("No metrics computed.")
                except Exception as e:
                    st.write("⚠️ Error computing metrics:", e)
            if st.button("🔍 Test Log Write"):
                try:
                    test_data = {"prompt": "test", "output": "test output", "feedback": "👍 Good", "score": 1}
                    with open("feedback_data.jsonl", "a", encoding="utf-8") as f:
                        json.dump(test_data, f)
                        f.write("\n")
                        f.flush()
                        os.fsync(f.fileno())
                    st.success("✅ Test write succeeded")
                except Exception as e:
                    st.error(f"❌ Test write failed: {e}")
            st.subheader("👥 Human Rating")
            with st.form("feedback_and_rating_form"):
                st.subheader("📣 Feedback and Human Rating")
                feedback = st.radio("How would you rate this article?", ["👍 Good", "👎 Bad"])
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
                log_feedback(prompt, output, feedback=feedback, scores=scores)
                st.success("✅ Feedback & ratings logged!")
                st.experimental_set_query_params()
                st.session_state.clear()
                st.rerun()