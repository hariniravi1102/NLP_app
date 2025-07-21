import streamlit as st
import torch
import nltk
from nltk.tokenize import sent_tokenize


#import subprocess
from transformers import T5ForConditionalGeneration, T5Tokenizer
# Load the model and tokenizer
model_name = "deep-learning-analytics/GrammarCorrector"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
# load function for grammar
def correct_grammar(text):
    # Preprocess input for T5 model
    input_text = text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    #input_ids = input_ids.to(device)
    # Generate correction
    outputs = model.generate(input_ids, max_length=1000, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return corrected_text

st.title("Grammatical check")

# Text input box
user_prompt = st.text_area("Enter the text for grammatical error:", height=300)


# Function to chunk and correct an entire essay
def correct_long_text(text):
    sentences = sent_tokenize(text)
    corrected_sentences = []
    for sentence in sentences:
        if sentence.strip():
            try:
                corrected = correct_grammar(sentence)
                corrected_sentences.append(corrected)
            except Exception as e:
                corrected_sentences.append(sentence)  # fallback to original if error
    return ' '.join(corrected_sentences)

# check gpu usage
#if torch.cuda.is_available():
#    gpu_info = subprocess.check_output(["nvidia-smi"]).decode("utf-8")#
#    st.text("nvidia-smi GPU status:")
#    st.text(gpu_info)

# Generate and display image
if st.button("Enter"):
    if user_prompt:
        text = correct_long_text(user_prompt)
        st.text_area(
            label="Output",
            value= text,
            height=300,
            max_chars=None,
            key="output_text_area",
            disabled=True,  # disables editing, making it read-only
            help="This is your corrected sentence output."
        )
    else:
        st.warning("Please enter some text before pressing Enter.")


if st.button("⬅ back to main page"):
    # Your redirect or logic here
    st.switch_page("main.py")
