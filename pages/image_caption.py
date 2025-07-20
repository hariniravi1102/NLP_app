import os
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
from huggingface_hub import login

# Load the token from the environment variable
login(st.secrets["HUGGINGFACE_TOKEN"])
# Load model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Streamlit UI
st.title("Image caption")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prepare inputs
    inputs = processor(image, return_tensors="pt")
    inputs = {k: v for k, v in inputs.items()}

    # Generate one caption
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    st.markdown("### Caption:")
    st.write(caption)

if st.button("⬅ back to main page"):
    # Your redirect or logic here
    st.switch_page("main.py")

