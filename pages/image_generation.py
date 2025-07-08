from diffusers import StableDiffusionPipeline
import torch
print(torch.__version__)               # Should show 2.6.0
print(torch.cuda.is_available())       # Should be True
print(torch.cuda.get_device_name(0))   # Should print your GPU name

import streamlit as st
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

def generate_image(prompt):
    with torch.autocast("cuda"):
        image = pipe(prompt, num_inference_steps=30).images[0]
    return image

st.title("Text-to-Image Generator")

# Text input box
user_prompt = st.text_input("Text to image generation:")



# Generate and display image
if st.button("Enter"):
    if user_prompt:
        st.write("Generating image...")
        img = generate_image(user_prompt)
        st.image(img, caption=f"Generated for: {user_prompt}", use_column_width=True)
        # Here you can add your image generation code using user_prompt
    else:
        st.warning("Please enter some text before pressing Enter.")

#


if st.button("⬅ back to main page"):
    # Your redirect or logic here
    st.switch_page("main.py")
