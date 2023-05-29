import streamlit as st
from PIL import Image
from transformers import pipeline


st.title('Image Captioning App')

uploaded_file = st.file_uploader("Upload Image")
image = Image.open(uploaded_file)

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

caption = image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")

st.write(caption)