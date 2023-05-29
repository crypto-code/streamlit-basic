import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


st.title('Image Captioning App')

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    inputs = processor(image, return_tensors="pt")

    out = model.generate(**inputs)
    st.write(processor.decode(out[0], skip_special_tokens=True))
