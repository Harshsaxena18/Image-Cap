import streamlit as st
from PIL import Image
from io import BytesIO
from image_captioning import generate_captions
import requests
import numpy as np

def main():
    st.title("Image Captioning App")
    st.write("Upload an image or enter the URL of an image.")

    # Image upload section
    upload_option = st.radio("Select an option", ("Upload Image", "Image URL"))

    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

    else:
        image_url = st.text_input("Enter the URL of an image")
        if image_url:
            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Image from URL", use_column_width=True)
            except:
                st.error("Invalid image URL. Please enter a valid URL.")

    if uploaded_file or image_url:
        captions = generate_captions(image)
        # Display the generated captions in the web interface

if __name__ == '__main__':
    main()
