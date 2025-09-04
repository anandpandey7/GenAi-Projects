import streamlit as st
import google.generativeai as genai
import os
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()

import os
# Load API key from environment variables (recommended for security)
# Ensure you have GOOGLE_API_KEY set in your environment or a .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_image_with_gemini(prompt_text):
    """
    Generates an image using the Gemini model based on a text prompt.
    """
    try:
        # Use a model capable of image generation (e.g., gemini-2.5-flash-image-preview)
        model = genai.GenerativeModel('gemini-2.5-flash-image-preview')
        response = model.generate_content(prompt_text)

        # Extract the image data from the response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = BytesIO(part.inline_data.data)
                return Image.open(image_data)
        return None # No image found in response
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None
    


st.set_page_config(page_title="Gemini Image Generator", layout="centered")
st.title("Image Generation with Gemini")

# User input for the prompt
input_prompt = st.text_area("Enter a detailed prompt for your image:", height=100)

# Button to trigger image generation
if st.button("Generate Image"):
    if input_prompt:
        with st.spinner("Generating image..."):
            generated_image = generate_image_with_gemini(input_prompt)
            if generated_image:
                st.image(generated_image, caption="Generated Image", use_column_width=True)
            else:
                st.warning("Could not generate image. Please try a different prompt or check the API response.")
    else:
        st.warning("Please enter a prompt to generate an image.")