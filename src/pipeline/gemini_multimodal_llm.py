# src/pipeline/gemini_multimodal_llm.py

import os
import google.generativeai as genai
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeminiMultimodalLLM:
    # UPDATED MODEL NAME HERE
    def __init__(self, model_name: str = "gemini-1.5-flash"): # Changed from "gemini-pro-vision"
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logging.error("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")
            raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logging.info(f"GeminiMultimodalLLM initialized with model: {model_name}")

    def get_image_description(self, image_path: str, prompt: str = "Describe this image.") -> str:
        """
        Generates a description for an image using the configured Gemini multimodal LLM.
        """
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"

        try:
            # Open and convert image to RGB, as some models prefer it
            img = Image.open(image_path).convert('RGB')
            response = self.model.generate_content([prompt, img])
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error generating image description: {e}")
            return f"Error generating image description: {e}"