# src/pipeline/groq_llm_model.py

import os
from groq import Groq
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GroqLLM:
    def __init__(self, model_name: str = "llama3-8b-8192", temperature: float = 0.5):
        # Retrieve API key from environment variables
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logging.error("GROQ_API_KEY environment variable not set. Please set it in your .env file.")
            raise ValueError("GROQ_API_KEY environment variable not set. Please set it in your .env file.")

        # Initialize Groq client
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        logging.info(f"GroqLLM initialized with model: {self.model_name}")

    def get_chat_completion(self, prompt: str) -> str:
        """
        Gets a chat completion from the Groq LLM based on the given prompt.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
                temperature=self.temperature,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error getting completion from Groq LLM: {e}")
            return f"Error: Could not get LLM response. {e}"