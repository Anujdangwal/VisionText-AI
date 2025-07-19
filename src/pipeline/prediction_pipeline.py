import os
import sys
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input


from src.logger import logging
from src.exception import CustomException


# --- PDF Summarization Function ---
def summarize_pdf_document(pdf_path: str, model_name: str = "llama3-8b-8192") -> str:
    """
    Summarizes a PDF document using a Groq LLM.
    """
    from src.pipeline.groq_llm_model import GroqLLM
    from src.utils.pdf_reader import PDFReader

    if not os.path.exists(pdf_path):
        return f"Error: PDF file not found at {pdf_path}"

    try:
        reader = PDFReader(pdf_path)
        text_content = reader.get_text()

        if not text_content.strip():
            return "The PDF is empty or no text could be extracted."

        MAX_LLM_INPUT_LENGTH = 5000
        if len(text_content) > MAX_LLM_INPUT_LENGTH:
            logging.warning(f"Document is very long ({len(text_content)} chars). Truncating.")
            text_content = text_content[:MAX_LLM_INPUT_LENGTH].strip() + "..."

        logging.info("Sending text to LLM for summarization...")
        llm = GroqLLM(model_name=model_name)
        prompt = f"Please provide a concise summary of the following document text:\n\n{text_content}\n\nSummary:"
        summary = llm.get_chat_completion(prompt)

        return summary

    except FileNotFoundError as e:
        return f"Error: {e}"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logging.exception("Error generating LLM completion during PDF summarization.")
        return f"Error generating LLM completion: {e}"
