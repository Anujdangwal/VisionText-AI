import os
import sys
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input


from src.logger import logging
from src.exception import CustomException
from src.constant import checkpoint_path


# --- Kidney CT Classification Pipeline (TensorFlow Version) ---
class PredictionPipeline:
    """
    Handles image classification prediction using a custom TensorFlow DenseNet model.
    Loads a pre-trained model from .h5 or SavedModel format.
    """
    def __init__(self):
        try:
            self.class_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']
            self.image_size = (256, 256)
            self.checkpoint_path = checkpoint_path

            if os.path.exists(self.checkpoint_path):
                logging.info(f"Loading classification model from: {self.checkpoint_path}")
                self.model = load_model(self.checkpoint_path)
                logging.info("TensorFlow classification model loaded successfully.")
            else:
                raise FileNotFoundError(f"Model not found at {self.checkpoint_path}. Please train and export the model.")

        except Exception as e:
            logging.error("Error during PredictionPipeline initialization", exc_info=True)
            raise CustomException(f"Failed to initialize PredictionPipeline: {e}", sys)

    def predict(self, image_path: str) -> str:
        """
        Performs inference on a given image using the loaded TensorFlow model.
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image = image.resize(self.image_size)
            image = img_to_array(image)
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Run inference
            predictions = self.model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = self.class_labels[predicted_class]

            logging.info(f"Predicted label for {os.path.basename(image_path)}: {predicted_label}")
            return predicted_label

        except Exception as e:
            logging.error(f"Prediction failed for image: {image_path}", exc_info=True)
            raise CustomException(f"Prediction failed: {e}", sys)


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
