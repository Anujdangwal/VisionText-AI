import os
import sys
import torch
from torchvision import transforms
from PIL import Image

from src.logger import logging
from src.exception import CustomException
from src.components.training_pipeline import build_model
from src.constant import checkpoint_path

# --- Kidney CT Classification Pipeline ---
class PredictionPipeline:
    """
    Handles image classification prediction using a custom PyTorch DenseNet model.
    Loads a pre-trained model from checkpoint.
    """
    def __init__(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.class_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']
            self.model = build_model(num_classes=len(self.class_labels))

            if os.path.exists(checkpoint_path):
                logging.info(f"Loading classification model from checkpoint: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logging.info("Classification model loaded successfully.")
            else:
                raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")

            self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            logging.info("PredictionPipeline initialized successfully.")

        except Exception as e:
            logging.error("Error during PredictionPipeline initialization", exc_info=True)
            raise CustomException(f"Failed to initialize PredictionPipeline: {e}", sys)

    def predict(self, image_path: str) -> str:
        """
        Performs inference on a given image using the loaded model.
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(image)
                _, predicted = torch.max(output, 1)
                predicted_label = self.class_labels[predicted.item()]

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
