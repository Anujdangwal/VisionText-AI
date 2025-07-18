# app.py
import os
TF_ENABLE_ONEDNN_OPTS=0

import sys
import logging
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables (including GROQ_API_KEY and GOOGLE_API_KEY)
load_dotenv()

# Add project root to path for absolute imports from 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our custom classes/functions
try:
    from src.components.pdf_reader import PDFReader
    from src.pipeline.groq_llm_model import GroqLLM
    from src.pipeline.prediction_pipeline import PredictionPipeline, summarize_pdf_document
    from src.pipeline.gemini_multimodal_llm import GeminiMultimodalLLM
except ImportError as e:
    logging.error(f"Failed to import necessary components. Check your 'src' folder structure and file contents. Error: {e}")
    sys.exit(1)

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploaded")
ALLOWED_PDF_EXTENSIONS = {'pdf'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_LLM_INPUT_LENGTH = 5000

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Home route - now renders index.html
@app.route('/')
def home():
    return render_template('index.html')

from src.pipeline.prediction_pipeline import PredictionPipeline
classifier = PredictionPipeline() 
logging.info("PredictionPipeline initialized successfully.")

# --- CT Kidney Classification Route ---
@app.route('/classify', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        try:
            file = request.files.get('file')
            if not file or file.filename == '':
                logging.warning("No file uploaded.")
                return render_template('classify.html', prediction="No file selected.")

            # Save uploaded image
            filename = secure_filename(file.filename)
            save_dir = os.path.join('static', 'uploads')
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, filename)
            file.save(file_path)

            # ✅ Use global classifier instance (model already loaded once)
            prediction = classifier.predict(file_path)

            # image_path for HTML must be relative to 'static' directory
            relative_path = f"uploads/{filename}"

            return render_template('classify.html', prediction=prediction, image_path=relative_path)

        except Exception as e:
            logging.error("Prediction failed", exc_info=True)
            return render_template('classify.html', prediction=f"Error: {str(e)}")

    return render_template('classify.html')


@app.route('/summarize', methods=['GET', 'POST'])
def summarize_page():
    summary = None
    error = None
    uploaded_file_path = None

    if request.method == 'POST':
        file = request.files['pdf_file']

        if file.filename == '':
            error = 'No selected file'
            return render_template('summarize.html', summary=summary, error=error)

        if file and allowed_file(file.filename, ALLOWED_PDF_EXTENSIONS):
            filename = secure_filename(file.filename)
            uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(uploaded_file_path)
            logging.info(f"Received PDF for summarization: {uploaded_file_path}")

            try:
                summary = summarize_pdf_document(uploaded_file_path, model_name="llama3-70b-8192")
                if summary.startswith("Error:") or summary.startswith("The PDF is empty"):
                    error = summary
                    summary = None
            except Exception as e:
                logging.exception("An unexpected error occurred during summarization.")
                error = f"An unexpected error occurred during summarization: {str(e)}"
            finally: # This finally block is correctly placed for the try above
                if uploaded_file_path and os.path.exists(uploaded_file_path):
                    os.remove(uploaded_file_path) # You might want to remove this if you need PDF files to persist for longer display or debugging
                    logging.info(f"Cleaned up uploaded file: {uploaded_file_path}")
        else: # This else corresponds to 'if file and allowed_file'
            error = 'Invalid file type. Please upload a PDF.'
        # The return here ensures the page is rendered after processing the POST request
        return render_template('summarize.html', summary=summary, error=error)
    # This return handles the initial GET request for the page
    return render_template('summarize.html', summary=summary, error=error)

@app.route('/image_captioning', methods=['GET', 'POST'])
def image_captioning_page():
    llm_response = None
    error = None
    image_display_path = None
    user_prompt = "Describe this image in detail." # Default prompt

    if request.method == 'POST':
        user_prompt = request.form.get('prompt', user_prompt)
        file = request.files.get('image_file')

        if not file or file.filename == '':
            error = 'No image selected.'
            return render_template('image_captioning.html', llm_response=llm_response, error=error, user_prompt=user_prompt)

        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            error = 'Invalid file type. Please upload an image (png, jpg, jpeg, gif).'
            return render_template('image_captioning.html', llm_response=llm_response, error=error, user_prompt=user_prompt)

        filename = secure_filename(file.filename)
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_image_path)
        logging.info(f"Received image for LLM processing: {uploaded_image_path}")

        # FIX: Ensure forward slashes for the URL path
        image_display_path = f"uploaded/{filename}" 

        try:
            gemini_llm = GeminiMultimodalLLM()
            llm_response = gemini_llm.get_image_description(uploaded_image_path, prompt=user_prompt)

            if llm_response.startswith("Error:"):
                error = llm_response
                llm_response = None
            elif llm_response == "No description could be generated for the image.":
                error = llm_response
                llm_response = None

        except ValueError as ve:
            logging.error(f"Configuration error for Gemini LLM: {ve}")
            error = f"Configuration Error: {ve}. Please check your GOOGLE_API_KEY."
            llm_response = None
        except Exception as e:
            logging.exception("An unexpected error occurred during multimodal LLM processing.")
            error = f"An unexpected error occurred during processing: {str(e)}"
            llm_response = None
        finally: # This finally block is correctly associated with the try block above
            # --- IMPORTANT: Keep this block commented out to display image ---
            # if uploaded_image_path and os.path.exists(uploaded_image_path):
            #     os.remove(uploaded_image_path)
            #     logging.info(f"Cleaned up uploaded file: {uploaded_image_path}")
            pass # This pass ensures the finally block is not empty

    return render_template('image_captioning.html', llm_response=llm_response, error=error, image_path=image_display_path, user_prompt=user_prompt)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)