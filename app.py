# app.py

import os

import sys
import logging
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# ---------------- Load Environment ----------------
load_dotenv()

# ---------------- Add Project Root ----------------
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------- Import Custom Modules ----------------
try:
    from src.pipeline.prediction_pipeline import summarize_pdf_document
    from src.pipeline.gemini_multimodal_llm import GeminiMultimodalLLM
    logging.info("All custom module imports succeeded.")
except ImportError as e:
    logging.exception("Module import failed. Check for invalid 'frontend' references or missing files.")
    raise

# ---------------- Flask App Setup ----------------
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploaded")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_PDF_EXTENSIONS = {'pdf'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

MAX_LLM_INPUT_LENGTH = 5000  # Optional: Limit for LLMs

# ---------------- Helpers ----------------
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# ---------------- Routes ----------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['GET', 'POST'])
def summarize_page():
    summary, error, uploaded_file_path = None, None, None

    if request.method == 'POST':
        file = request.files.get('pdf_file')

        if not file or file.filename == '':
            error = 'No selected file'
        elif allowed_file(file.filename, ALLOWED_PDF_EXTENSIONS):
            filename = secure_filename(file.filename)
            uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(uploaded_file_path)
            logging.info(f"Received PDF for summarization: {uploaded_file_path}")

            try:
                summary = summarize_pdf_document(uploaded_file_path, model_name="llama3-70b-8192")
                if summary.startswith("Error:") or summary.startswith("The PDF is empty"):
                    error, summary = summary, None
            except Exception as e:
                logging.exception("Error during PDF summarization.")
                error = f"Unexpected error: {str(e)}"
            finally:
                if uploaded_file_path and os.path.exists(uploaded_file_path):
                    os.remove(uploaded_file_path)
                    logging.info(f"Deleted uploaded file: {uploaded_file_path}")
        else:
            error = 'Invalid file type. Please upload a PDF.'

    return render_template('summarize.html', summary=summary, error=error)


@app.route('/image_captioning', methods=['GET', 'POST'])
def image_captioning_page():
    llm_response, error, image_display_path = None, None, None
    user_prompt = request.form.get('prompt', "Describe this image in detail.")

    if request.method == 'POST':
        file = request.files.get('image_file')

        if not file or file.filename == '':
            error = 'No image selected.'
        elif not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            error = 'Invalid image type (png, jpg, jpeg, gif allowed).'
        else:
            filename = secure_filename(file.filename)
            uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(uploaded_image_path)
            image_display_path = f"uploaded/{filename}"
            logging.info(f"Image received for LLM processing: {uploaded_image_path}")

            try:
                gemini_llm = GeminiMultimodalLLM()
                llm_response = gemini_llm.get_image_description(uploaded_image_path, prompt=user_prompt)

                if llm_response.startswith("Error:") or llm_response == "No description could be generated for the image.":
                    error, llm_response = llm_response, None
            except ValueError as ve:
                logging.error("Gemini LLM configuration error.")
                error = f"Configuration Error: {ve}. Check your GOOGLE_API_KEY."
            except Exception as e:
                logging.exception("Unexpected error during image captioning.")
                error = f"Unexpected error: {str(e)}"

    return render_template('image_captioning.html',
                           llm_response=llm_response,
                           error=error,
                           image_path=image_display_path,
                           user_prompt=user_prompt)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
