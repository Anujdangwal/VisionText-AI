import os

root = os.getcwd()
# Define the base artifact folder
ARTIFACT_FOLDER = os.path.join(root,"artifacts")

# Path to the image dataset
IMAGE_DIR = os.path.join(ARTIFACT_FOLDER, r"CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")
# Path to the kidney data CSV file
KIDNEY_CSV_PATH = os.path.join(ARTIFACT_FOLDER, "kidneyData.csv")

# Example
checkpoint_path = os.path.join("model_checkpoint", "best_model.pth")
