import os

root = os.getcwd()
# Define the base artifact folder
ARTIFACT_FOLDER = os.path.join(root,"artifacts")


# Example
checkpoint_path = os.path.join(root , "model_checkpoint", "best_model.h5")
