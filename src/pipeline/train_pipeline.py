# src/pipeline/train_pipeline.py
import subprocess

class TrainingPipeline:
    def __init__(self, model_name="tinyllama"):
        self.model_name = model_name

    def preload_model(self):
        try:
            print(f"Pulling Ollama model: {self.model_name}")
            subprocess.run(["ollama", "pull", self.model_name], check=True)
            print(f"Model '{self.model_name}' is ready.")
        except subprocess.CalledProcessError:
            print(f"[ERROR] Could not pull model: {self.model_name}")

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.preload_model()
