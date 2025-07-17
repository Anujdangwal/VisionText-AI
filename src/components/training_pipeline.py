import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.logger import logging
from src.exception import CustomException
from src.constant import checkpoint_path, IMAGE_DIR


from torchvision.models import densenet121, DenseNet121_Weights

def build_model(num_classes):
    # Use updated weights API
    weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model



class TrainingPipeline:
    def __init__(self, image_dir=IMAGE_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_dir = image_dir
        self.batch_size = 32
        self.epochs = 10
        self.model_save_path = checkpoint_path

    def start_training(self):
        try:
            # Transformations
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            # Load Dataset
            dataset = ImageFolder(self.image_dir, transform=transform)
            class_names = dataset.classes
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Build Model
            model = build_model(num_classes=len(class_names)).to(self.device)

            # Loss and Optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training Loop
            best_loss = float('inf')
            for epoch in range(self.epochs):
                model.train()
                running_loss, correct = 0.0, 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    correct += (outputs.argmax(1) == labels).sum().item()

                epoch_loss = running_loss / len(train_loader.dataset)
                epoch_acc = correct / len(train_loader.dataset)

                logging.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

                # Save best model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), self.model_save_path)
                    logging.info(f"Best model saved at {self.model_save_path}")

            return model

        except Exception as e:
            logging.error("Training failed", exc_info=True)
            raise CustomException(e, sys)


# train_pipeline.py

import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def train_and_save_model(model_name="google/pegasus-xsum", save_dir="artifacts/pegasus-xsum"):
    if os.path.exists(save_dir):
        print(f"[INFO] Model already saved at: {save_dir}")
        return

    print("[INFO] Downloading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print("[INFO] Saving model to artifacts...")
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    print(f"[SUCCESS] Model saved to: {save_dir}")

if __name__ == "__main__":
    train_and_save_model()
