import os
import sys
from dataclasses import dataclass
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.logger import logging
from src.exception import CustomException
from src.constant import IMAGE_DIR

@dataclass
class DataTransformationConfig:
    image_dir: str = IMAGE_DIR
    target_size: tuple = (256, 256)
    batch_size: int = 32
    validation_split: float = 0.2

class ImageDataPipeline:
    def __init__(self, image_dir):
        self.config = DataTransformationConfig(image_dir=IMAGE_DIR)

        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.config.validation_split
        )

    def get_generators(self):
        try:
            logging.info("Creating training and validation generators.")

            train_generator = self.train_datagen.flow_from_directory(
                directory=self.config.image_dir,
                target_size=self.config.target_size,
                batch_size=self.config.batch_size,
                class_mode='categorical'
            )

            val_generator = self.train_datagen.flow_from_directory(
                directory=self.config.image_dir,
                target_size=self.config.target_size,
                batch_size=self.config.batch_size,
                class_mode='categorical',
                subset='validation'
            )

            return train_generator, val_generator

        except Exception as e:
            logging.error("Error in data transformation", exc_info=True)
            raise CustomException(e, sys)
