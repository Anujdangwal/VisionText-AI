import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.constant import *
from src.exception import CustomException  # Make sure it's imported

@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(ARTIFACT_FOLDER)

class DataIngestion:

    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.config = DataIngestionConfig() 

    def export_data_as_dataframe(self):
        try:
            logging.info(f"Importing data from: {self.csv_file_path}")
            
            df = pd.read_csv(self.csv_file_path)  # ✅ Correct function
            logging.info("Data loaded successfully as DataFrame")
            return df

        except Exception as e:
            logging.error("Error loading CSV file", exc_info=True)
            raise CustomException(str(e), sys)

obj = DataIngestion(KIDNEY_CSV_PATH)

print(obj.export_data_as_dataframe())
