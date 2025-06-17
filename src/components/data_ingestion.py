# Importing required libraries
import os  # For file and directory operations
import sys  # For system-specific parameters and exception handling
from src.exception import CustomException  # Custom exception class for better error tracing
from src.logger import logging  # Custom logging for tracking the pipeline
import pandas as pd  # For data manipulation

from sklearn.model_selection import train_test_split  # For splitting the dataset
from dataclasses import dataclass  # To create simple classes for holding data/config

# Configuration class using @dataclass to define file paths for raw, train, and test data
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to save training data
    test_data_path: str = os.path.join('artifacts', "test.csv")  # Path to save test data
    raw_data_path: str = os.path.join('artifacts', "data.csv")  # Path to save original/raw data

# Data ingestion class responsible for reading, processing, and saving data
class DataIngestion:
    def __init__(self):
        # Initialize config object with file paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            # STEP 1: Load the original dataset from the given CSV file
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # STEP 2: Create 'artifacts/' directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # STEP 3: Save the raw/original data into artifacts/data.csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # STEP 4: Split the dataset into training (80%) and testing (20%)
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # STEP 5: Save the training data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # STEP 6: Save the testing data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # STEP 7: Return paths to train and test data for use in the next pipeline steps
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Catch any exception and raise it with custom handling
            raise CustomException(e, sys)

# Main entry point for running this module directly
if __name__ == "__main__":
    obj = DataIngestion()  # Create object of DataIngestion class
    train_data, test_data = obj.initiate_data_ingestion()  # Start ingestion process and get output paths
