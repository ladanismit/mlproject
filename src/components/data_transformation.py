# Standard libraries
import sys
from dataclasses import dataclass

# Data manipulation and numerical computing
import numpy as np 
import pandas as pd

# Scikit-learn tools for preprocessing
from sklearn.compose import ColumnTransformer              # Used to apply different preprocessing to different columns
from sklearn.impute import SimpleImputer                   # Handles missing values
from sklearn.pipeline import Pipeline                      # Chains multiple preprocessing steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Encodes categorical data and scales features

# Custom modules for exception handling and logging
from src.exception import CustomException
from src.logger import logging
import os

# Utility function to save Python objects like preprocessor pipeline
from src.utils import save_object

# Configuration class for storing transformation-related paths
@dataclass
class DataTransformationConfig:
    # Path to save the preprocessor object (like pipeline) after building
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")

# Main class to handle data transformation
class DataTransformation:
    def __init__(self):
        # Load the config for storing preprocessor file path
        self.data_transformation_config = DataTransformationConfig()

    # Function that creates and returns a preprocessor pipeline object
    def get_data_transformer_object(self):
        '''
        This function is responsible for creating the preprocessing pipeline
        for both numerical and categorical features.
        '''
        try:
            # Define columns that need different preprocessing
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numerical pipeline: fill missing values with median, then scale
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline: fill missing with most frequent, one-hot encode, then scale
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),  # Converts categorical data into binary columns
                    ("scaler", StandardScaler(with_mean=False))  # Scale the one-hot encoded values
                ]
            )

            # Logging for debugging and tracking
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor  # Return the final pipeline object

        except Exception as e:
            raise CustomException(e, sys)  # Raise custom error if something breaks

    # Function to initiate transformation process on train and test data
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # STEP 1: Read training and test datasets from provided file paths
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # STEP 2: Create the preprocessing pipeline
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Define the name of the target variable (the label we want to predict)
            target_column_name = "math_score"

            # STEP 3: Separate input features and target for train set
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Same separation for test set
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # STEP 4: Apply the preprocessor
            logging.info("Applying preprocessing object on training and testing data")

            # Fit on training data and transform it
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            # Only transform the test data (do NOT fit on test data)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # STEP 5: Combine the transformed input features with target variable
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # STEP 6: Save the fitted preprocessor object for use during inference
            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed train & test arrays + path to the saved preprocessor
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)  # Custom error handling if anything breaks
