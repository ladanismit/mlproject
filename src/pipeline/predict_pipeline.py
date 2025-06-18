# Standard library imports
import sys
import os
import pandas as pd

# Custom error and utility functions
from src.exception import CustomException
from src.utils import load_object  # Used to load pickled models or preprocessors


class PredictPipeline:
    def __init__(self):
        pass  # No setup needed in constructor for now

    def predict(self, features):
        """
        Takes processed user features (in DataFrame format), applies the saved preprocessor,
        and then uses the saved model to make predictions.
        """
        try:
            # Define paths to saved model and preprocessor files
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading")

            # Load the serialized (pickled) model and preprocessor objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading")

            # Apply preprocessing (scaling, encoding) to input features
            data_scaled = preprocessor.transform(features)

            # Make predictions using the trained model
            preds = model.predict(data_scaled)

            return preds  # Returns predictions as NumPy array

        except Exception as e:
            # Custom exception handling
            raise CustomException(e, sys)


class CustomData:
    """
    A class used to collect and organize custom input from a user or form (e.g., in a web app)
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education :str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        # Assign input values to object attributes
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    def get_data_as_data_frame(self):
        """
        Converts the user inputs into a single-row pandas DataFrame,
        which is the expected format for the prediction pipeline.
        """
        try:
            # Create a dictionary from the input values
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert to pandas DataFrame so it can be fed into the pipeline
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
