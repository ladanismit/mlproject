# Basic libraries for file paths and system-level error handling
import os
import sys
from dataclasses import dataclass

# Importing a bunch of regression models from sklearn, XGBoost, CatBoost
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score              # Evaluation metric
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Custom modules for error logging and exception handling
from src.exception import CustomException
from src.logger import logging

# Utility functions to save model and evaluate all models with grid search
from src.utils import save_object, evaluate_models

# Configuration class to define where the trained model should be saved
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# ModelTrainer class handles the model training pipeline
class ModelTrainer:
    def __init__(self):
        # Load config file which contains path to save the model
        self.model_trainer_config = ModelTrainerConfig()

    # Main function to start the model training process
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            # Separate input features and labels for both train and test
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All columns except last (features)
                train_array[:, -1],   # Last column (target)
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define multiple ML models to try
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameters to tune for each model
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},  # No hyperparameters to tune
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Call a helper function that trains all models using GridSearchCV & returns performance
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Get the best score from model report
            best_model_score = max(sorted(model_report.values()))

            # Get the model name which gave the best score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # If no model performs well (R² < 0.6), throw custom error
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model to file (so you can load it later for predictions)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict using the best model on test set
            predicted = best_model.predict(X_test)

            # Calculate R2 score on test set
            r2_square = r2_score(y_test, predicted)

            logging.info(f"And the best model is : {best_model} with the R2 Score of : {r2_square}")

            return r2_square  # Return final score for reporting

        except Exception as e:
            raise CustomException(e, sys)  # Error handling
