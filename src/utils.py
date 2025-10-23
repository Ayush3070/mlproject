import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Saves any Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Trains and evaluates multiple regression models with hyperparameter tuning.
    Returns a dictionary containing model names and their R² test scores.
    """
    try:
        report = {}

        for name, model in models.items():
            print(f"\n🔍 Training and tuning model: {name} ...")
            param_grid = params.get(name, {})

            # Perform Grid Search only if params exist
            if len(param_grid) > 0:
                gs = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Model scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            print(f"✅ {name} -> Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
            report[name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
