# ===============================================================
# ENERGY CONSUMPTION MODEL TRAINING PIPELINE
# ===============================================================
# Created by: Nova Energy Research & Development
# Date: October 2025
# Description:
# Trains multiple regression models to predict appliance energy
# usage based on temperature and humidity features.
# ===============================================================

# Imports
import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Regression Models








def train_all_models(models:dict, training:tuple, folder_output:str="Data/output/models/", log:bool=True) -> tuple:
    """
    training (X_train, X_test, y_train, y_test)
    X_train = 0
    X_test = 1
    y_train = 2
    y_test = 3
    
    """
    X_train = training[0]
    X_test = training[1]
    y_train = training[2]
    y_test = training[3]

    results = []

    for name, model in models.items():
        if log: logging.info(f"Training model: {name}")
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            raise ValueError(f"error: {e}")

        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Save model
        model_path = os.path.join(folder_output, f"{name}_model.pkl")
        joblib.dump(model, model_path)

        results.append({
            "Model": name,
            "R² Score": round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4)
        })


    results_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
    return results_df, str(model_path)


def get_models_accuracy(models:dict, training:tuple, log:bool=True) -> tuple:
    """

    
    """
    X_train = training[0]
    X_test = training[1]
    y_train = training[2]
    y_test = training[3]

    results = []

    for name, model in models.items():
        if log: logging.info(f"Calculating accuracy for model: {name}")

        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)


        results.append({
            "Model": name,
            "R² Score": round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4)
        })


    results_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
    return results, results_df


def load_and_predict(model_path, X_test):
    model = joblib.load(model_path)
    return model.predict(X_test)

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def visual_outputs(rs:pd.DataFrame, save_path, filename:str="figuur.png"):
    try:
        plt.figure(figsize=(10, 5))
        plt.bar(rs["Model"], rs["R² Score"], color='skyblue')
        plt.xticks(rotation=45)
        plt.ylabel("R² Score")
        plt.title("Model Performance Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename))
        plt.show()
    except Exception as e:
        raise ValueError(f"Error: {e}")


"""
This model is designed to predict appliance energy consumption based on various temperature and humidity features from different rooms and outside conditions. The model uses polynomial features to capture non-linear relationships in the data. The predictions are made on a sample of the input data, and the results are printed for review.
Created by: Nova Energy Research & Development
in Oktober, 2025

"""