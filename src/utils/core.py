# Handling Outliers & Outlier treatments
import pandas as pd
import numpy as np

def find_outliers_iqr(data):
    # Calculate the first quartile (Q1) and third quartile (Q3) for each column
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    # Calculate the interquartile range (IQR) for each column
    iqr = q3 - q1

    # Calculate the lower and upper bounds for outliers for each column
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Check for outliers in each column and count the number of outliers
    outliers_count = (data < lower_bound) | (data > upper_bound)
    num_outliers = outliers_count.sum()

    return num_outliers


def define_beste_model(scores: list[dict]):
    """
    Bepaalt het beste model op basis van R², RMSE en MAE.

    Hoger R² = beter.
    Lager RMSE/MAE = beter.
    We combineren deze in één genormaliseerde score.
    """

    if not scores:
        raise ValueError("De lijst 'scores' is leeg.")

    # Extract all RMSE/MAE values to compute scaling
    all_rmse = [s["RMSE"] for s in scores]
    all_mae = [s["MAE"] for s in scores]

    rmse_max, rmse_min = max(all_rmse), min(all_rmse)
    mae_max, mae_min = max(all_mae), min(all_mae)

    results = []

    for s in scores:
        model = s["Model"]
        r2 = s["R² Score"]
        rmse = s["RMSE"]
        mae = s["MAE"]

        # Normaliseer RMSE en MAE (tussen 0 en 1, lager = beter)
        rmse_norm = 1 - ((rmse - rmse_min) / (rmse_max - rmse_min))
        mae_norm = 1 - ((mae - mae_min) / (mae_max - mae_min))

        # Combineer in één score
        combined = (r2 + rmse_norm + mae_norm) / 3

        results.append({"Model": model, "CombinedScore": combined})

    # Beste model op basis van hoogste gecombineerde score
    best_model = max(results, key=lambda x: x["CombinedScore"])

    return best_model["Model"], best_model["CombinedScore"]

"""
This model is designed to predict appliance energy consumption based on various temperature and humidity features from different rooms and outside conditions. The model uses polynomial features to capture non-linear relationships in the data. The predictions are made on a sample of the input data, and the results are printed for review.
Created by: Nova Energy Research & Development
in Oktober, 2025

"""