
from .utils.overview import cleaning_data, report_data, generate_report
from .data.process_data import preprocess
from .models.train_models import (
    train_all_models, get_models_accuracy,
    load_model, load_and_predict, visual_outputs
)
from .reports.make_report import generate_report, main_make_report

"""
This model is designed to predict appliance energy consumption based on various temperature and humidity features from different rooms and outside conditions. The model uses polynomial features to capture non-linear relationships in the data. The predictions are made on a sample of the input data, and the results are printed for review.
Created by: Ahmad Al Dibo
in Oktober, 2025

Duurt een week werk + 8 uur elke dag
"""