# to run this use: python -m src.reports.make_report

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "src.reports"

from .. import report_data, generate_report


import pandas as pd

column_mapping = {'T1': 'KITCHEN_TEMP',
    'RH_1': 'KITCHEN_HUM',
    'T2': 'LIVING_TEMP',
    'RH_2' :'LIVING_HUM',
    'T3': 'BEDROOM_TEMP',
    'RH_3':'BEDROOM_HUM',
    'T4' : 'OFFICE_TEMP',
    'RH_4' : 'OFFICE_HUM',
    'T5' : 'BATHROOM_TEMP',
    'RH_5': 'BATHROOM_HUM',
    'T6':'OUTSIDE_TEMP_build',
    'RH_6': 'OUTSIDE_HUM_build',
    'T7': 'IRONING_ROOM_TEMP',
    'RH_7' : 'IRONING_ROOM_HUM',
    'T8' :'TEEN_ROOM_2_TEMP',
    'RH_8' : 'TEEN_ROOM_HUM',
    'T9': 'PARENTS_ROOM_TEMP',
    'RH_9': 'PARENTS_ROOM_HUM',
    'T_out' :'OUTSIDE_TEMP_wstn',
    'RH_out' :'OUTSIDE_HUM_wstn'
}


def main_make_report():
    data_raw = pd.read_csv("Data/raw/KAG_energydata_complete.csv")
    data = report_data(data_raw, column_mapping)[0]

    path = generate_report(data, "Data/reports/plots/")
    print(path)

if __name__ == "__main__":
    main_make_report()
    
"""
This model is designed to predict appliance energy consumption based on various temperature and humidity features from different rooms and outside conditions. The model uses polynomial features to capture non-linear relationships in the data. The predictions are made on a sample of the input data, and the results are printed for review.
Created by: Nova Energy Research & Development
in Oktober, 2025

"""

