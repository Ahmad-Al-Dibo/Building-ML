
"""
main.py
========
Centrale orchestrator voor de Energy Prediction pipeline.

Stappen:
1. Data inladen & preprocessen
2. Modellen trainen (indien nog niet aanwezig)
3. Modellen evalueren en beste model selecteren
4. Beste model gebruiken voor voorspellingen en resultaten opslaan
5. Rapporten / visualisaties genereren

Auteur: Ahmad Al Dibo
Datum: Oktober 2025
"""

import os
import yaml
import logging
import sys
from pathlib import Path
import pandas as pd

try:
    from src import preprocess
except Exception as e:
    raise ImportError("Kon 'preprocess' niet importeren uit src.process_data of process_data.py") from e

try:
    from src import (
        train_all_models,
        get_models_accuracy,
        load_model,
        load_and_predict,
        visual_outputs,
    )
except Exception as e:
    raise ImportError("Kon functies uit train_models niet importeren.") from e

try:
    from src import main_make_report as generate_report_from_script
    HAVE_GENERATE_REPORT = True
except Exception:
    HAVE_GENERATE_REPORT = False


class ConsoleColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"



with open("config.yaml") as f:
    config = yaml.safe_load(f)
DATASET_PATH = config["data"]["raw_path"]

MODEL_OUTPUT_DIR = config["data"]["model_dir"]
PLOTS_OUTPUT_DIR = config["data"]["plots_dir"]
PREDICTIONS_FILE = config["data"]["productions_f"]
LABELS_FILE = config["data"]["productions_f"]
column_mapping = {
    'T1': 'KITCHEN_TEMP','RH_1': 'KITCHEN_HUM',
    'T2': 'LIVING_TEMP','RH_2' :'LIVING_HUM',
    'T3': 'BEDROOM_TEMP','RH_3':'BEDROOM_HUM',
    'T4' : 'OFFICE_TEMP','RH_4' : 'OFFICE_HUM',
    'T5' : 'BATHROOM_TEMP','RH_5': 'BATHROOM_HUM',
    'T6':'OUTSIDE_TEMP_build','RH_6': 'OUTSIDE_HUM_build',
    'T7': 'IRONING_ROOM_TEMP','RH_7' : 'IRONING_ROOM_HUM',
    'T8' :'TEEN_ROOM_2_TEMP','RH_8' : 'TEEN_ROOM_HUM',
    'T9': 'PARENTS_ROOM_TEMP','RH_9': 'PARENTS_ROOM_HUM',
    'T_out' :'OUTSIDE_TEMP_wstn','RH_out' :'OUTSIDE_HUM_wstn'
}

desired_order = [
    'KITCHEN_TEMP','LIVING_TEMP','BEDROOM_TEMP','OFFICE_TEMP','BATHROOM_TEMP',
    'OUTSIDE_TEMP_build','IRONING_ROOM_TEMP','TEEN_ROOM_2_TEMP','PARENTS_ROOM_TEMP','OUTSIDE_TEMP_wstn',
    'KITCHEN_HUM','LIVING_HUM','BEDROOM_HUM','OFFICE_HUM','BATHROOM_HUM','OUTSIDE_HUM_build',
    'IRONING_ROOM_HUM','TEEN_ROOM_HUM','PARENTS_ROOM_HUM','OUTSIDE_HUM_wstn',
    "Tdewpoint","Press_mm_hg","Windspeed","Visibility","rv1","rv2",
    'month','weekday','hour','week','day','day_of_week',"Appliances"
]

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb

MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, max_iter=2000),
    "Lasso": Lasso(alpha=0.001, max_iter=5000),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=3, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=3),
    "SVR": SVR(kernel='rbf', C=100, gamma='scale'),
    "XGBoost": xgb.XGBRegressor(n_estimators=250, learning_rate=0.1, max_depth=4, random_state=3, n_jobs=-1),
}


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nova_energy")


def ensure_dirs():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

def find_existing_models(models_dict):
    """
    Controleer welke modellen reeds als .pkl aanwezig zijn in MODEL_OUTPUT_DIR.
    Retourneer een dict met geladen modellen.
    """
    loaded = {}
    for name in models_dict.keys():
        path = os.path.join(MODEL_OUTPUT_DIR, f"{name}_model.pkl")
        if os.path.exists(path):
            try:
                loaded[name] = load_model(path)
                logger.info(f"✔ Gevonden en geladen: {path}")
            except Exception as e:
                logger.warning(f"Kon model {path} niet laden: {e}")
    return loaded

def define_beste_model(results_df):
    """
    Verwacht een pandas DataFrame gesorteerd op 'R² Score' (desc).
    Returned (model_name, score)
    """
    if results_df is None or results_df.empty:
        raise ValueError("Empty results_df in define_beste_model")
    top = results_df.iloc[0]
    return top["Model"], float(top["R² Score"])


def main():
    print(f"{ConsoleColors.HEADER}=== Start Energy Prediction Pipeline ==={ConsoleColors.ENDC}")
    ensure_dirs()

    # --- Stap 1 ---
    print(f"{ConsoleColors.OKBLUE}Stap 1: Data schoonmaken en verdelen{ConsoleColors.ENDC}")
    try:
        preprocessed = preprocess(DATASET_PATH, column_mapping, desired_order)
        X_train, X_test, y_train, y_test = preprocessed["Training"]
        print(f"{ConsoleColors.OKGREEN}Preprocessing voltooid{ConsoleColors.ENDC}")
    except Exception as e:
        print(f"{ConsoleColors.FAIL}Fout bij preprocessing: {e}{ConsoleColors.ENDC}")
        sys.exit(1)

    # --- Stap 2 ---
    print(f"{ConsoleColors.OKBLUE}Stap 2: Controleren op bestaande modellen{ConsoleColors.ENDC}")
    existing_models = find_existing_models(MODELS)

    # --- Stap 3 ---
    if not existing_models:
        print(f"{ConsoleColors.WARNING}Geen bestaande modellen gevonden. Start training van alle modellen.{ConsoleColors.ENDC}")
        try:
            results_df, last_model_path = train_all_models(
                MODELS,
                (X_train, X_test, y_train, y_test),
                folder_output=MODEL_OUTPUT_DIR,
            )
            print(f"{ConsoleColors.OKGREEN}Training voltooid. Modellen opgeslagen in: {MODEL_OUTPUT_DIR}{ConsoleColors.ENDC}")
            existing_models = find_existing_models(MODELS)
        except Exception as e:
            print(f"{ConsoleColors.FAIL}Fout bij trainen van modellen: {e}{ConsoleColors.ENDC}")
            sys.exit(1)
    else:
        print(f"{ConsoleColors.OKBLUE}Er zijn bestaande modellen geladen; accuratesse wordt berekend.{ConsoleColors.ENDC}")
        try:
            _, results_df = get_models_accuracy(existing_models, (X_train, X_test, y_train, y_test))
        except Exception as e:
            print(f"{ConsoleColors.FAIL}Fout bij berekenen van accuratesse: {e}{ConsoleColors.ENDC}")
            sys.exit(1)

    print(f"{ConsoleColors.OKCYAN}Modellen ranking:\n{results_df.to_string(index=False)}{ConsoleColors.ENDC}")

    # --- Stap 4 ---
    try:
        best_model_name, best_score = define_beste_model(results_df)
        print(f"{ConsoleColors.OKGREEN}Beste model: {best_model_name} (R² = {best_score:.4f}){ConsoleColors.ENDC}")

        best_model_file = os.path.join(MODEL_OUTPUT_DIR, f"{best_model_name}_model.pkl")
        if not os.path.exists(best_model_file):
            print(f"{ConsoleColors.FAIL}Verwacht modelbestand niet gevonden: {best_model_file}{ConsoleColors.ENDC}")
            if best_model_name in existing_models:
                print(f"{ConsoleColors.WARNING}Gebruik geladen model uit geheugen.{ConsoleColors.ENDC}")
                best_model = existing_models[best_model_name]
            else:
                print(f"{ConsoleColors.FAIL}Kan het beste model niet vinden of laden.{ConsoleColors.ENDC}")
                sys.exit(1)
        else:
            best_model = load_model(best_model_file)

        print(f"{ConsoleColors.OKBLUE}Voorspellingen maken met {best_model_name}{ConsoleColors.ENDC}")
        predictions = best_model.predict(X_test)

        pred_df = pd.DataFrame({"Prediction": predictions})
        os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
        pred_df.to_csv(PREDICTIONS_FILE, index=False)

        labels_df = pd.DataFrame({"Label": y_test.reset_index(drop=True)})
        labels_df.to_csv(LABELS_FILE, index=False)

        print(f"{ConsoleColors.OKGREEN}Voorspellingen opgeslagen in {PREDICTIONS_FILE} en labels in {LABELS_FILE}{ConsoleColors.ENDC}")
    except Exception as e:
        print(f"{ConsoleColors.FAIL}Fout bij gebruik van het beste model: {e}{ConsoleColors.ENDC}")
        sys.exit(1)

    # --- Stap 5 ---
    print(f"{ConsoleColors.OKBLUE}Stap 5: Visualisaties & rapporten genereren{ConsoleColors.ENDC}")
    try:
        try:
            visual_outputs(results_df, PLOTS_OUTPUT_DIR, filename="models_r2_comparison.png")
            print(f"{ConsoleColors.OKGREEN}Visualisaties opgeslagen in {PLOTS_OUTPUT_DIR}{ConsoleColors.ENDC}")
        except Exception as e:
            print(f"{ConsoleColors.WARNING}Visualisatie fout: {e}{ConsoleColors.ENDC}")

        if HAVE_GENERATE_REPORT:
            try:
                generate_report_from_script()
                print(f"{ConsoleColors.OKGREEN}Rapport succesvol gegenereerd{ConsoleColors.ENDC}")
            except Exception as e:
                print(f"{ConsoleColors.WARNING}Rapportgeneratie fout: {e}{ConsoleColors.ENDC}")
        else:
            print(f"{ConsoleColors.WARNING}Geen rapportfunctie beschikbaar; stap overgeslagen.{ConsoleColors.ENDC}")
    except Exception as e:
        print(f"{ConsoleColors.FAIL}Fout bij rapportage of visualisatie: {e}{ConsoleColors.ENDC}")

    print(f"{ConsoleColors.HEADER}=== Pipeline voltooid ==={ConsoleColors.ENDC}")

if __name__ == "__main__":
    main()
