import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

def cleaning_data(data: pd.DataFrame, column_mapping:dict=None, desired_order:list=None) -> pd.DataFrame:
    if data is None or data.empty:
        raise ValueError("Input data is None or empty.")
    

    if 'Appliances' not in data.columns:
        raise ValueError("Expected 'Appliances' column in the data for target variable.")
    
    if column_mapping is None:
        print(f"column_mapping: {column_mapping}, rn: {rn}")
        raise ValueError("column_mapping should not be None if rn")

    # Renaming the columns based on the provided mapping. It is important for making the column names more descriptive.
    # Filtering the data to include only the desired columns. it helps to focus on relevant features for analysis.
    data = data.rename(columns=column_mapping)
    data = data[[col for col in desired_order if col in data.columns]] if desired_order else data
    return data

def report_data(data: pd.DataFrame, column_mapping: dict) -> tuple[pd.DataFrame, dict]:
    """Schoont de data op en geeft een beknopt rapport terug."""
    data = cleaning_data(data, column_mapping)

    # Eenvoudig rapport
    report = {
        "Aantal rijen": len(data),
        "Aantal kolommen": len(data.columns),
        "Kolommen": list(data.columns),
        "Aantal missende waarden": int(data.isnull().sum().sum()),
        "Gemiddelde energieproductie (kWh)": (
            data["energy"].mean() if "energy" in data.columns else None
        ),
    }

    return data, report


def generate_report(data: pd.DataFrame, save_path: str = "reports/plots", strict: bool = False) -> str:
    """
    Genereert en slaat diverse energievisualisaties op.
    Alle grafieken worden als PNG-bestanden opgeslagen in save_path.

    Args:
        data: pandas DataFrame met de dataset.
        save_path: pad waar de figuren worden opgeslagen.
        strict: als True -> stop bij eerste fout; als False -> sla fouten over.

    Returns:
        str: absoluut pad waar alle figuren zijn opgeslagen.
    """

    if data is None or data.empty:
        raise ValueError("Input data is None of leeg.")

    os.makedirs(save_path, exist_ok=True)

    # 1️⃣ Daily Energy Heatmap
    try:
        if {"Appliances", "day", "month"}.issubset(data.columns):
            daily_energy = data.pivot_table(
                values="Appliances", index="day", columns="month", aggfunc="mean"
            )
            plt.figure(figsize=(10, 5))
            plt.title("Daily Energy Consumption")
            plt.xlabel("Month")
            plt.ylabel("Day")
            plt.imshow(daily_energy, cmap="YlGnBu", aspect="auto")
            plt.colorbar(label="Energy Consumption")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "daily_energy_heatmap.png"))
            plt.close()
    except Exception as e:
        msg = f"[Warning] Daily heatmap skipped: {e}"
        if strict:
            raise ValueError(msg)
        print(msg)

    # 2️⃣ Boxplot by Day of Week
    try:
        if {"day_of_week", "Appliances"}.issubset(data.columns):
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            data["day_of_week"] = data["day_of_week"].map(
                lambda x: day_names[int(x)] if pd.notnull(x) and int(x) < len(day_names) else np.nan
            )
            plt.figure(figsize=(10, 6))
            sns.boxplot(x="day_of_week", y="Appliances", data=data, order=day_names)
            plt.title("Appliance Energy Consumption by Day of the Week")
            plt.xlabel("Day of the Week")
            plt.ylabel("Energy Consumption")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "boxplot_dayofweek.png"))
            plt.close()
    except Exception as e:
        msg = f"[Warning] Boxplot skipped: {e}"
        if strict:
            raise ValueError(msg)
        print(msg)

    # 3️⃣ Boxplots per column (Outliers)
    try:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        num_columns = len(numeric_cols)
        fig, axes = plt.subplots(nrows=num_columns, figsize=(8, num_columns * 3))
        if num_columns == 1:
            axes = [axes]
        for i, column in enumerate(numeric_cols):
            data.boxplot(column=column, ax=axes[i])
            axes[i].set_title(f"Box Plot for {column}")
            axes[i].set_xlabel("")
            axes[i].set_ylabel("Values")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "all_boxplots.png"))
        plt.close()
    except Exception as e:
        msg = f"[Warning] Column boxplots skipped: {e}"
        if strict:
            raise ValueError(msg)
        print(msg)

    # 4️⃣ Hourly energy pattern
    try:
        if {"hour", "Appliances"}.issubset(data.columns):
            hourly_energy = data.groupby("hour")["Appliances"].mean()
            plt.figure(figsize=(12, 6))
            plt.plot(hourly_energy.index, hourly_energy.values, marker="o")
            plt.title("Hourly Energy Consumption Patterns")
            plt.xlabel("Hour of the Day")
            plt.ylabel("Energy Consumption (mean)")
            plt.xticks(range(24))
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "hourly_energy_pattern.png"))
            plt.close()
    except Exception as e:
        msg = f"[Warning] Hourly pattern skipped: {e}"
        if strict:
            raise ValueError(msg)
        print(msg)

    # 5️⃣ Regression plots (temperatures vs Appliances)
    temp_columns = [
        "KITCHEN_TEMP", "LIVING_TEMP", "BEDROOM_TEMP", "OFFICE_TEMP",
        "BATHROOM_TEMP", "OUTSIDE_TEMP_build", "IRONING_ROOM_TEMP",
        "TEEN_ROOM_2_TEMP", "PARENTS_ROOM_TEMP", "OUTSIDE_TEMP_wstn",
    ]
    for col in temp_columns:
        if {"Appliances", col}.issubset(data.columns):
            try:
                plt.figure(figsize=(10, 6))
                sns.regplot(
                    x=col, y="Appliances", data=data,
                    scatter_kws={"alpha": 0.5}, line_kws={"color": "red"}
                )
                plt.title(f"{col} vs. Energy Consumption")
                plt.xlabel(col)
                plt.ylabel("Energy Consumption (Appliances)")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f"regplot_{col}.png"))
                plt.close()
            except Exception as e:
                msg = f"[Warning] Regression for {col} skipped: {e}"
                if strict:
                    raise ValueError(msg)
                print(msg)

    # 6️⃣ Weekday vs Weekend Comparison
    try:
        if {"weekday", "hour", "Appliances"}.issubset(data.columns):
            weekday_energy = data[data["weekday"] < 5].groupby("hour")["Appliances"].mean()
            weekend_energy = data[data["weekday"] >= 5].groupby("hour")["Appliances"].mean()
            plt.figure(figsize=(10, 6))
            plt.plot(weekday_energy.index, weekday_energy.values, label="Weekdays", marker="o")
            plt.plot(weekend_energy.index, weekend_energy.values, label="Weekends", marker="o")
            plt.title("Energy Consumption on Weekdays vs. Weekends")
            plt.xlabel("Hour of the Day")
            plt.ylabel("Mean Energy Consumption")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "weekday_vs_weekend.png"))
            plt.close()
    except Exception as e:
        msg = f"[Warning] Weekday/weekend chart skipped: {e}"
        if strict:
            raise ValueError(msg)
        print(msg)

    # 7️⃣ Correlation Heatmap
    try:
        corr = data.corr(numeric_only=True)
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr, annot=False, cmap="RdYlGn")
        plt.title("Correlation Matrix Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "correlation_heatmap.png"))
        plt.close()
    except Exception as e:
        msg = f"[Warning] Correlation heatmap skipped: {e}"
        if strict:
            raise ValueError(msg)
        print(msg)

    # 8️⃣ Missing Values Matrix
    try:
        plt.figure(figsize=(12, 6))
        msno.matrix(data)
        plt.title("Missing Values Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "missing_values.png"))
        plt.close()
    except Exception as e:
        msg = f"[Warning] Missing values plot skipped: {e}"
        if strict:
            raise ValueError(msg)
        print(msg)

    # 9️⃣ Histogram Distributions
    try:
        data.hist(figsize=(20, 20), grid=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "histograms.png"))
        plt.close()
    except Exception as e:
        msg = f"[Warning] Histograms skipped: {e}"
        if strict:
            raise ValueError(msg)
        print(msg)

    return str(os.path.abspath(save_path))


"""
This model is designed to predict appliance energy consumption based on various temperature and humidity features from different rooms and outside conditions. The model uses polynomial features to capture non-linear relationships in the data. The predictions are made on a sample of the input data, and the results are printed for review.
Created by: Nova Energy Research & Development
in Oktober, 2025

"""