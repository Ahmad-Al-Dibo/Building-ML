# ⚡ Energy Consumption Prediction Pipeline  
### Freelance Project — Ahmad Al Dibo student
---

## 🌍 Over het project

Dit project is ontwikkeld door **Ahmad Al Dibo student** als onderdeel van een onderzoeksinitiatief naar **energie-efficiëntie en slimme gebouwanalyse**.  
Het doel is om **energieverbruik van huishoudelijke apparaten** te voorspellen aan de hand van **sensor- en omgevingsdata** (temperatuur, luchtvochtigheid, luchtdruk, etc.).

De pipeline bevat een volledige **machine-learning workflow**:
- **Data preprocessing**  
- **Modeltraining** (meerdere regressiemodellen)  
- **Model evaluatie en selectie**  
- **Voorspellingen en rapportage**  
- **Visualisaties van prestaties**

---

## 🧠 Kerncomponenten

### 1️⃣ Data Preprocessing
Het bestand [`process_data.py`](src/data/process_data.py) verzorgt het volledige schoonmaakproces:

- Detecteert **uitbijters** met de *IQR-methode*  
- Analyseert **scheefheid (skewness)** en past *PowerTransformer* toe  
- Schaal features met *StandardScaler*  
- Splitst automatisch in **train/test**  
- Genereert **PolynomialFeatures** (graad 2) om niet-lineaire verbanden te modelleren  

**Belangrijkste functie:**

```python
def preprocess(dataset_path: str, column_mapping: dict, desired_order: list) -> dict:
    """Schoont en transformeert de dataset; retourneert train/test-sets."""
````

Retourneert een dictionary:

```python
{
  "Training": (X_train_poly, X_test_poly, y_train, y_test),
  "Full": (X, y)
}
```

---

### 2️⃣ Modeltraining en evaluatie

Het bestand [`train_models.py`](src/models/train_models.py) bevat:

* Training van meerdere regressiemodellen:

  * LinearRegression, Ridge, Lasso, ElasticNet
  * RandomForest, GradientBoosting, SVR, XGBoost
* Automatische berekening van:

  * R²-score
  * RMSE
  * MAE
* Automatisch opslaan van modellen als `.pkl`
* Evaluatie en visualisatie van modelprestaties

Voorbeeld:

```python
from src import train_all_models

results_df, best_model_path = train_all_models(models, (X_train, X_test, y_train, y_test))
```

---

### 3️⃣ Centraal orchestratiebestand: `main.py`

Het bestand [`main.py`](main.py) coördineert de hele pipeline:

1. **Data inladen en preprocessen**
2. **Controleren op bestaande modellen**
3. **Trainen van nieuwe modellen indien nodig**
4. **Evaluatie en selectie van beste model**
5. **Voorspellingen opslaan en visualisaties genereren**

---

## 🧩 Structuur van het project

```
nova_energy_appliance_prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── output/
│       ├── models/
│       └── train_test/
│
├── notebooks/
│   ├── LinearRegressionOnderzoek.ipynb
│
├── reports/
│   ├── plots/
│   └── summary/
│
├── src/
│   ├── data/
│   │   ├── process_data.py
│   ├── models/
│   │   ├── train_models.py
│   ├── reports/
│   │   └── make_report.py
│   └── utils/
│       ├── core.py
│       └── overview.py
│
├── requirements.txt
├── config.yaml
├── README.md
└── main.py
```

---

## 🎨 Consolekleurondersteuning

De verbeterde versie van `main.py` bevat een **ConsoleColors**-klasse
die zorgt voor duidelijke kleuruitvoer in de terminal — zonder emoji’s of symbolen.

```python
class ConsoleColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
```

Voorbeeld van de gekleurde uitvoer:

```text
=== Start Energy Prediction Pipeline ===
Stap 1: Data schoonmaken en verdelen
Preprocessing voltooid
Stap 2: Controleren op bestaande modellen
Training voltooid. Modellen opgeslagen in Data/output/models
Beste model: RandomForest (R² = 0.9781)
Pipeline voltooid
```

Kleurcodes helpen om snel te onderscheiden:

* Informatie (blauw)
* Succes (groen)
* Waarschuwingen (geel)
* Fouten (rood)
* Pipeline-begins/ends (paars)

---

## ⚙️ Installatie

### 1️⃣ Clone de repository

```bash
git clone https://github.com/Ahmad-Al-Dibo/energy-preprocessing.git
cd energy-preprocessing
```

### 2️⃣ Virtuele omgeving aanmaken

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Dependencies installeren

```bash
pip install -r requirements.txt
```

---

## 🚀 Pipeline uitvoeren

Om de volledige pipeline te draaien:

```bash
python main.py
```

Of om enkel preprocessing te testen:

```python
from src.process_data import preprocess
data = preprocess("data/raw/KAG_energydata_complete.csv", column_mapping, desired_order)
```

---

## 📊 Voorbeeldoutput

Tijdens uitvoering zie je o.a.:

```text
FEATURES FOLLOWED SYMMETRICAL DISTRIBUTION :
Index(['KITCHEN_TEMP', 'BEDROOM_TEMP', 'OFFICE_TEMP'], dtype='object')

FEATURES FOLLOWED SKEWED DISTRIBUTION :
Index(['LIVING_TEMP', 'BATHROOM_TEMP', 'OUTSIDE_TEMP_wstn'], dtype='object')

Beste model: RandomForest (R² = 0.9812)
Voorspellingen opgeslagen in Data/output/Xpredictions.csv
```

De resultaten en plots worden automatisch opgeslagen in:

```
Data/output/reports/plots/
```

---

## 🧪 Gebruikte Technologieën

| Categorie        | Technologie           |
| ---------------- | --------------------- |
| Programmeertaal  | Python 3.10+          |
| Data Processing  | pandas, numpy         |
| Machine Learning | scikit-learn, XGBoost |
| Visualisatie     | matplotlib            |
| Logging & Config | logging, YAML         |
| Structuurbeheer  | pathlib, os           |

---

## 📈 Rapporten & Visualisaties

De pipeline genereert automatisch:

* Barplot met R²-scores van alle modellen
* CSV-bestanden met voorspellingen en labels
* Optioneel: tekstueel rapport via `make_report.py`

```bash
python -m src.reports.make_report
```

---

## 👥 Auteur & Contact

**Auteur:**

> Ahmad Al Dibo — Data Engineer & Researcher

📧 Contact: [ahmadaldibo212009@gmail.com](mailto:ahmadaldibo212009@gmail.com)

---

## 🔮 Toekomstige uitbreidingen

* Integratie van **PCA (dimensionality reduction)**
* Geavanceerde **feature selection**
* Geautomatiseerde **hyperparameter tuning**
* Live-dashboard voor modelmonitoring
* CI/CD integratie voor modeldeployment

---

## 💡 Inspiratie

> “Data is the new fuel — preprocessing is the refinery.”
> — Ahmad Al Dibo
