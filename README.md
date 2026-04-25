# Solar Energy Output Prediction System

## Project Overview
This project addresses the challenge of grid instability in renewable energy systems. By analyzing historical solar generation and weather sensor data from a real PV plant, I developed a machine learning model to forecast DC power output. The system includes a live Python application that integrates with the OpenWeatherMap API to provide real-time generation predictions.

**Key Results:**
| Metric | Value |
|---|---|
| R² Score | 0.9797 (97.97%) |
| RMSE | 532.96 kW |
| MAE | 181.15 kW |
| MAE as % of mean output | 6.3% |
| Cross-validation R² (5-fold) | 0.9797 ± 0.0103 |

**Tech Stack:** Python · XGBoost · Scikit-Learn · Pandas · NumPy · OpenWeatherMap API

---

## Methodology

### Data
- **Dataset:** [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) — Kaggle (Plant 1)
- 68,774 samples spanning 2020-05-15 to 2020-06-17
- Generation data merged with weather sensor readings on timestamp

### Features Used
| Feature | Description |
|---|---|
| `IRRADIATION` | Solar irradiance (most important feature) |
| `MODULE_TEMPERATURE` | PV panel surface temperature |
| `AMBIENT_TEMPERATURE` | Surrounding air temperature |
| `irrad_x_temp` | Interaction term: irradiation × module temperature |
| `hour` | Hour of day |
| `month` | Month |
| `day_of_week` | Day of week |
| `is_daytime` | Binary flag (06:00–18:00) |

### Evaluation Approach
Solar generation data is a time series. Using a random train/test split on time-series data causes **temporal leakage**, training samples from the same day appear in both sets, allowing the model to interpolate rather than forecast. This inflates R² scores significantly.

This project uses a **strict chronological split**:
- **Training set:** first 80% of timesteps (55,019 samples, up to 2020-06-11)
- **Test set:** last 20% of timesteps (13,755 samples, after 2020-06-11)

Additionally, **TimeSeriesSplit cross-validation** (5 folds) is applied to the training set to verify stability across time periods.

> **Note on earlier version:** A previous iteration of this project used `sklearn.train_test_split` with `random_state=42`, which caused temporal leakage and produced an inflated R² of 98.64%. The corrected chronological methodology produces a leakage-free R² of 97.97% — a marginal difference that confirms the original model was not fundamentally wrong, but the evaluation was methodologically unsound.

---

## How It Works

1. **Data Processing:** Generation data is merged with sensor readings and sorted chronologically. Features are engineered to capture temporal patterns and non-linear irradiation effects.
2. **Training:** XGBoost Regressor trained on the first 80% of timesteps with early stopping. Evaluated on the held-out final 20%.
3. **Live Application:** `predict_app.py` fetches live weather data for a given location via the OpenWeatherMap API and estimates current solar power generation in kW.

---

## Usage

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) and place these files in the project root:
- `Plant_1_Generation_Data.csv`
- `Plant_1_Weather_Sensor_Data.csv`

**3. Train the model:**
```bash
python train_model.py
```

**4. Run the live predictor:**
```bash
python predict_app.py
```
> You will need a free API key from [OpenWeatherMap](https://openweathermap.org/api).

---

## Feature Importance
IRRADIATION is the dominant predictor by a wide margin, followed by the irradiation × module temperature interaction term. Time-based features (hour, is_daytime) contribute meaningfully, confirming that the model captures diurnal solar patterns correctly.

---

## Limitations
- The dataset covers a single plant over ~33 days. Generalization to other plants or seasons may require retraining.
- RMSE (532.96 kW) is higher than MAE (181.15 kW), indicating occasional large errors, likely at dawn/dusk transitions where irradiation changes rapidly.
- The live predictor uses weather forecast data as a proxy for irradiation, which introduces additional uncertainty not present in the training data.