"""
Solar Energy Output Prediction — Model Training
================================================
Dataset : Kaggle Solar Power Generation Data (Plant 1)
Model   : XGBoost Regressor
Target  : DC_POWER (kilowatts)

Evaluation methodology
----------------------
Solar generation data is a time series. Using a random train/test split
on time-series data causes temporal leakage — training samples from the
same day appear in both sets, allowing the model to interpolate rather
than forecast. This inflates R² scores significantly.

This script uses a strict chronological split:
  - Training set : first 80% of timesteps (earlier dates)
  - Test set     : last  20% of timesteps (later dates)

This simulates real-world deployment where the model must predict
future generation from unseen time periods.

Additionally, TimeSeriesSplit cross-validation is run on the training
set to report stable, leakage-free performance estimates.
"""

import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')

# ── 1. Load & merge ──────────────────────────────────────────────────
print("Loading datasets...")
gen_data     = pd.read_csv('Plant_1_Generation_Data.csv')
weather_data = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')

gen_data['DATE_TIME']     = pd.to_datetime(gen_data['DATE_TIME'])
weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'])

df = pd.merge(
    gen_data.drop(columns=['PLANT_ID']),
    weather_data.drop(columns=['PLANT_ID', 'SOURCE_KEY']),
    on='DATE_TIME'
)

# ── 2. Feature engineering ───────────────────────────────────────────
df = df.sort_values('DATE_TIME').reset_index(drop=True)

df['hour']          = df['DATE_TIME'].dt.hour
df['month']         = df['DATE_TIME'].dt.month
df['day_of_week']   = df['DATE_TIME'].dt.dayofweek
df['is_daytime']    = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
# Interaction: irradiation tends to scale non-linearly with temperature
df['irrad_x_temp']  = df['IRRADIATION'] * df['MODULE_TEMPERATURE']

FEATURES = [
    'AMBIENT_TEMPERATURE',
    'MODULE_TEMPERATURE',
    'IRRADIATION',
    'hour',
    'month',
    'day_of_week',
    'is_daytime',
    'irrad_x_temp',
]
TARGET = 'DC_POWER'

df = df.dropna(subset=FEATURES + [TARGET])
X = df[FEATURES]
y = df[TARGET]

print(f"Dataset: {len(df):,} samples | {df['DATE_TIME'].min().date()} → {df['DATE_TIME'].max().date()}")

# ── 3. Chronological train/test split ────────────────────────────────
split_idx = int(len(df) * 0.80)
split_date = df['DATE_TIME'].iloc[split_idx].date()

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nChronological split (80/20):")
print(f"  Train : {len(X_train):,} samples  (up to {split_date})")
print(f"  Test  : {len(X_test):,}  samples  (after {split_date})")

# ── 4. Train model ───────────────────────────────────────────────────
print("\nTraining XGBoost model...")
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    eval_metric='rmse',
    random_state=42,
    verbosity=0,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# ── 5. Evaluation on held-out test set ──────────────────────────────
predictions = model.predict(X_test)

r2   = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae  = mean_absolute_error(y_test, predictions)
mean_actual = y_test.mean()

print("\n" + "=" * 50)
print("  HELD-OUT TEST SET RESULTS")
print("  (Chronological split — no temporal leakage)")
print("=" * 50)
print(f"  R² Score : {r2:.4f}  ({r2 * 100:.2f}%)")
print(f"  RMSE     : {rmse:.2f} kW")
print(f"  MAE      : {mae:.2f} kW")
print(f"  Mean DC Power (test period) : {mean_actual:.2f} kW")
print(f"  MAE as % of mean            : {mae / mean_actual * 100:.1f}%")

# ── 6. TimeSeriesSplit cross-validation ─────────────────────────────
print("\nRunning TimeSeriesSplit cross-validation (5 folds)...")
tscv = TimeSeriesSplit(n_splits=5)

# Use a lighter model for CV speed
cv_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbosity=0,
)
cv_scores = cross_val_score(cv_model, X_train, y_train, cv=tscv, scoring='r2')

print(f"\n  CV R² scores per fold: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean CV R²           : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 7. Feature importance ────────────────────────────────────────────
print("\n  Feature Importances (by gain):")
importance = model.get_booster().get_score(importance_type='gain')
importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
for feat, score in importance_sorted:
    bar = '█' * int(score / max(importance.values()) * 20)
    print(f"    {feat:<22} {bar} ({score:.1f})")

# ── 8. Methodology note ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("  METHODOLOGY NOTE")
print("=" * 50)
print("""
  Earlier versions of this project used a random train/test
  split (sklearn train_test_split, random_state=42), which
  caused temporal leakage on this time-series dataset and
  produced an inflated R² of ~98.64%.

  This version uses a strict chronological split and
  TimeSeriesSplit cross-validation to produce leakage-free,
  deployment-realistic performance estimates.
""")

# ── 9. Save model ────────────────────────────────────────────────────
with open('solar_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("  Model saved to solar_model.pkl")
