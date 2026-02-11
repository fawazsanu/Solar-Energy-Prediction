import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

#LOAD DATASET
gen_data = pd.read_csv('Plant_1_Generation_Data.csv')
weather_data = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')

#DATES
gen_data['DATE_TIME'] = pd.to_datetime(gen_data['DATE_TIME'])
weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'])

#MERGE DATASETS
df = pd.merge(gen_data.drop(columns=['PLANT_ID']), weather_data.drop(columns=['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')

#FEATURES
df['hour'] = df['DATE_TIME'].dt.hour
df['month'] = df['DATE_TIME'].dt.month

features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'hour', 'month']
target = 'DC_POWER'

X = df[features]
y = df[target]

#SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data successfully processed. Training model...")

#TRAIN XGBOOST MODEL
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05)
model.fit(X_train, y_train)

#PERFORMANCE EVALUATION
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(f"Model Accuracy (R^2 Score): {score * 100:.2f}%")

#SAVE
with open('solar_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to solar_model.pkl")