# Solar Energy Output Prediction System

## Project Overview
This project addresses the challenge of grid instability in renewable energy systems. By analyzing historical solar generation data and weather patterns, I developed a Machine Learning model to forecast energy output. The system includes a live Python application that integrates with the OpenWeatherMap API to provide real-time generation predictions.

**Key Results:**
- **Model Accuracy:** 98.64% (R² Score)
- **Tech Stack:** Python, XGBoost, Scikit-Learn, Pandas, REST API

## How It Works
1.  **Data Processing:** Merged generation data with sensor readings to correlate ambient temperature, module temperature, and irradiance with DC Power output.
2.  **Training:** Utilized the XGBoost Regressor algorithm to train a predictive model on the processed dataset.
3.  **Live Application:** Built a script (`predict_app.py`) that fetches live weather data for a specific location and estimates current solar power generation in kilowatts (kW).

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Train the model:
   python train_model.py

3. Run the live predictor:
   python predict_app.py

(NOTE: You will need your own API key from OpenWeatherMap)
