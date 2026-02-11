import pickle
from datetime import datetime
import requests
import pandas as pd
import numpy as np

#LOAD
with open('solar_model.pkl', 'rb') as f:
    model = pickle.load(f)

#CONFIG
API_KEY = 'YOUR_API_KEY_HERE'
CITY = 'Lagos'
URL = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

def get_live_prediction():

    #GET LIVE WEATHER DATA
    response = requests.get(URL, timeout=5)
    data = response.json()

    if response.status_code != 200:
        print("Error fetching weather data")
        return
    
    temp = data['main']['temp']

    module_temp = temp * 1.2

    cloud_cover = data['clouds']['all']
    irradiation = (100 - cloud_cover) / 100 * 1000

    current_hour = datetime.now().hour
    current_month = datetime.now().month

    #PREPARE DATA FOR MODEL
    input_data = pd.DataFrame([[temp, module_temp, irradiation, current_hour, current_month]], columns=['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'hour', 'month'])

    #PREDICT
    prediction = model.predict(input_data)

    print(f"Live Solar Forecast for {CITY}")
    print(f"Current Temperature: {temp}°C")
    print(f"Cloud Cover: {cloud_cover}%")
    print(f"Estimated Generation: {prediction[0]:.2f} kW")

if __name__ == "__main__":
    get_live_prediction()