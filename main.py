import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Constants
API_KEY = 'dc44760c84aa5c702647d61a519432da'
CITY = 'Malvern'
URL = f'http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}'

# Function to convert Kelvin to Celsius
def kelvin_to_celsius(kelvin_temp):
    return kelvin_temp - 273.15

# Function to convert Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius_temp):
    return (celsius_temp * 9/5) + 32

# Data Collection
response = requests.get(URL)
if response.status_code != 200:
    raise Exception(f"Error fetching data from OpenWeatherMap API: {response.status_code}, {response.text}")

data = response.json()

# Check if 'list' key is present in the response
if 'list' not in data:
    raise KeyError("'list' key not found in the API response")

weather_data = []
for entry in data['list']:
    weather = {
        'date': entry['dt_txt'],
        'temperature_celsius': kelvin_to_celsius(entry['main']['temp']),
        'temperature_fahrenheit': celsius_to_fahrenheit(kelvin_to_celsius(entry['main']['temp'])),
        'humidity': entry['main']['humidity'],
        'pressure': entry['main']['pressure'],
        'weather': entry['weather'][0]['description']
    }
    weather_data.append(weather)

df = pd.DataFrame(weather_data)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Data Preprocessing
df['temp_shifted'] = df['temperature_fahrenheit'].shift(1)
df['humidity_shifted'] = df['humidity'].shift(1)
df['pressure_shifted'] = df['pressure'].shift(1)
df.dropna(inplace=True)

# Feature Engineering
df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
df['weekday'] = df.index.weekday

# Train-Test Split
X = df[['temp_shifted', 'humidity_shifted', 'pressure_shifted', 'hour', 'day', 'month', 'weekday']]
y = df['temperature_fahrenheit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Temperature')
plt.plot(y_test.index, y_pred, label='Predicted Temperature', alpha=0.7)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')
plt.title('Actual vs Predicted Temperature')
plt.show()

# Forecasting
future_data = X_test.tail(1).copy()
future_data['temp_shifted'] = y_pred[-1]
future_pred = model.predict(future_data)
future_pred_fahrenheit = future_pred[0]
print(f'Forecasted Temperature: {future_pred_fahrenheit} °F')
