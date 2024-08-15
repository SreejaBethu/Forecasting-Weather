# Weather-Forecasting-Project
This project leverages the OpenWeatherMap API and machine learning techniques to forecast weather temperatures. The project uses historical weather data to train a Random Forest Regressor model, which predicts future temperatures. The forecasted temperatures are converted to Fahrenheit for ease of understanding. The project also provides visualizations to compare actual vs predicted temperatures, allowing for model evaluation and improvement.

****Table of Contents**
Features
Installation
Usage
Project Structure
Contributing
License

1) **Features**
    **Weather Data Retrieval:** Retrieves historical weather data from the OpenWeatherMap API
    **Temperature Conversion:** Converts temperature data from Celsius to Fahrenheit for ease of understanding
    **Random Forest Regressor:** Trains a Random Forest Regressor model on historical weather data to predict future temperatures
    **Visualization:** Provides visualizations to compare actual vs predicted temperatures, allowing for model evaluation and improvement


2) **Installation**
Prerequisites
Python 3.x
Pip (Python package installer)
Libraries

3) **You will need the following Python libraries:**
requests
pandas
numpy
scikit-learn
matplotlib

**Project Structure**
    **data**: Contains the historical weather data retrieved from the OpenWeatherMap API
    **models**: Holds the trained Random Forest Regressor model
    **src**: Contains the source code for the weather forecasting system
    **utils**: Holds utility functions for data preprocessing, feature engineering, and visualization
    **visualizations**: Contains the visualizations comparing actual vs predicted temperatures


5) You can install these libraries using pip: **pip install requests pandas numpy scikit-learn matplotlib**

6) **Run the script:**
You can run the script from the command line or terminal: python weather_forecast.py


7) **Set up your environment:**
Make sure you have Python installed on your machine. You can download it from **python.org**.

**Get your OpenWeatherMap API key:**
**Sign up at OpenWeatherMap to get a free API key.**
**Update the script:**
**Open the weather_forecast.py file and replace 'your_api_key' and 'your_city' with your actual OpenWeatherMap API key and city name.**
