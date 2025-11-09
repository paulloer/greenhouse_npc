import requests
import pandas as pd
from constants import TIME_SENSOR

TIME_FORECAST = 3600

def get_weather_forecast(lat=36.8347, lon=-2.4022):
    """
    Fetches weather forecast data from Open-Meteo.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        list: A list of dictionaries containing hourly weather data.
    """
    # Open-Meteo API endpoint
    url = f"https://api.open-meteo.com/v1/forecast"
    
    # Request parameters
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m,shortwave_radiation",
        "timezone": "auto"  # Automatically adjusts for local time zone
    }
    
    try:
        # Make the API request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
    
    # Parse the JSON response
    data = response.json()
    hourly_data = data.get("hourly", {})
    
    # Extract relevant fields
    forecast = []
    for i in range(len(hourly_data["time"])):
        forecast.append({
            "Date": hourly_data["time"][i],  # Time (ISO 8601)
            "Temperature_outside": hourly_data["temperature_2m"][i],  # Temperature in °C
            "Humidity_outside": hourly_data["relative_humidity_2m"][i],  # Humidity in %
            "Radiation_inside": hourly_data.get("shortwave_radiation", [0])[i]/3,  # Solar radiation in W/m²
            "Radiation_outside": hourly_data.get("shortwave_radiation", [0])[i],  # Solar radiation in W/m²
            "Wind_speed_outside": hourly_data["windspeed_10m"][i]  # Wind speed in km/h
        })

    forecast = pd.DataFrame(forecast)
    forecast['Date'] = pd.to_datetime(forecast['Date'], format="%Y-%m-%dT%H:%M")
    forecast['Date'] = forecast['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    df = forecast.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    forecast_resampled = df.resample(f'{TIME_SENSOR}s').interpolate(method='linear')

    # Reset index and convert 'Date' index back to formatted string column
    forecast_resampled = forecast_resampled.reset_index()
    forecast_resampled['Date'] = forecast_resampled['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return forecast_resampled


# Example usage
if __name__ == "__main__":
    # Coordinates for the desired location (e.g., Almeria, Spain)
    latitude = 36.8347
    longitude = -2.4022

    weather_forecast = get_weather_forecast(latitude, longitude)
    print("Weather forecast for the next 24 hours:")
    print(weather_forecast[:24*int(3600/TIME_SENSOR)])

