import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the cleaned dataset
df = pd.read_csv("India_Weather_2000_2024_Cleaned.csv")

# CONVERT DATE
df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# BASIC DATASET OVERVIEW:

print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

# YEARLY AVG MAX TEMP TREND

yearly_temp = df.groupby('year')['temperature_2m_max'].mean()

plt.figure(figsize=(10,5))
plt.plot(yearly_temp, label="Average Temp")

# Add trend line
z = np.polyfit(yearly_temp.index, yearly_temp.values, 1)
p = np.poly1d(z)
plt.plot(yearly_temp.index, p(yearly_temp.index))

plt.title("Average Yearly Maximum Temperature (2000–2024)")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()

# Group by year and sum rainfall
yearly_rain = df.groupby('year')['precipitation_sum'].sum()

plt.figure(figsize=(10,5))
plt.plot(yearly_rain, label="Total Rainfall")

# Add trend line
z = np.polyfit(yearly_rain.index, yearly_rain.values, 1)
p = np.poly1d(z)
plt.plot(yearly_rain.index, p(yearly_rain.index))

plt.title("Total Yearly Rainfall (2000–2024)")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.legend()
plt.show()

# AVG MAX TEMP PER CITY
city_temp = df.groupby('city')['temperature_2m_max'].mean()

plt.figure(figsize=(10,5))
city_temp.plot(kind='bar')
plt.title("Average Maximum Temperature by City (2000–2024)")
plt.xlabel("City")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.show()

print(city_temp.sort_values(ascending=False))

# TOTAL RAINFALL PER CITY
city_rain = df.groupby('city')['precipitation_sum'].sum()

plt.figure(figsize=(10,5))
city_rain.plot(kind='bar')
plt.title("Total Rainfall by City (2000–2024)")
plt.xlabel("City")
plt.ylabel("Rainfall (mm)")
plt.xticks(rotation=45)
plt.show()

print(city_rain.sort_values(ascending=False))

# ML

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Features (X) and Target (y)
X = df[['temperature_2m_min', 'precipitation_sum',
        'wind_speed_10m_max', 'wind_gusts_10m_max',
        'year', 'month']]

y = df['temperature_2m_max']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

from sklearn.ensemble import RandomForestRegressor

# Create model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)

# Evaluate
print("\nRandom Forest Results:")
print("R2 Score:", r2_score(y_test, rf_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, rf_pred))

# 📈 Predict Temperature for 2025

# Take average values of other features
avg_temp_min = df['temperature_2m_min'].mean()
avg_precip = df['precipitation_sum'].mean()
avg_wind_speed = df['wind_speed_10m_max'].mean()
avg_wind_gust = df['wind_gusts_10m_max'].mean()

# Create 2025 monthly dataset
future_2025 = pd.DataFrame({
    'temperature_2m_min': [avg_temp_min]*12,
    'precipitation_sum': [avg_precip]*12,
    'wind_speed_10m_max': [avg_wind_speed]*12,
    'wind_gusts_10m_max': [avg_wind_gust]*12,
    'year': [2025]*12,
    'month': list(range(1,13))
})

# Predict using Random Forest
temp_2025_pred = rf_model.predict(future_2025)

print("\nPredicted Maximum Temperature for 2025 (Monthly):")
for m, temp in zip(range(1,13), temp_2025_pred):
    print(f"Month {m}: {temp:.2f} °C")

print("\nAverage Predicted Temperature for 2025:",
      round(temp_2025_pred.mean(), 2), "°C")

# 📈 Actual vs Predicted Plot (Temperature)

plt.figure(figsize=(8,6))

plt.scatter(y_test, rf_pred, alpha=0.5)

plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Actual vs Predicted Temperature (Random Forest)")

# Perfect prediction line
min_val = min(y_test.min(), rf_pred.min())
max_val = max(y_test.max(), rf_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.show()

# 🧠 Rainfall Prediction Model

# Features (excluding precipitation)
X_rain = df[['temperature_2m_max', 'temperature_2m_min',
             'wind_speed_10m_max', 'wind_gusts_10m_max',
             'year', 'month']]

y_rain = df['precipitation_sum']

# Split
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_rain, y_rain, test_size=0.2, random_state=42
)

# Train Random Forest
rf_rain = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_rain.fit(Xr_train, yr_train)

# Predict
rain_pred = rf_rain.predict(Xr_test)

print("\nRainfall Prediction Results:")
print("R2 Score:", r2_score(yr_test, rain_pred))
print("Mean Absolute Error:", mean_absolute_error(yr_test, rain_pred))