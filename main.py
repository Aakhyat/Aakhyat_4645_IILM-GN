import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================
# 1. LOAD DATA
# ================================
df = pd.read_csv("India_Weather_2000_2024_Cleaned.csv")

# Convert date
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ================================
# 2. (OPTIONAL) EDA SECTION
# ================================
EDA = False  # Set True if you want graphs

if EDA:
    # Yearly Temperature Trend
    yearly_temp = df.groupby('year')['temperature_2m_max'].mean()
    plt.plot(yearly_temp)
    plt.title("Yearly Temperature Trend")
    plt.show()

    # Yearly Rainfall Trend
    yearly_rain = df.groupby('year')['precipitation_sum'].sum()
    plt.plot(yearly_rain)
    plt.title("Yearly Rainfall Trend")
    plt.show()

# ================================
# 3. FEATURE SELECTION
# ================================
features = ['temperature_2m_min', 'precipitation_sum',
            'wind_speed_10m_max', 'wind_gusts_10m_max',
            'year', 'month']

X = df[features]
y = df['temperature_2m_max']

# ================================
# 4. TIME-BASED SPLIT (IMPORTANT)
# ================================
train = df[df['year'] < 2020]
test = df[df['year'] >= 2020]

X_train = train[features]
y_train = train['temperature_2m_max']

X_test = test[features]
y_test = test['temperature_2m_max']

# ================================
# 5. MODELS
# ================================
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\nLinear Regression:")
print("R2:", r2_score(y_test, lr_pred))
print("MAE:", mean_absolute_error(y_test, lr_pred))

# Decision Tree
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\nDecision Tree:")
print("R2:", r2_score(y_test, dt_pred))
print("MAE:", mean_absolute_error(y_test, dt_pred))

# Random Forest (Best Model)
rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest:")
print("R2:", r2_score(y_test, rf_pred))
print("MAE:", mean_absolute_error(y_test, rf_pred))

# ================================
# 6. FEATURE IMPORTANCE
# ================================
importance = pd.Series(rf.feature_importances_, index=features)
importance = importance.sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure()
importance.plot(kind='bar')
plt.title("Feature Importance")
plt.show()

# ================================
# 7. ACTUAL VS PREDICTED
# ================================
plt.figure()
plt.scatter(y_test, rf_pred, alpha=0.5)

min_val = min(y_test.min(), rf_pred.min())
max_val = max(y_test.max(), rf_pred.max())

plt.plot([min_val, max_val], [min_val, max_val])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Random Forest)")
plt.show()

# ================================
# 8. ERROR DISTRIBUTION
# ================================
errors = y_test - rf_pred

plt.figure()
plt.hist(errors, bins=30)
plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

# ================================
# 9. 2025 TEMPERATURE PREDICTION
# ================================
avg_values = {
    'temperature_2m_min': df['temperature_2m_min'].mean(),
    'precipitation_sum': df['precipitation_sum'].mean(),
    'wind_speed_10m_max': df['wind_speed_10m_max'].mean(),
    'wind_gusts_10m_max': df['wind_gusts_10m_max'].mean()
}

future_2025 = pd.DataFrame({
    'temperature_2m_min': [avg_values['temperature_2m_min']]*12,
    'precipitation_sum': [avg_values['precipitation_sum']]*12,
    'wind_speed_10m_max': [avg_values['wind_speed_10m_max']]*12,
    'wind_gusts_10m_max': [avg_values['wind_gusts_10m_max']]*12,
    'year': [2025]*12,
    'month': list(range(1,13))
})

pred_2025 = rf.predict(future_2025)

print("\n2025 Temperature Prediction:")
for i, val in enumerate(pred_2025, 1):
    print(f"Month {i}: {val:.2f} °C")

print("Average 2025 Temperature:", round(pred_2025.mean(), 2), "°C")

# 📈 Plot 2025 Temperature Prediction

months = list(range(1, 13))

plt.figure()
plt.plot(months, pred_2025, marker='o')

plt.title("Predicted Monthly Temperature for 2025")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")

plt.xticks(months)
plt.grid()

plt.show()

# Historical monthly average
monthly_avg = df.groupby('month')['temperature_2m_max'].mean()

plt.figure()
plt.plot(months, pred_2025, marker='o', label="Predicted 2025")
plt.plot(months, monthly_avg, marker='x', label="Historical Avg")

plt.title("2025 Prediction vs Historical Average")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()

# ================================
# 10. RAINFALL MODEL
# ================================
X_rain = df[['temperature_2m_max', 'temperature_2m_min',
             'wind_speed_10m_max', 'wind_gusts_10m_max',
             'year', 'month']]

y_rain = df['precipitation_sum']

train_r = df[df['year'] < 2020]
test_r = df[df['year'] >= 2020]

Xr_train = train_r[X_rain.columns]
yr_train = train_r['precipitation_sum']

Xr_test = test_r[X_rain.columns]
yr_test = test_r['precipitation_sum']

rf_rain = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_rain.fit(Xr_train, yr_train)
rain_pred = rf_rain.predict(Xr_test)

print("\nRainfall Model:")
print("R2:", r2_score(yr_test, rain_pred))
print("MAE:", mean_absolute_error(yr_test, rain_pred))

# ================================
# END
# ================================
print("\nProject Execution Completed Successfully 🚀")


import joblib

joblib.dump(rf, "model.pkl")

