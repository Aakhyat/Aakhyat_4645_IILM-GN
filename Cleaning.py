import pandas as pd
import numpy as np

# 1️⃣ Load dataset
df = pd.read_csv(r"C:\Users\aakhy\OneDrive\Desktop\DataScience\india_2000_2024_daily_weather.csv")

# 2️⃣ Make a copy
df_clean = df.copy()

# 3️⃣ Clean & standardize column names
df_clean.columns = (
    df_clean.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)

# 4️⃣ Convert date column to datetime
df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")

# 5️⃣ Remove duplicates
df_clean.drop_duplicates(inplace=True)

# 6️⃣ Handle impossible values
# Rain and precipitation can't be negative
df_clean["rain_sum"] = df_clean["rain_sum"].clip(lower=0)
df_clean["precipitation_sum"] = df_clean["precipitation_sum"].clip(lower=0)

# Temperatures: remove absurd values (India won't have -50 or +60 normally)
df_clean = df_clean[
    (df_clean["temperature_2m_max"] > -10) & (df_clean["temperature_2m_max"] < 60) &
    (df_clean["temperature_2m_min"] > -10) & (df_clean["temperature_2m_min"] < 60)
]

# Wind speeds: remove absurd values
df_clean = df_clean[
    (df_clean["wind_speed_10m_max"] >= 0) & (df_clean["wind_speed_10m_max"] < 200)
]

# 7️⃣ Remove extreme outliers using 99th percentile (for some numeric columns)
for col in [
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "precipitation_sum",
    "rain_sum"
]:
    upper = df_clean[col].quantile(0.99)
    df_clean = df_clean[df_clean[col] <= upper]

# 8️⃣ Final validation
print("Missing values:\n", df_clean.isnull().sum())
print("\nFinal shape:", df_clean.shape)
print("\nData types:\n", df_clean.dtypes)

# 9️⃣ Save cleaned dataset
df_clean.to_csv("India_Weather_2000_2024_Cleaned.csv", index=False)

print("\n✅ Cleaned file saved as: India_Weather_2000_2024_Cleaned.csv")
