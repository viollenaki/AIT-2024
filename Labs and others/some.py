import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Создаем даты
dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")
n = len(dates)

# Генерируем продажи с трендом, сезонностью и шумом
np.random.seed(42)
trend = np.linspace(50, 300, n)  # растущий тренд
seasonality = 20 * np.sin(2 * np.pi * np.arange(n)/365)  # годовая сезонность
noise = np.random.normal(0, 15, n)  # случайный шум

sales = trend + seasonality + noise

# Создаем DataFrame
df = pd.DataFrame({"Date": dates, "Sales": sales})
df.set_index("Date", inplace=True)

# Смотрим первые строки
print(df.head())

# Сохраняем в CSV
df.to_csv("sales_timeseries.csv")

# Визуализация
plt.figure(figsize=(12,6))
plt.plot(df.index, df["Sales"], label="Sales")
plt.title("Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()
# Скользящее среднее