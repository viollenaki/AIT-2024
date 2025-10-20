
# Оптимизированный код прогнозирования продаж
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Генерация данных за последние 10 лет (если файла нет или требуется обновить)
import os
if not os.path.exists("sales_timeseries.csv"):
	# Параметры генерации
	start_date = pd.to_datetime("2015-01-01")
	end_date = pd.to_datetime("2025-12-31")
	dates = pd.date_range(start=start_date, end=end_date, freq="D")
	n = len(dates)
	np.random.seed(42)
	trend = np.linspace(50, 300, n)
	seasonality = 20 * np.sin(2 * np.pi * np.arange(n)/365)
	noise = np.random.normal(0, 15, n)
	sales = trend + seasonality + noise
	df = pd.DataFrame({"Date": dates, "Sales": sales})
	df.to_csv("sales_timeseries.csv", index=False)

# Чтение данных за 10 лет
df = pd.read_csv("sales_timeseries.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Преобразование дат в числовой формат для регрессии
df["DayNumber"] = np.arange(len(df))
X = df[["DayNumber"]]
y = df["Sales"]

# Обучение модели
model = LinearRegression()
model.fit(X, y)

# Прогноз на 2 года вперёд
future_periods = 2 * 365  # 2 года
future_days = np.arange(len(df), len(df) + future_periods).reshape(-1, 1)
future_sales = model.predict(future_days)
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_periods)

# Визуализация
plt.figure(figsize=(16, 7))
plt.plot(df.index, df["Sales"], label="Fact (10 years)")
plt.plot(future_dates, future_sales, color="red", linestyle="--", label="Prediction (2 years)")
plt.title("Sales Forecast (10 years of data)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()
