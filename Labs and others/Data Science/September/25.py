import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем данные
df = pd.read_csv(r"C:\Users\tanak\Desktop\2025-2026\Data Science\September\Finance_data.csv")

# 1. Общая информация
print(df.head())
print(df.info())

# 2. Распределение по полу
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="gender", palette="Set2")
plt.title("Распределение по полу")
plt.show()

# 3. Возрастное распределение
plt.figure(figsize=(6,4))
sns.histplot(df["age"], bins=10, kde=True, color="skyblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# 4. Популярность инвестиционных направлений
investment_cols = [
    "Mutual_Funds", "Equity_Market", "Debentures",
    "Government_Bonds", "Fixed_Deposits", "PPF", "Gold", "Stock_Marktet"
]

investment_counts = df[investment_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=investment_counts.values, y=investment_counts.index, palette="viridis")
plt.title("Популярность инвестиционных направлений")
plt.xlabel("Количество инвесторов")
plt.ylabel("Инструмент")
plt.show()

# 5. Цели сбережений (Objective)
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="Objective", order=df["Objective"].value_counts().index, palette="pastel")
plt.xticks(rotation=45)
plt.title("Цели сбережений")
plt.show()

# 6. Ожидания инвесторов (Expect)
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Expect", order=df["Expect"].value_counts().index, palette="coolwarm")
plt.title("Ожидания инвесторов")
plt.xticks(rotation=45)
plt.show()

# 7. Сравнение возраста и выбора инвестиционного инструмента
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="Avenue", y="age", palette="Set3")
plt.title("Возраст и выбор инвестиционного инструмента")
plt.xticks(rotation=45)
plt.show()
