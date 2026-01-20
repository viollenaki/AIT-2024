import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\tanak\Desktop\2025-2026\Data Science\September\diabetes.csv")

# Ensure numeric columns
num_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop missing values if any
df = df.dropna()

# 1. Glucose vs Outcome (Regression)
# plt.figure(figsize=(8,6))
# sns.regplot(data=df, x="Glucose", y="Outcome",
#             scatter_kws={'alpha':0.4}, line_kws={"color":"red"})
# plt.title("Glucose vs Outcome (Regression)")
# plt.xlabel("Glucose Level")
# plt.ylabel("Diabetes Outcome")
# plt.show()

# # 2. BMI vs Outcome
# plt.figure(figsize=(8,6))
# sns.regplot(data=df, x="BMI", y="Outcome",
#             scatter_kws={'alpha':0.4}, line_kws={"color":"blue"})
# plt.title("BMI vs Outcome (Regression)")
# plt.xlabel("BMI")
# plt.ylabel("Diabetes Outcome")
# plt.show()

# # 3. Age vs Outcome
# plt.figure(figsize=(8,6))
# sns.regplot(data=df, x="Age", y="Outcome",
#             scatter_kws={'alpha':0.4}, line_kws={"color":"green"})
# plt.title("Age vs Outcome (Regression)")
# plt.xlabel("Age")
# plt.ylabel("Diabetes Outcome")
# plt.show()

# 4. Diabetes Pedigree Function vs Outcome
# plt.figure(figsize=(8,6))
# sns.regplot(data=df, x="DiabetesPedigreeFunction", y="Outcome",
#             scatter_kws={'alpha':0.4}, line_kws={"color":"purple"})
# plt.title("Diabetes Pedigree Function vs Outcome (Regression)")
# plt.xlabel("Diabetes Pedigree Function")
# plt.ylabel("Diabetes Outcome")
# plt.show()

# 5. Pairwise regression for multiple variables
sns.lmplot(data=df, x="Glucose", y="BMI", hue="Outcome",
           height=6, aspect=1.3, scatter_kws={'alpha':0.5})
plt.title("Glucose vs BMI by Outcome (Regression)")
plt.show()
