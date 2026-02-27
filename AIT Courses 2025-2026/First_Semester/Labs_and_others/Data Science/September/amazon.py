import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\tanak\Desktop\2025-2026\Data Science\September\amazon.csv")

# Convert numeric columns to proper dtype
num_cols = ["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing values in key columns
df = df.dropna(subset=["discounted_price", "rating"])

# 1. Discounted Price vs Rating
plt.figure(figsize=(8,6))
sns.regplot(data=df, x="discounted_price", y="rating",
            scatter_kws={'alpha':0.4}, line_kws={"color":"red"})
plt.title("Discounted Price vs Rating (Regression)")
plt.xlabel("Discounted Price")
plt.ylabel("Rating")
plt.show()
