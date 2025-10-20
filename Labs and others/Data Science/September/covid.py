import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(r"C:\Users\tanak\Desktop\2025-2026\Data Science\September\country_wise_latest.csv")

# 1. Top-10 countries by confirmed cases
top_confirmed = df.nlargest(10, "Confirmed")

plt.figure(figsize=(10,6))
sns.barplot(data=top_confirmed, x="Confirmed", y="Country/Region", palette="Reds_r")
plt.title("Top 10 Countries by Confirmed Cases")
plt.xlabel("Confirmed Cases")
plt.ylabel("Country")
plt.show()

# 2. Top-10 countries by deaths
top_deaths = df.nlargest(10, "Deaths")

plt.figure(figsize=(10,6))
sns.barplot(data=top_deaths, x="Deaths", y="Country/Region", palette="Greys_r")
plt.title("Top 10 Countries by Deaths")
plt.xlabel("Deaths")
plt.ylabel("Country")
plt.show()

# 3. Mortality rate (deaths per 100 cases)
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="Confirmed", y="Deaths / 100 Cases", 
                hue="WHO Region", size="Confirmed", sizes=(40, 300))
plt.title("Mortality Rate (Deaths per 100 Cases)")
plt.xlabel("Confirmed Cases")
plt.ylabel("Mortality Rate (%)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# 4. Recovered vs Deaths
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="Recovered", y="Deaths", hue="WHO Region", alpha=0.7)
plt.title("Recovered vs Deaths")
plt.xlabel("Recovered")
plt.ylabel("Deaths")
plt.show()

# 5. Weekly change in confirmed cases
plt.figure(figsize=(10,6))
sns.barplot(data=df.sort_values("1 week change", ascending=False).head(10),
            x="1 week change", y="Country/Region", palette="coolwarm")
plt.title("Top 10 Countries by Weekly Increase")
plt.xlabel("Weekly Change")
plt.ylabel("Country")
plt.show()

# 6. WHO Regions comparison
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="WHO Region", y="Confirmed", palette="Set2")
plt.yscale("log")  # log scale for better visualization
plt.title("Distribution of Confirmed Cases by WHO Region")
plt.xlabel("WHO Region")
plt.ylabel("Confirmed Cases (log scale)")
plt.show()
