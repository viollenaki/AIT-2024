import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import log2

# Load CSVs
country_df = pd.read_csv('country_level_data.csv')
param_dist_df = pd.read_csv('parameter_distributions.csv')
perf_matrix_df = pd.read_csv('performance_matrix.csv')  # For reference; we'll recreate it

# Step 1: Compute Parameters from Raw Data
# Parameter definitions
country_df['Production_Capacity'] = country_df['Cattle_Per_1000_People']
country_df['Economic_Accessibility'] = (country_df['GDP_Per_Capita_USD'] * country_df['Urbanization_Rate_Pct']) / 100
country_df['Health_Cultural_Barrier'] = 100 - country_df['Lactose_Intol_Rate_Pct']
country_df['Policy_Support'] = country_df['Govt_Ag_Spending_Pct_GDP']
country_df['Overall_Market_Potential'] = (
    country_df['Production_Capacity'] + 
    country_df['Economic_Accessibility'] + 
    country_df['Health_Cultural_Barrier'] + 
    country_df['Policy_Support']
) / 4

# Normalize scores (0-100) for matrix
def normalize(col):
    return 100 * (col - col.min()) / (col.max() - col.min())

params = ['Production_Capacity', 'Economic_Accessibility', 'Health_Cultural_Barrier', 'Policy_Support', 'Overall_Market_Potential']
for param in params:
    country_df[param + '_Norm'] = normalize(country_df[param])

# Step 2: Compute Gini Index (for inequality in parameter distributions)
def gini_index(values):
    # Sort and compute Gini for a list of values (0=equal, 1=unequal)
    sorted_vals = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_vals)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_vals)) / (n * cumsum[-1])) - (n+1)/n
    return gini

gini_results = {}
for param in params:
    gini_results[param] = gini_index(country_df[param].values)
print("Gini Index Results:", gini_results)

# Step 3: Compute Entropy and Information Gain
# Entropy function
def entropy(probs):
    return -sum(p * log2(p) for p in probs if p > 0)

# Total Entropy (based on Consumption_Class)
classes = param_dist_df['Consumption_Class'].value_counts(normalize=True)
total_entropy = entropy(classes.values)
print("Total Entropy:", total_entropy)

# Information Gain for each parameter bin
info_gain = {}
for col in param_dist_df.columns[1:-1]:  # Exclude Country and Consumption_Class
    gain = total_entropy
    for bin_val in ['Low', 'High']:
        subset = param_dist_df[param_dist_df[col] == bin_val]
        if len(subset) == 0:
            continue
        subset_classes = subset['Consumption_Class'].value_counts(normalize=True)
        subset_entropy = entropy(subset_classes.values)
        gain -= (len(subset) / len(param_dist_df)) * subset_entropy
    info_gain[col.replace('_Bin', '')] = gain
print("Information Gain Results:", info_gain)

# Step 4: Build Performance Matrix (Kyrgyzstan vs Top 10 Avg)
top10_df = country_df[country_df['Rank_Milk_Cons'] <= 10]
kyrgyzstan = country_df[country_df['Country'] == 'Kyrgyzstan'].iloc[0]

matrix_data = {
    'Parameter': params,
    'Kyrgyzstan_Score_Normalized': [kyrgyzstan[p + '_Norm'] for p in params],
    'Top10_Avg_Score_Normalized': [top10_df[p + '_Norm'].mean() for p in params],
}
matrix_data['Gap'] = np.array(matrix_data['Kyrgyzstan_Score_Normalized']) - np.array(matrix_data['Top10_Avg_Score_Normalized'])
# Add plans from original (or placeholder)
matrix_data['Improvement_Plan'] = perf_matrix_df['Improvement_Plan'].tolist()  # Reuse from CSV

perf_matrix = pd.DataFrame(matrix_data)
perf_matrix.to_csv('recomputed_performance_matrix.csv', index=False)
print("\nPerformance Matrix:\n", perf_matrix)

# Step 5: Visualizations
# 1. Bar Chart: Parameter Comparison
plt.figure(figsize=(12, 8))
perf_matrix.plot(kind='bar', x='Parameter', y=['Kyrgyzstan_Score_Normalized', 'Top10_Avg_Score_Normalized'])
plt.title('Parameter Comparison: Kyrgyzstan vs Top 10 Avg', fontsize=16)
plt.ylabel('Normalized Score (0-100)', fontsize=12)
plt.xlabel('Parameter', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Scatter Plot: Correlation of Economic Accessibility to Milk Consumption
plt.figure(figsize=(10, 8))
plt.scatter(country_df['Economic_Accessibility'], country_df['Milk_Cons_Per_Capita_kg_year'], color='blue', alpha=0.7, s=50)
kyrgyz_idx = country_df[country_df['Country'] == 'Kyrgyzstan'].index[0]
plt.scatter(country_df['Economic_Accessibility'][kyrgyz_idx], country_df['Milk_Cons_Per_Capita_kg_year'][kyrgyz_idx], color='red', s=100, label='Kyrgyzstan')
plt.title('Economic Accessibility vs Milk Consumption', fontsize=16)
plt.xlabel('Economic Accessibility Score', fontsize=12)
plt.ylabel('Milk Consumption Per Capita (kg/year)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Pie Chart: Entropy Information Gain Breakdown
gains = list(info_gain.values())
labels = [label.replace('_', '\n') for label in info_gain.keys()]  # Break long labels
plt.figure(figsize=(10, 8))
plt.pie(gains, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 10})
plt.title('Information Gain Breakdown by Parameter', fontsize=16)
plt.tight_layout()
plt.savefig('pie_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Heatmap: Gini Index Across Parameters
gini_df = pd.DataFrame({'Parameter': list(gini_results.keys()), 'Gini': list(gini_results.values())})
# Shorten parameter names for better display
gini_df['Parameter'] = gini_df['Parameter'].str.replace('_', '\n')
gini_pivot = gini_df.set_index('Parameter')  # For heatmap
plt.figure(figsize=(8, 10))
sns.heatmap(gini_pivot, annot=True, cmap='RdYlGn', vmin=0, vmax=1, 
            fmt='.2f', annot_kws={'fontsize': 12}, cbar_kws={'label': 'Gini Index'})
plt.title('Gini Index Heatmap', fontsize=16)
plt.ylabel('Parameter', fontsize=12)
plt.xlabel('Gini Index Value', fontsize=12)
plt.tight_layout()
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations saved as PNG files.")