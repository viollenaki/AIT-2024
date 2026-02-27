"""
Breast Cancer Classification: Linear Regression vs Logistic Regression
Comparison using ROC AUC and Gini Coefficient
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score

# Specific features to use (as requested)
SELECTED_FEATURES = [
    'texture_worst',

'smoothness_worst',

'symmetry_worst',

'fractal_dimension_worst'
]
N_FEATURES = len(SELECTED_FEATURES)

# ============================================
# 1. Load the dataset
# ============================================
print("=" * 60)
print("BREAST CANCER CLASSIFICATION MODEL COMPARISON")
print("=" * 60)

# Load data
df = pd.read_csv('breast-cancer.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ============================================
# 2. Preprocess the data
# ============================================
print("\n" + "-" * 60)
print("DATA PREPROCESSING")
print("-" * 60)

# Encode the binary target variable (diagnosis: M=1, B=0)
label_encoder = LabelEncoder()
df['diagnosis_encoded'] = label_encoder.fit_transform(df['diagnosis'])
print(f"\nTarget encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
print(f"  - M (Malignant) = 1")
print(f"  - B (Benign) = 0")

# Remove non-predictive columns (id and original diagnosis)
columns_to_drop = ['id', 'diagnosis']
X = df.drop(columns=columns_to_drop + ['diagnosis_encoded'])
y = df['diagnosis_encoded']

print(f"\nTotal features available: {X.shape[1]} columns")
print(f"Samples: {X.shape[0]}")

# Check for missing values
missing_values = X.isnull().sum().sum()
print(f"Missing values: {missing_values}")

# ============================================
# 3. Feature Selection - Use specified features
# ============================================
print("\n" + "-" * 60)
print("FEATURE SELECTION")
print("-" * 60)

# Use the specified features
print(f"\nUsing {N_FEATURES} specified features:")
for i, feature in enumerate(SELECTED_FEATURES, 1):
    print(f"  {i}. {feature}")

# Keep only selected features
X = X[SELECTED_FEATURES]
print(f"\nFeatures used for training: {N_FEATURES}")

# ============================================
# 4. Split data into train/test sets
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Train positive ratio: {y_train.mean():.2%}")
print(f"Test positive ratio: {y_test.mean():.2%}")

# ============================================
# 5. Scale numerical features
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures scaled using StandardScaler")

# ============================================
# 6. Train models
# ============================================
print("\n" + "-" * 60)
print("MODEL TRAINING")
print("-" * 60)

# Linear Regression
print("\nTraining Linear Regression...")
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)

# Logistic Regression
print("Training Logistic Regression...")
logistic_reg = LogisticRegression(max_iter=1000, random_state=42)
logistic_reg.fit(X_train_scaled, y_train)

print("Both models trained successfully!")

# ============================================
# 7. Make predictions and compute metrics
# ============================================
print("\n" + "-" * 60)
print("MODEL EVALUATION")
print("-" * 60)

# Linear Regression predictions (raw values, clipped to [0,1] for probability interpretation)
y_pred_linear = linear_reg.predict(X_test_scaled)
y_pred_linear_clipped = np.clip(y_pred_linear, 0, 1)  # Clip to valid probability range

# Logistic Regression predictions (probability scores)
y_pred_logistic = logistic_reg.predict_proba(X_test_scaled)[:, 1]

# Compute ROC AUC scores
roc_auc_linear = roc_auc_score(y_test, y_pred_linear_clipped)
roc_auc_logistic = roc_auc_score(y_test, y_pred_logistic)

# Compute Gini coefficients (Gini = 2 * AUC - 1)
gini_linear = 2 * roc_auc_linear - 1
gini_logistic = 2 * roc_auc_logistic - 1

# Compute ROC curves
fpr_linear, tpr_linear, _ = roc_curve(y_test, y_pred_linear_clipped)
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_pred_logistic)

# Compute accuracy scores
y_pred_linear_class = (y_pred_linear_clipped >= 0.5).astype(int)
y_pred_logistic_class = logistic_reg.predict(X_test_scaled)
acc_linear = accuracy_score(y_test, y_pred_linear_class)
acc_logistic = accuracy_score(y_test, y_pred_logistic_class)

# Compute confusion matrices
cm_linear = confusion_matrix(y_test, y_pred_linear_class)
cm_logistic = confusion_matrix(y_test, y_pred_logistic_class)

# ============================================
# 8. Display results
# ============================================
print("\n" + "=" * 60)
print("RESULTS COMPARISON")
print("=" * 60)

print("\n┌─────────────────────────┬─────────────┬─────────────┐")
print("│         Model           │   ROC AUC   │    Gini     │")
print("├─────────────────────────┼─────────────┼─────────────┤")
print(f"│ Linear Regression       │   {roc_auc_linear:.4f}    │   {gini_linear:.4f}    │")
print(f"│ Logistic Regression     │   {roc_auc_logistic:.4f}    │   {gini_logistic:.4f}    │")
print("└─────────────────────────┴─────────────┴─────────────┘")

# Determine the winner
print("\n" + "-" * 60)
print("ANALYSIS")
print("-" * 60)

print("\n📊 ROC AUC Interpretation:")
print("   - 0.5 = Random guessing (no discrimination)")
print("   - 0.7-0.8 = Acceptable discrimination")
print("   - 0.8-0.9 = Excellent discrimination")
print("   - 0.9+ = Outstanding discrimination")

print("\n📈 Gini Coefficient Interpretation:")
print("   - Gini = 2 × AUC - 1")
print("   - Range: [-1, 1], higher is better")
print("   - 0 = Random model, 1 = Perfect model")

# ============================================
# 9. Conclusion
# ============================================
print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

auc_diff = abs(roc_auc_logistic - roc_auc_linear)
gini_diff = abs(gini_logistic - gini_linear)

if roc_auc_logistic > roc_auc_linear:
    winner = "Logistic Regression"
    winner_auc = roc_auc_logistic
    winner_gini = gini_logistic
    loser = "Linear Regression"
    loser_auc = roc_auc_linear
else:
    winner = "Linear Regression"
    winner_auc = roc_auc_linear
    winner_gini = gini_linear
    loser = "Logistic Regression"
    loser_auc = roc_auc_logistic

print(f"""
✅ WINNER: {winner}
   - ROC AUC: {winner_auc:.4f}
   - Gini: {winner_gini:.4f}

📋 Performance Difference:
   - AUC difference: {auc_diff:.4f} ({auc_diff*100:.2f}%)
   - Gini difference: {gini_diff:.4f}

💡 Key Insights:
   1. Logistic Regression is specifically designed for binary classification,
      outputting proper probabilities in the range [0, 1].
   
   2. Linear Regression outputs continuous values that can fall outside [0, 1],
      requiring clipping for probability interpretation.
   
   3. For classification tasks, Logistic Regression is the appropriate choice
      as it models the log-odds of the target variable.
   
   4. Both models achieve high AUC scores on this dataset, indicating that
      the breast cancer features are highly predictive of the diagnosis.

🎯 Recommendation:
   Use {winner} for this classification task due to:
   - Better probabilistic interpretation
   - Proper handling of binary outcomes
   - Higher discriminative ability (Gini = {winner_gini:.4f})
""")

print("=" * 60)
print("END OF ANALYSIS")
print("=" * 60)

# ============================================
# 10. Visualizations
# ============================================
print("\nGenerating visualizations...")

# Set up the figure with subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle(f'Breast Cancer Classification: Linear vs Logistic Regression\n(Using Top {N_FEATURES} Features)', 
             fontsize=14, fontweight='bold')

# Plot 1: ROC Curves Comparison
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(fpr_linear, tpr_linear, 'b-', linewidth=2, 
         label=f'Linear Regression (AUC = {roc_auc_linear:.4f})')
ax1.plot(fpr_logistic, tpr_logistic, 'r-', linewidth=2, 
         label=f'Logistic Regression (AUC = {roc_auc_logistic:.4f})')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax1.set_xlabel('False Positive Rate', fontsize=10)
ax1.set_ylabel('True Positive Rate', fontsize=10)
ax1.set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1.05])

# Plot 2: AUC and Gini Comparison Bar Chart
ax2 = fig.add_subplot(2, 3, 2)
x = np.arange(2)
width = 0.35
bars1 = ax2.bar(x - width/2, [roc_auc_linear, gini_linear], width, 
                label='Linear Regression', color='steelblue', edgecolor='black')
bars2 = ax2.bar(x + width/2, [roc_auc_logistic, gini_logistic], width, 
                label='Logistic Regression', color='coral', edgecolor='black')
ax2.set_ylabel('Score', fontsize=10)
ax2.set_title('ROC AUC & Gini Coefficient Comparison', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['ROC AUC', 'Gini Coefficient'])
ax2.legend(loc='lower right', fontsize=9)
ax2.set_ylim([0.9, 1.02])
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax2.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# Plot 3: Feature Importance (Top N)
# ax3 = fig.add_subplot(2, 3, 3)
# top_features = all_feature_scores.head(N_FEATURES)
# colors = plt.cm.viridis(np.linspace(0.2, 0.8, N_FEATURES))
# bars3 = ax3.barh(range(N_FEATURES), top_features['Score'].values[::-1], color=colors[::-1], edgecolor='black')
# ax3.set_yticks(range(N_FEATURES))
# ax3.set_yticklabels(top_features['Feature'].values[::-1], fontsize=9)
# ax3.set_xlabel('ANOVA F-Score', fontsize=10)
# ax3.set_title(f'Top {N_FEATURES} Selected Features', fontsize=12, fontweight='bold')
# ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Confusion Matrix - Linear Regression
ax4 = fig.add_subplot(2, 3, 4)
im4 = ax4.imshow(cm_linear, interpolation='nearest', cmap='Blues')
ax4.set_title(f'Confusion Matrix - Linear Regression\n(Accuracy: {acc_linear:.2%})', 
              fontsize=12, fontweight='bold')
ax4.set_xlabel('Predicted Label', fontsize=10)
ax4.set_ylabel('True Label', fontsize=10)
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Benign (0)', 'Malignant (1)'])
ax4.set_yticklabels(['Benign (0)', 'Malignant (1)'])
# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax4.text(j, i, cm_linear[i, j], ha="center", va="center", 
                       color="white" if cm_linear[i, j] > cm_linear.max()/2 else "black",
                       fontsize=14, fontweight='bold')
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

# Plot 5: Confusion Matrix - Logistic Regression
ax5 = fig.add_subplot(2, 3, 5)
im5 = ax5.imshow(cm_logistic, interpolation='nearest', cmap='Oranges')
ax5.set_title(f'Confusion Matrix - Logistic Regression\n(Accuracy: {acc_logistic:.2%})', 
              fontsize=12, fontweight='bold')
ax5.set_xlabel('Predicted Label', fontsize=10)
ax5.set_ylabel('True Label', fontsize=10)
ax5.set_xticks([0, 1])
ax5.set_yticks([0, 1])
ax5.set_xticklabels(['Benign (0)', 'Malignant (1)'])
ax5.set_yticklabels(['Benign (0)', 'Malignant (1)'])
# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax5.text(j, i, cm_logistic[i, j], ha="center", va="center", 
                       color="white" if cm_logistic[i, j] > cm_logistic.max()/2 else "black",
                       fontsize=14, fontweight='bold')
plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

# Plot 6: Model Comparison Summary
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
Model Comparison Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Features Used: {N_FEATURES} (out of 30)

Linear Regression:
  • ROC AUC: {roc_auc_linear:.4f}
  • Gini:    {gini_linear:.4f}
  • Accuracy: {acc_linear:.2%}

Logistic Regression:
  • ROC AUC: {roc_auc_logistic:.4f}
  • Gini:    {gini_logistic:.4f}
  • Accuracy: {acc_logistic:.2%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Winner: {winner}
"""
ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('model_comparison_results.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved as 'model_comparison_results.png'")
plt.show()

# ============================================
# 11. Additional Plot: Prediction Distribution
# ============================================
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle(f'Prediction Score Distributions (Top {N_FEATURES} Features)', fontsize=14, fontweight='bold')

# Linear Regression predictions distribution
ax7 = axes[0]
ax7.hist(y_pred_linear_clipped[y_test == 0], bins=20, alpha=0.7, color='green', 
         label='Benign (True)', edgecolor='black')
ax7.hist(y_pred_linear_clipped[y_test == 1], bins=20, alpha=0.7, color='red', 
         label='Malignant (True)', edgecolor='black')
ax7.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
ax7.set_xlabel('Predicted Probability', fontsize=10)
ax7.set_ylabel('Frequency', fontsize=10)
ax7.set_title('Linear Regression Predictions', fontsize=12, fontweight='bold')
ax7.legend(loc='upper center', fontsize=9)
ax7.grid(True, alpha=0.3)

# Logistic Regression predictions distribution
ax8 = axes[1]
ax8.hist(y_pred_logistic[y_test == 0], bins=20, alpha=0.7, color='green', 
         label='Benign (True)', edgecolor='black')
ax8.hist(y_pred_logistic[y_test == 1], bins=20, alpha=0.7, color='red', 
         label='Malignant (True)', edgecolor='black')
ax8.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
ax8.set_xlabel('Predicted Probability', fontsize=10)
ax8.set_ylabel('Frequency', fontsize=10)
ax8.set_title('Logistic Regression Predictions', fontsize=12, fontweight='bold')
ax8.legend(loc='upper center', fontsize=9)
ax8.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('prediction_distributions.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved as 'prediction_distributions.png'")
plt.show()

print("\n🎉 All visualizations generated successfully!")
