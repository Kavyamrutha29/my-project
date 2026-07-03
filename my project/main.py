# =====================================================================
# INTEGRATED COMPREHENSIVE WORKFLOW GENERATOR FOR VS CODE (FIXED)
# =====================================================================

import os
import pickle
import numpy as np
import pandas as pd

# STEP 1: FORCE NON-INTERACTIVE BACKEND BEFORE IMPORTING PYPLOT
# This completely eliminates the "Backend tkagg is interactive" lockup error.
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# ---------------------------------------------------------------------
# 1. READ DATAFRAME FROM LOCAL WORKSPACE MANIFEST FILE
# ---------------------------------------------------------------------
csv_path = "HDI.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Missing file target allocation wrapper path: {csv_path}. Place the CSV in your workspace directory.")

# Read the local data asset file
Development = pd.read_csv(csv_path)

print("--- Dataset Initialized Successfully ---")
print(f"Total entries loaded: {Development.shape[0]} rows across {Development.shape[1]} metrics.\n")
print(Development.head())

# ---------------------------------------------------------------------
# 2. DATA EXPLORATION & PREPROCESSING
# ---------------------------------------------------------------------
# Extract subset data1 to protect structural plot clarity limit from assignment constraints
data1 = Development.head(20)

# Check and clear column formatting properties manually if required by pipeline parameters
features = ['Life_Expectancy', 'Expected_Years_Schooling', 'GNI_Per_Capita', 'Internet_Users_Pct', 'Carbon_Emission_Per_Capita']

# ---------------------------------------------------------------------
# 3. GENERATING 10+ MANDATORY EXPLORATORY CHARTS
# ---------------------------------------------------------------------
print("\n--- Plotting Model Diagnostics to static/plots/ Directory ---")
os.makedirs('static/plots', exist_ok=True)
sns.set_theme(style="whitegrid")

# Vis 1: Mean Schooling vs HDI Score (Strip Plot matching screenshot parameters)
plt.figure(figsize=(10, 5))
sns.stripplot(x="Expected_Years_Schooling", y="HDI", data=data1, jitter=True)
plt.xticks(rotation=90)
plt.title("1. Educational Access Metrics Variance Model (Top 20 rows)")
plt.tight_layout()
plt.savefig('static/plots/1_schooling_strip.png')
plt.close()

# Vis 2: Distribution Profile of Target metric
plt.figure()
sns.histplot(Development['HDI'], kde=True, color='purple')
plt.title("2. HDI Global Structural Target Distribution Profile")
plt.savefig('static/plots/2_hdi_distribution.png')
plt.close()

# Vis 3: Correlation Strength Heatmap Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(Development[features + ['HDI']].corr(), annot=True, cmap='mako')
plt.title("3. Feature Inter-Correlation Density Map Matrix")
plt.tight_layout()
plt.savefig('static/plots/3_correlation_matrix.png')
plt.close()

# Vis 4: Life Expectancy Tracking Scatter Map
plt.figure()
sns.scatterplot(data=Development, x='Life_Expectancy', y='HDI', color='green')
plt.title("4. Country Life Expectancy vs Index Scatter Matrix")
plt.savefig('static/plots/4_life_expectancy.png')
plt.close()

# Vis 5: Financial Wealth Transform Distribution Boxplot Comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(ax=axes[0], data=Development, y='GNI_Per_Capita', color='teal').set(title="Raw GNI Scale")
sns.boxplot(ax=axes[1], data=Development, y='GNI_Per_Capita', color='blue').set(yscale="log", title="Log Logarithmic Scale")
plt.suptitle("5. GNI Per Capita Scale Invariance Structural Testing")
plt.tight_layout()
plt.savefig('static/plots/5_gni_transformations.png')
plt.close()

# Vis 6: Internet Infrastructure Population Density Curve
plt.figure()
sns.kdeplot(data=Development, x='Internet_Users_Pct', fill=True, color='orange')
plt.title("6. Digital Connectivity Probability Mass Function Curve")
plt.savefig('static/plots/6_internet_connectivity.png')
plt.close()

# Vis 7: Pairwise Dimension Relationships Matrix Array
p = sns.pairplot(Development[['HDI', 'Life_Expectancy', 'Expected_Years_Schooling']])
p.savefig('static/plots/7_pairwise_dimensions.png')
plt.close()

# Vis 8: Environmental Carbon Production Hex Density Map
j = sns.jointplot(data=Development, x='Carbon_Emission_Per_Capita', y='HDI', kind='hex', color='darkblue')
j.savefig('static/plots/8_carbon_footprint.png')
plt.close()

# Vis 9: Grouped Regional Clusters Violin Metrics Map 
Development['Development_Group'] = pd.qcut(Development['HDI'], q=3, labels=['Lower', 'Mid', 'Upper'])
plt.figure()
sns.violinplot(data=Development, x='Development_Group', y='Expected_Years_Schooling', palette='muted')
plt.title("9. Schooling Variances Map Across Tiers")
plt.savefig('static/plots/9_violin_distribution.png')
plt.close()

# Vis 10: Index Growth Path Curve (ECDF)
plt.figure()
sns.ecdfplot(data=Development, x='HDI', color='red')
plt.title("10. Empirical Cumulative Trend Path Map (ECDF)")
plt.savefig('static/plots/10_ecdf_trend.png')
plt.close()

print("All 10 required project visualizations rendered successfully into 'static/plots/' folder.")


# ---------------------------------------------------------------------
## ---------------------------------------------------------------------
# 4. TRAINING & EVALUATION MATRIX (reg Model Setup)
# ---------------------------------------------------------------------
print("\n--- Segregating Matrix Blocks & Fitting Estimators ---")
X = Development[features]
y = Development['HDI']

# Train and test split configuration setup matching assignment rules
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the Model using the 'reg' identifier variable name
reg = LinearRegression()
reg.fit(x_train, y_train)

# Evaluation Metric Calculations
y_pred = reg.predict(x_test)
print(f"R2 Accuracy Matrix Determination:       {metrics.r2_score(y_test, y_pred):.4f}")
print(f"MAE Performance Calculation Metric:     {metrics.mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE Calculation Value Target:           {metrics.mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE Variance Deviation Index:          {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.4f}")


# =====================================================================
# ADDED: CLASSIFICATION FUNCTION FOR INTERNSHIP SCENARIOS
# =====================================================================
def classify_hdi(score):
    """Maps the predicted continuous regression score to the 4 mandated tiers."""
    if score >= 0.800:
        return "Very High Human Development"
    elif score >= 0.700:
        return "High Human Development"
    elif score >= 0.550:
        return "Medium Human Development"
    else:
        return "Low Human Development"

# Sanity check validation using test data
sample_score = y_pred[0]
print(f"\nSample Predicted HDI Score: {sample_score:.4f}")
print(f"Mapped Classification Tier: {classify_hdi(sample_score)}")
# =====================================================================


# Serializing model map parameters explicitly using the built-in pickle library
pickle.dump(reg, open('HDI.pkl', 'wb'))
print("\nState files compiled successfully. Parameter model map exported as 'HDI.pkl'.")