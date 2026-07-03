
import os
import io
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 1. RAW RESOURCE EMBEDDED DATASET 
csv_data = """Country,Life_Expectancy,Expected_Years_Schooling,GNI_Per_Capita,Internet_Users_Pct,Carbon_Emission_Per_Capita,HDI
Norway,82.4,18.1,68000,97.2,8.3,0.957
Switzerland,83.8,16.5,69000,93.4,4.1,0.955
Ireland,82.3,18.7,61000,92.1,7.2,0.955
Germany,81.3,17.0,55000,89.6,8.5,0.947
Australia,83.4,22.0,49000,86.5,15.4,0.944
Iceland,83.0,19.1,54000,99.0,5.7,0.949
Sweden,82.8,19.4,53000,94.5,3.8,0.945
Singapore,83.6,16.4,52000,92.0,8.4,0.938
Netherlands,82.2,18.5,57000,95.2,8.1,0.944
Canada,82.4,16.2,48000,92.7,15.1,0.929
United States,78.9,16.3,63000,89.0,14.7,0.926
United Kingdom,81.3,17.5,46000,94.8,5.2,0.932
Finland,81.9,19.3,47000,91.5,4.6,0.938
New Zealand,82.3,18.8,42000,90.8,6.5,0.931
Belgium,81.6,19.6,48000,89.3,7.2,0.931
Japan,84.6,15.2,41000,92.7,8.7,0.919
Austria,81.5,16.0,56000,87.5,6.9,0.922
France,82.5,15.6,46000,85.6,4.5,0.901
Israel,82.9,16.0,40000,86.8,5.9,0.919
Luxembourg,82.3,14.3,71000,97.1,13.2,0.916
South Korea,83.0,16.5,43000,96.0,11.7,0.916
Spain,83.3,17.9,38000,86.1,5.0,0.904
Czechia,79.3,16.8,38000,81.3,9.2,0.900
Italy,83.5,16.1,42000,74.4,5.3,0.892
United Arab Emirates,78.0,14.3,67000,98.5,20.0,0.890
Greece,82.2,17.9,29000,72.5,6.0,0.888
Cyprus,81.0,15.1,37000,84.4,5.8,0.887
Lithuania,75.9,16.6,35000,80.1,4.3,0.882
Poland,78.7,16.4,31000,78.2,7.5,0.880
Chile,80.2,16.4,24000,82.3,4.6,0.851
Saudi Arabia,75.1,16.1,49000,93.3,17.1,0.854
Portugal,82.0,16.5,33000,75.1,4.4,0.864
Latvia,75.3,16.1,30000,81.5,3.7,0.866
Croatia,78.5,15.2,28000,74.8,3.9,0.851
Argentina,76.7,17.7,22000,74.3,3.7,0.845
Oman,77.9,14.7,36000,80.2,15.4,0.813
Russia,72.6,15.5,27000,76.1,11.1,0.824
Belarus,74.8,15.6,18000,79.1,6.3,0.823
Kazakhstan,73.2,15.6,23000,78.9,13.0,0.825
Malaysia,76.2,13.5,27000,84.2,7.9,0.810
Costa Rica,80.3,15.4,18000,74.1,1.6,0.810
Panama,78.5,13.0,26000,58.5,2.7,0.815
Mauritius,75.0,15.2,22000,58.6,3.2,0.804
Serbia,76.0,14.7,17000,73.4,5.3,0.806
Albania,78.6,14.7,13000,71.8,1.7,0.795
Cuba,78.8,14.4,10000,46.2,2.6,0.783
Sri Lanka,77.0,14.1,12000,34.1,1.0,0.782
Bosnia,77.4,13.8,13000,70.3,6.2,0.780
Mexico,75.1,14.3,19000,70.1,3.7,0.779
Thailand,77.2,15.4,18000,66.7,3.9,0.777
China,76.9,14.0,16000,54.3,7.4,0.761
Brazil,75.9,15.4,14000,67.5,2.1,0.765
Colombia,77.3,14.4,14000,62.3,1.6,0.767
Armenia,75.1,13.1,11000,64.7,1.9,0.776
Algeria,76.9,14.6,11000,49.0,3.5,0.748
Tunisia,76.7,15.1,10000,55.5,2.5,0.740
Lebanon,78.9,11.3,15000,78.2,4.3,0.744
Ukraine,72.1,15.1,13000,62.6,4.1,0.779
Peru,76.7,15.0,12000,52.5,1.7,0.777
Ecuador,77.0,14.6,11000,57.3,1.9,0.759
Mongolia,69.9,14.2,12000,23.7,7.1,0.737
South Africa,65.0,13.8,12000,56.2,7.5,0.709
Egypt,72.0,13.1,11000,46.9,2.5,0.707
Gabon,66.5,13.0,16000,48.1,2.8,0.703
Maldives,78.9,12.2,14000,63.2,3.3,0.740
Bolivia,71.5,14.2,8000,43.8,1.8,0.718
Indonesia,71.7,13.6,11000,39.9,2.2,0.718
Vietnam,75.4,12.7,8000,54.2,2.3,0.704
Philippines,71.2,13.1,9000,60.1,1.2,0.718
Botswana,69.6,12.6,16000,47.0,2.9,0.735
Morocco,76.7,13.1,7000,64.8,1.7,0.686
Kyrgyzstan,71.5,13.0,5000,38.2,1.7,0.697
Guyana,69.9,12.1,11000,37.3,2.6,0.682
El Salvador,73.3,12.4,8000,33.8,1.0,0.673
Tajikistan,71.1,11.5,4000,22.0,0.9,0.668
Guatemala,74.3,10.8,8000,40.7,1.1,0.663
India,69.7,12.2,6800,20.1,1.8,0.645
Honduras,75.3,10.2,5000,31.7,1.1,0.634
Bangladesh,72.6,11.6,4800,15.0,0.5,0.632
Vanuatu,70.5,11.5,3000,26.0,0.5,0.609
Namibia,63.7,12.6,9000,31.0,1.6,0.646
Nicaragua,74.5,12.1,5000,28.0,0.8,0.660
Myanmar,67.1,10.3,5000,31.0,0.6,0.583
Ghana,64.1,11.5,5000,37.9,0.6,0.611
Nepal,70.8,12.2,3400,21.4,0.3,0.602
Kenya,66.7,11.3,4200,17.8,0.4,0.601
Cambodia,69.8,11.5,4000,32.9,0.7,0.594
Angola,61.2,11.8,6000,14.3,1.0,0.581
Zambia,63.9,11.5,3300,14.3,0.3,0.584
Cameroon,59.3,12.1,3600,23.2,0.4,0.563
Pakistan,67.3,8.3,5000,15.5,0.9,0.557
Zimbabwe,61.5,11.0,2700,25.1,0.7,0.571
Nigeria,54.7,10.0,5000,42.0,0.6,0.539
Uganda,63.3,10.3,2100,23.7,0.1,0.544
Rwanda,69.0,11.2,2200,21.8,0.1,0.543
Tanzania,65.5,8.1,2700,13.0,0.2,0.529
Ethiopia,66.6,8.8,2200,18.6,0.1,0.485
Malawi,64.2,11.2,1000,13.8,0.1,0.483
Mali,59.3,7.5,2100,13.0,0.1,0.434
Yemen,66.1,8.8,1500,26.7,0.4,0.470
Sudan,65.3,7.9,2400,30.9,0.5,0.510
Niger,62.4,6.5,1200,10.2,0.1,0.394
Chad,54.2,7.3,1600,6.5,0.1,0.394
Burundi,61.6,11.1,700,9.4,0.1,0.433
Central African Republic,53.3,7.6,900,4.3,0.1,0.397
"""

# #Importing the dataset
Development = pd.read_csv(io.StringIO(csv_data))
print("--- Dataset Initialized Successfully ---")
print(f"Total entries loaded: {Development.shape[0]} rows across {Development.shape[1]} metrics.\n")

# #Listing the first five rows of the dataset
print(Development.head())

# Save original CSV locally for project integrity checks
Development.to_csv("HDI.csv", index=False)

# 2. EXPLORATORY CHARTS GENERATION LAYERS (10 Visualizations)
print("\n--- Plotting Model Diagnostics to plots/ Directory ---")
os.makedirs('static/plots', exist_ok=True)
sns.set_theme(style="whitegrid")

# Subset target variable data1 to eliminate visualization noise (Top 20 rows per screen constraint)
data1 = Development.head(20)

# Vis 1: Mean Schooling vs HDI Score (Strip Plot from Snapshot Requirements)
plt.figure(figsize=(10, 5))
sns.stripplot(x="Expected_Years_Schooling", y="HDI", data=data1, jitter=True)
plt.xticks(rotation=90)
plt.title("1. Educational Access Metrics Variance Model")
plt.tight_layout()
plt.savefig('static/plots/1_schooling_strip.png')
plt.close()

# Vis 2: Distribution Profile of Target metric
plt.figure()
sns.histplot(Development['HDI'], kde=True, color='purple')
plt.title("2. HDI Global Structural Target Distribution Profile")
plt.savefig('static/plots/2_hdi_distribution.png')
plt.close()

# Vis 3: Correlation Strength Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(Development.select_dtypes(include=[np.number]).corr(), annot=True, cmap='mako')
plt.title("3. Feature Inter-Correlation Density Map Matrix")
plt.tight_layout()
plt.savefig('static/plots/3_correlation_matrix.png')
plt.close()

# Vis 4: Life Expectancy Tracking Scatter Map
plt.figure()
sns.scatterplot(data=Development, x='Life_Expectancy', y='HDI', color='emerald' if 'emerald' in sns.colors.SEABORN_PALETTES else 'green')
plt.title("4. Country Life Expectancy vs Index Scatter Matrix")
plt.savefig('static/plots/4_life_expectancy.png')
plt.close()

# Vis 5: Financial Wealth Transform Comparisons
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(ax=axes[0], data=Development, y='GNI_Per_Capita', color='teal')
sns.boxplot(ax=axes[1], data=Development, y='GNI_Per_Capita', color='blue').set(yscale="log")
plt.suptitle("5. GNI Per Capita Scale Invariance Structural Testing")
plt.savefig('static/plots/5_gni_transformations.png')
plt.close()

# Vis 6: Internet Infrastructure Population Penetration Curve
plt.figure()
sns.kdeplot(data=Development, x='Internet_Users_Pct', fill=True, color='orange')
plt.title("6. Digital Connectivity Probability Mass Function Curve")
plt.savefig('static/plots/6_internet_connectivity.png')
plt.close()

# Vis 7: Pairwise Dimension Relationships Matrix Array
p = sns.pairplot(Development[['HDI', 'Life_Expectancy', 'Expected_Years_Schooling']])
p.savefig('static/plots/7_pairwise_dimensions.png')
plt.close()

# Vis 8: Environmental Carbon Production Hex Plots
j = sns.jointplot(data=Development, x='Carbon_Emission_Per_Capita', y='HDI', kind='hex', color='darkblue')
j.savefig('static/plots/8_carbon_footprint.png')
plt.close()

# Vis 9: Grouped Regional Clusters Schooling Metrics 
Development['Development_Group'] = pd.qcut(Development['HDI'], q=3, labels=['Lower', 'Mid', 'Upper'])
plt.figure()
sns.violinplot(data=Development, x='Development_Group', y='Expected_Years_Schooling', palette='muted')
plt.title("9. Schooling Variances Map Across Tiers")
plt.savefig('static/plots/9_violin_distribution.png')
plt.close()

# Vis 10: Index Growth Path (ECDF Analysis)
plt.figure()
sns.ecdfplot(data=Development, x='HDI', color='red')
plt.title("10. Empirical Cumulative Trend Path Map (ECDF)")
plt.savefig('static/plots/10_ecdf_trend.png')
plt.close()

print("All 10 required project visualizations rendered successfully into 'static/plots/' folder.")

# 3. MATRIX TRAINING PIPELINE ARCHITECTURE (reg Model Configuration)
print("\n--- Segregating Matrix Blocks & Fitting Estimators ---")
features = ['Life_Expectancy', 'Expected_Years_Schooling', 'GNI_Per_Capita', 'Internet_Users_Pct', 'Carbon_Emission_Per_Capita']
X = Development[features]
y = Development['HDI']

# train and test split configuration mapping matching assignments parameters
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the Model
reg = LinearRegression()
reg.fit(x_train, y_train)

# Evaluation Metric Calculations
y_pred = reg.predict(x_test)
print(f"R2 Accuracy Matrix Determination:       {metrics.r2_score(y_test, y_pred):.4f}")
print(f"MAE Performance Calculation Metric:     {metrics.mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE Calculation Value Target:           {metrics.mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE Variance Deviation Index:          {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.4f}")

# #saving our model into a file via pickle implementation
pickle.dump(reg, open('HDI.pkl', 'wb'))
print("State files compiled successfully. Parameter model map exported as 'HDI.pkl'.")

# 4. ENVIRONMENT FILES REPOSITORY WRITER LAYER
print("\n--- Programmatically Drafting Local Web Application Component Shells ---")
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

