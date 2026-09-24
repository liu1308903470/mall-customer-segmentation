# ==================================================
# Full Workflow Customer Segmentation Analysis Script (One-Click Version)
# Fix: String concatenation error caused by qcut returning Categorical type
# Output: All results are automatically saved to output/ directory
# ==================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ========== Initialization: Create Directories ==========
os.makedirs('../data', exist_ok=True)
os.makedirs('../output', exist_ok=True)

# Chinese font display settings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================================================
# Part 1: Generate Simulated Customer Data
# ==================================================
np.random.seed(42)

# Construct 5 typical customer groups
cluster1 = pd.DataFrame({
    'CustomerID': [f'C{str(i).zfill(4)}' for i in range(1, 101)],
    'Annual_Income': np.random.normal(80000, 12000, 100),
    'Total_Spend': np.random.normal(35000, 5000, 100),
    'Purchase_Frequency': np.random.normal(24, 4, 100),
    'Average_Order_Value': np.random.normal(1450, 200, 100),
    'Age': np.random.normal(38, 6, 100)
})
cluster2 = pd.DataFrame({
    'CustomerID': [f'C{str(i).zfill(4)}' for i in range(101, 201)],
    'Annual_Income': np.random.normal(75000, 10000, 100),
    'Total_Spend': np.random.normal(12000, 2500, 100),
    'Purchase_Frequency': np.random.normal(8, 2, 100),
    'Average_Order_Value': np.random.normal(1500, 250, 100),
    'Age': np.random.normal(42, 7, 100)
})
cluster3 = pd.DataFrame({
    'CustomerID': [f'C{str(i).zfill(4)}' for i in range(201, 351)],
    'Annual_Income': np.random.normal(45000, 8000, 150),
    'Total_Spend': np.random.normal(18000, 3000, 150),
    'Purchase_Frequency': np.random.normal(15, 3, 150),
    'Average_Order_Value': np.random.normal(1200, 180, 150),
    'Age': np.random.normal(32, 5, 150)
})
cluster4 = pd.DataFrame({
    'CustomerID': [f'C{str(i).zfill(4)}' for i in range(351, 431)],
    'Annual_Income': np.random.normal(30000, 5000, 80),
    'Total_Spend': np.random.normal(6000, 1500, 80),
    'Purchase_Frequency': np.random.normal(6, 2, 80),
    'Average_Order_Value': np.random.normal(1000, 150, 80),
    'Age': np.random.normal(25, 3, 80)
})
cluster5 = pd.DataFrame({
    'CustomerID': [f'C{str(i).zfill(4)}' for i in range(431, 501)],
    'Annual_Income': np.random.normal(55000, 9000, 70),
    'Total_Spend': np.random.normal(22000, 4000, 70),
    'Purchase_Frequency': np.random.normal(5, 1.5, 70),
    'Average_Order_Value': np.random.normal(4400, 600, 70),
    'Age': np.random.normal(45, 8, 70)
})

# Merge, shuffle, and round to integers
df = pd.concat([cluster1, cluster2, cluster3, cluster4, cluster5], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
for col in ['Annual_Income', 'Total_Spend', 'Purchase_Frequency', 'Average_Order_Value', 'Age']:
    df[col] = df[col].round(0).astype(int)

# Save raw data
df.to_csv('../data/customer_data.csv', index=False, encoding='utf-8-sig')
print("✅ Simulated data generated: data/customer_data.csv")

# ==================================================
# Part 2: Clustering Model Training and Effect Validation
# ==================================================
print("\n" + "="*50)
print("📊 Part 2: Clustering Model Training and Effect Validation")
print("="*50)

# Select numerical features
feature_cols = ['Annual_Income', 'Total_Spend', 'Purchase_Frequency', 'Average_Order_Value']
features = df[feature_cols].dropna()

# Data standardization
scaler = StandardScaler()
X = scaler.fit_transform(features)

# ----- 2.1 Determine Optimal k Value via Elbow Method -----
sse = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Plot and save elbow chart
plt.figure(figsize=(8, 4))
plt.plot(k_range, sse, 'o-', color='#2c7be5', linewidth=2)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
plt.title('Elbow Method for Optimal k Value', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../output/elbow_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Elbow method chart saved: output/elbow_plot.png")

# ----- 2.2 Silhouette Score Validation at k=5 -----
kmeans_final = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans_final.fit_predict(X)
avg_score = silhouette_score(X, labels)
print(f"✅ Average Silhouette Score at k=5: {avg_score:.3f} (0.4~0.6 indicates good segmentation)")

# ----- 2.3 Segment Population Distribution -----
df = df.dropna(subset=feature_cols).copy()
df['Cluster'] = labels
cluster_dist = df['Cluster'].value_counts(normalize=True).sort_index().round(4) * 100
print("\n📌 Population proportion of each cluster:")
for c, p in cluster_dist.items():
    print(f"   Cluster {c}: {p:.1f}%")

# ==================================================
# Part 3: Business Logic Verification
# ==================================================
print("\n" + "="*50)
print("🧠 Part 3: Business Logic Verification")
print("="*50)

# ----- 3.1 Generate Segment Profile Table -----
cluster_profile = df.groupby('Cluster').agg({
    'Annual_Income': ['mean', 'median'],
    'Total_Spend': ['mean', 'median'],
    'Purchase_Frequency': ['mean', 'median'],
    'Average_Order_Value': ['mean', 'median'],
    'Age': ['mean', 'median'],
    'CustomerID': 'count'
}).round(0)
cluster_profile.columns = [
    'Avg Annual Income', 'Median Annual Income',
    'Avg Total Spend', 'Median Total Spend',
    'Avg Purchase Frequency', 'Median Purchase Frequency',
    'Avg Order Value', 'Median Order Value',
    'Avg Age', 'Median Age',
    'Population'
]
cluster_profile['Population Proportion'] = (cluster_profile['Population'] / len(df) * 100).round(1).astype(str) + '%'

# Save profile table
cluster_profile.to_csv('../output/segment_profile_summary.csv', encoding='utf-8-sig')
print("✅ Segment profile table saved: output/segment_profile_summary.csv")

# ----- 3.2 VIP Group Dual-High Verification -----
income_top20 = df['Annual_Income'].quantile(0.8)
spend_top20 = df['Total_Spend'].quantile(0.8)
vip_cluster = cluster_profile['Avg Total Spend'].idxmax()
vip_income = cluster_profile.loc[vip_cluster, 'Avg Annual Income']
vip_spend = cluster_profile.loc[vip_cluster, 'Avg Total Spend']

print("\n🔍 VIP Group Dual-High Verification:")
print(f"   Top 20% income threshold: {income_top20:.0f}")
print(f"   Top 20% spend threshold: {spend_top20:.0f}")
print(f"   VIP cluster (Cluster {vip_cluster}) income meets standard: {vip_income >= income_top20}")
print(f"   VIP cluster (Cluster {vip_cluster}) spend meets standard: {vip_spend >= spend_top20}")

# ----- 3.3 Young Customer Group Activity Verification -----
youngest_cluster = cluster_profile['Avg Age'].idxmin()
oldest_cluster = cluster_profile['Avg Age'].idxmax()
young_freq = cluster_profile.loc[youngest_cluster, 'Avg Purchase Frequency']
old_freq = cluster_profile.loc[oldest_cluster, 'Avg Purchase Frequency']

print("\n🔍 Young Customer Group Activity Verification:")
print(f"   Youngest cluster (Cluster {youngest_cluster}): {cluster_profile.loc[youngest_cluster, 'Avg Age']:.0f} years old, {young_freq:.1f} times/year")
print(f"   Oldest cluster (Cluster {oldest_cluster}): {cluster_profile.loc[oldest_cluster, 'Avg Age']:.0f} years old, {old_freq:.1f} times/year")
print(f"   Frequency ratio: {young_freq/old_freq:.2f} (≥1.5 is consistent with business common sense)")

# ----- 3.4 High-Income Low-Consumption Group Scale Verification -----
income_median = df['Annual_Income'].median()
spend_median = df['Total_Spend'].median()
potential_clusters = cluster_profile[
    (cluster_profile['Avg Annual Income'] >= income_median) & 
    (cluster_profile['Avg Total Spend'] < spend_median)
]
total_pct = potential_clusters['Population'].sum() / len(df) * 100

print("\n🔍 High-Income Low-Consumption Group Scale Verification:")
print(f"   Target clusters: {list(potential_clusters.index)}")
print(f"   Total proportion: {total_pct:.1f}% (15%~30% is reasonable range)")

# ----- 3.5 Quantile Cross-Validation (Type Error Fixed) -----
# Key fix: Add .astype(str) after qcut to convert to string, avoid Categorical concatenation error
df['Income_Tier'] = pd.qcut(
    df['Annual_Income'], 
    q=[0, 0.2, 0.8, 1], 
    labels=['Low Income', 'Medium Income', 'High Income']
).astype(str)

df['Spend_Tier'] = pd.qcut(
    df['Total_Spend'], 
    q=[0, 0.2, 0.8, 1], 
    labels=['Low Spend', 'Medium Spend', 'High Spend']
).astype(str)

df['Manual_Segment'] = df['Income_Tier'] + '-' + df['Spend_Tier']
cross_tab = pd.crosstab(df['Cluster'], df['Manual_Segment'])
max_per_cluster = cross_tab.max(axis=1)
overall_match = max_per_cluster.sum() / len(df) * 100

print(f"\n🔍 Clustering vs Manual Segmentation Overlap: {overall_match:.1f}% (≥70% is credible)")

# ==================================================
# Part 4: CLV Calculation and Result Verification
# ==================================================
print("\n" + "="*50)
print("💰 Part 4: CLV Calculation and Result Verification")
print("="*50)

# Basic CLV: Average Order Value × Annual Frequency × 3-year lifecycle
df['CLV'] = df['Average_Order_Value'] * df['Purchase_Frequency'] * 3
cluster_clv = df.groupby('Cluster')['CLV'].agg(['mean', 'median']).round(0)
cluster_clv.columns = ['Average CLV', 'Median CLV']

# Plot and save CLV comparison chart
plt.figure(figsize=(8, 4))
cluster_clv['Average CLV'].plot(kind='bar', color='#2c7be5', width=0.6)
plt.xlabel('Segment Cluster', fontsize=12)
plt.ylabel('Average Customer Lifetime Value (Yuan)', fontsize=12)
plt.title('Average CLV Comparison by Segment', fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../output/clv_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ CLV comparison chart saved: output/clv_comparison.png")

# Sampling verification samples
sample = df.sample(5, random_state=42)[['CustomerID', 'Average_Order_Value', 'Purchase_Frequency', 'CLV']]
print("\n🧪 Sampling verification samples (can be copied to Excel for manual check):")
print(sample)

# Outlier detection
clv_mean = df['CLV'].mean()
negative_count = len(df[df['CLV'] <= 0])
null_count = df['CLV'].isnull().sum()
extreme_count = len(df[df['CLV'] > clv_mean * 5])

print("\n⚠️ Outlier Detection:")
print(f"   Overall CLV mean: {clv_mean:.0f}")
print(f"   Negative records: {negative_count}")
print(f"   Null records: {null_count}")
print(f"   Extreme values (>5x mean): {extreme_count}, proportion: {extreme_count/len(df)*100:.1f}%")

# ==================================================
# Final Result Export
# ==================================================
df.to_csv('../output/customer_segmentation_with_CLV.csv', index=False, encoding='utf-8-sig')

print("\n" + "="*50)
print("🎉 Full workflow completed! All files exported:")
print("  1. data/customer_data.csv —— Raw simulated customer data")
print("  2. output/customer_segmentation_with_CLV.csv —— Detailed data with cluster labels and CLV")
print("  3. output/segment_profile_summary.csv —— Summary of core metrics for each customer group")
print("  4. output/elbow_plot.png —— Elbow method optimal k value chart")
print("  5. output/clv_comparison.png —— CLV comparison chart by segment")
print("="*50)
