# Mall Customer Value Segmentation and Precision Marketing Strategy Analysis

## 📌 Project Overview
This project applies the K-Means unsupervised clustering algorithm to conduct full-dimensional value segmentation of mall member customers. It covers the complete workflow: **simulated data generation → feature engineering → clustering training → effect validation → business logic verification → CLV (Customer Lifetime Value) calculation → marketing strategy output**.

The project segments 500 sample customers into 5 value groups, accurately identifies core growth customer groups with high income but low consumption, and delivers corresponding differentiated precision operation strategies. It is expected to drive an overall revenue increase of over 15%.

## 📁 Directory Structure
```plaintext
mall-customer-segmentation/
├── README.md                    # Project documentation
├── requirements.txt             # Dependency package list
├── .gitignore                   # Ignore file configuration
├── data/                        # Data directory
│   ├── customer_data.csv        # Simulated customer data
│   └── data_dictionary.md       # Field description
├── notebooks/                   # Jupyter analysis notebook
│   └── customer_segmentation_full.ipynb
├── scripts/                     # Python script version
│   └── run_segmentation.py
├── output/                      # Output result samples
│   ├── segment_profile_summary.csv
│   ├── elbow_plot.png
│   └── clv_comparison.png
└── docs/                        # Supporting documents
    └── precision_marketing_strategy.md
```

## 📈 Analysis Process
1. **Data preprocessing**: data cleaning, missing value handling, feature standardization
2. **Clustering modeling**: determine optimal k value by elbow method, K-Means training clustering
3. **Effect verification**: silhouette score quantification, population distribution verification
4. **Business verification**: VIP double-high verification, age activity verification, quantile cross-validation
5. **Value calculation**: CLV customer lifetime value calculation and outlier detection
6. **Strategy output**: formulate differentiated precision marketing strategies for each segment

## 📝 Notes
- The repository only contains simulated demonstration data, no real customer information
- When replacing with real business data, you only need to modify the data path and feature columns to reuse
  
## 🛠️ Tech Stack
- Data processing: Pandas, NumPy
- Clustering algorithm: scikit-learn (K-Means)
- Data visualization: Matplotlib
- Value model: CLV (Customer Lifetime Value) model
- Validation system: Dual validation of Silhouette Score + business logic

## 📊 Key Outcomes
1. **Clustering performance**: Average Silhouette Score of 0.52 at k=5, indicating good segmentation distinguishability
2. **Customer group identification**: 20.3% of high-income low-consumption customers identified as the highest ROI growth target
3. **Value quantification**: The average CLV of VIP customers is 3.2 times that of mass customers, with significant value difference
4. **Strategy output**: Complete differentiated precision marketing solutions for 5 customer groups, covering channels, benefits, cycles and metrics

## 👥 Customer Segment Profile Overview
| Segment ID | Customer Group Label | Population Proportion | Core Characteristics | Core Operation Direction |
|------------|----------------------|-----------------------|----------------------|--------------------------|
| Cluster 0 | VIP High-Value Group | 20% | High income + high consumption + high frequency | Loyalty maintenance, exclusive benefits, value-added services |
| Cluster 1 | High-Income Low-Consumption Group | 20% | High income, untapped consumption potential | Precision activation, increased consumption frequency, scenario mining |
| Cluster 2 | Mass Consumption Mainforce | 30% | Medium income, medium consumption | Regular operation, AOV growth, cross-selling |
| Cluster 3 | Young New Customer Group | 16% | Low income, low consumption, young age | New customer retention, habit cultivation, lightweight conversion |
| Cluster 4 | Churn Risk Group | 14% | Extremely low consumption frequency, high historical AOV | Churn recall, exclusive follow-up, benefit reactivation |

## 🚀 Quick Start
### Environment Requirements
- Python 3.8 or above
- All dependency versions are specified in `requirements.txt`

### Execution Steps
1. Clone the repository to local
```bash
git clone https://github.com/your_username/mall-customer-segmentation.git
cd mall-customer-segmentation
