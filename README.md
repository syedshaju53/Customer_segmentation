
# Customer Segmentation Using K-Means Clustering

An unsupervised machine learning project that analyzes customer behavior and groups customers into meaningful segments using the K-Means clustering algorithm.

## 📌 Overview

Customer segmentation helps businesses understand different types of customers based on their characteristics and purchasing behavior.

In this project, customer data is analyzed using Exploratory Data Analysis (EDA), feature scaling, the Elbow Method, and K-Means clustering.

The final model divides customers into four meaningful segments:

- Low Value Customers
- Premium Customers
- Regular Customers
- Potential Customers

The results can be used to support targeted marketing, customer retention, personalization, and business decision-making.

---

## 🎯 Objectives

- Analyze customer data and identify important patterns.
- Perform Exploratory Data Analysis.
- Analyze customer income and spending behavior.
- Select relevant features for clustering.
- Standardize numerical features.
- Determine the optimal number of clusters using the Elbow Method.
- Apply K-Means clustering.
- Profile the resulting customer segments.
- Visualize customer groups.
- Generate a final customer segmentation dataset.
- Extract business insights from the identified segments.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Project development |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| StandardScaler | Feature scaling |
| K-Means | Customer clustering |

---

## 🔄 Project Workflow

```text
Customer Dataset
       │
       ▼
Data Loading
       │
       ▼
Data Exploration
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature Selection
       │
       ▼
Feature Scaling
       │
       ▼
Elbow Method
       │
       ▼
K-Means Clustering
       │
       ▼
Customer Segmentation
       │
       ▼
Segment Profiling
       │
       ▼
Visualization
       │
       ▼
Business Insights


##  Dataset

The dataset contains customer information related to demographics and purchasing behavior.

The analysis includes variables such as:

Customer Age
Annual Income
Spending Score
Purchase Frequency
Recency
Total Spend


🔍 Exploratory Data Analysis

The project performs EDA to understand customer behavior and relationships between different variables.

Analysis Performed
Customer age distribution
Annual income distribution
Spending score distribution
Purchase frequency analysis
Customer recency analysis
Total spending analysis
Annual income vs. spending score
Feature correlation analysis

👥 Customer Segments

The clusters are interpreted into business-friendly customer categories.

Segment	Description
Low Value Customers	Customers with relatively lower spending and overall customer value
Premium Customers	High-value customers with strong spending behavior
Regular Customers	Customers showing consistent purchasing behavior
Potential Customers	Customers with potential for increased engagement and spending

These segments provide a simplified view of customer behavior that can be used for business analysis.

📈 Visualizations

The project generates multiple visualizations to understand customer characteristics and clustering results.

Visualizations Include
Age distribution
Annual income distribution
Spending score analysis
Income vs. spending
Purchase frequency
Customer recency
Total spending
Correlation heatmap
Elbow Method
Customer clusters
Customer segment analysis

All generated visualizations are stored in the Outputs directory.


## Project Structure 

Customer_Segmentation/
│
├── Data.csv
│
├── Customer_segmentation.py
│
├── Outputs/
│   ├── Customer_Segmentation_Result.csv
│   ├── EDA visualizations
│   ├── Elbow Method visualization
│   └── Cluster visualizations
│
├── requirements.txt
├── .gitignore
└── README.md



Author
Syed Shajahan

B.Tech – Data Science

License
This project is developed for educational and research purposes.

