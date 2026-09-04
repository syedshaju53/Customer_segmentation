# ==========================================================
# CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
# Skills: Pandas, Matplotlib, Seaborn, K-Means, Elbow Method
# ==========================================================

# -----------------------------
# Import Libraries
# -----------------------------
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Data.csv"
OUTPUT_DIR = BASE_DIR / "Outputs"

# Create Outputs folder if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading Dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found:\n{DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print("\nDataset Loaded Successfully!\n")

# ==========================================================
# BASIC DATA INFORMATION
# ==========================================================

print("========== FIRST FIVE RECORDS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATASET INFO ==========")
df.info()

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# ==========================================================
# TASK 1 : EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

# Age Distribution
plt.figure(figsize=(7,5))
sns.histplot(df["Age"], kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig(OUTPUT_DIR / "01_Age_Distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Annual Income Distribution
plt.figure(figsize=(7,5))
sns.histplot(df["Annual_Income"], kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Count")
plt.savefig(OUTPUT_DIR / "02_Annual_Income_Distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Spending Score Distribution
plt.figure(figsize=(7,5))
sns.histplot(df["Spending_Score"], kde=True)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Count")
plt.savefig(OUTPUT_DIR / "03_Spending_Score_Distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Income vs Spending Score
plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Spending_Score",
    s=80
)
plt.title("Income vs Spending Score")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.savefig(OUTPUT_DIR / "04_Income_vs_Spending.png", dpi=300, bbox_inches="tight")
plt.show()

# Purchase Frequency
plt.figure(figsize=(7,5))
sns.histplot(df["Purchase_Frequency"], kde=True)
plt.title("Purchase Frequency Distribution")
plt.xlabel("Purchase Frequency")
plt.ylabel("Count")
plt.savefig(OUTPUT_DIR / "05_Purchase_Frequency.png", dpi=300, bbox_inches="tight")
plt.show()

# Customer Recency
plt.figure(figsize=(7,5))
sns.histplot(df["Recency_Days"], kde=True)
plt.title("Customer Recency Distribution")
plt.xlabel("Recency Days")
plt.ylabel("Count")
plt.savefig(OUTPUT_DIR / "06_Recency_Distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.savefig(OUTPUT_DIR / "07_Correlation_Heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# TASK 2 : APPLY K-MEANS CLUSTERING
# ==========================================================

print("\nApplying K-Means Clustering...")

# Feature Selection
X = df[["Annual_Income", "Spending_Score"]]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
wcss = []

for i in range(1, 11):
    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(7,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.savefig(OUTPUT_DIR / "08_Elbow_Method.png", dpi=300, bbox_inches="tight")
plt.show()

# Final Model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# ==========================================================
# TASK 3 : VISUALIZE CLUSTERS
# ==========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Spending_Score",
    hue="Cluster",
    palette="Set1",
    s=100
)

# Cluster Centroids
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:,0],
    centers[:,1],
    c="black",
    marker="X",
    s=250,
    label="Centroids"
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()

plt.savefig(OUTPUT_DIR / "09_Customer_Segmentation.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# TASK 4 : INTERPRET RESULTS
# ==========================================================

print("\n========== CLUSTER SUMMARY ==========")
cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
print(cluster_summary)

print("\n========== CUSTOMERS IN EACH CLUSTER ==========")
print(df["Cluster"].value_counts().sort_index())

# Segment Labels
segment_names = {
    0: "Low Value Customers",
    1: "Premium Customers",
    2: "Regular Customers",
    3: "Potential Customers"
}

df["Customer_Segment"] = df["Cluster"].map(segment_names)

print("\n========== CUSTOMER PROFILE ==========")

customer_profile = df.groupby("Customer_Segment").agg({
    "Annual_Income":"mean",
    "Spending_Score":"mean",
    "Purchase_Frequency":"mean",
    "Total_Spend":"mean"
})

print(customer_profile)

# Save Final Dataset
output_csv = OUTPUT_DIR / "Customer_Segmentation_Result.csv"
df.to_csv(output_csv, index=False)

print("\n===================================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("===================================================")
print("Output Folder :", OUTPUT_DIR)
print("Result CSV    :", output_csv)
print("All graphs have been saved as PNG files.")
print("===================================================")