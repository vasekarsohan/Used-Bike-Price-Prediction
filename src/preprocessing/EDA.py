# ===============================================
# 📌 IMPORT LIBRARIES
# ===============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# ===============================================
# 📌 LOAD DATA
# ===============================================
df = pd.read_csv("./data/raw/BIKEDETAILS.csv")
print("\nDataset Loaded Successfully!")
print("Columns:", df.columns.tolist())


# ===============================================
# 📌 1. BASIC DATA CHECKS
# ===============================================
print("\n----- HEAD -----")
print(df.head())

print("\n----- INFO -----")
df.info()

print("\n----- DESCRIBE -----")
print(df.describe())

print("\n----- UNIQUE VALUES -----")
print(df.nunique())

print("\n----- DUPLICATES -----")
print("Duplicate rows:", df.duplicated().sum())


# ===============================================
# 📌 2. OWNER MAPPING (Ordinal Encoding)
# ===============================================
owner_map = {'1st owner': 0, '2nd owner': 1, '3rd owner': 2, '4th owner': 3}
df['owner'] = df['owner'].map(owner_map)
print("\nOwner column mapped successfully!")


# ===============================================
# 📌 3. HANDLE MISSING VALUES
# ===============================================
print("\nMissing values before cleaning:\n", df.isnull().sum())

df['ex_showroom_price'] = df['ex_showroom_price'].fillna(df['ex_showroom_price'].median())

print("\nMissing values after cleaning:\n", df.isnull().sum())


# ===============================================
# 📌 4. DISTRIBUTIONS BEFORE OUTLIER CAPPING
# ===============================================
os.makedirs("plots/distribution/before", exist_ok=True)

numeric_cols = ['selling_price', 'km_driven', 'ex_showroom_price']

for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} Distribution (Before Outlier Capping)")
    plt.tight_layout()
    plt.savefig(f"plots/distribution/before/{col}_before.png")
    plt.close()


# ===============================================
# 📌 5. OUTLIER CAPPING USING IQR
# ===============================================
def cap_outliers(column):
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower, lower,
                   np.where(df[column] > upper, upper, df[column]))

df_before = df.copy()
for col in numeric_cols:
    cap_outliers(col)

print("\nOutlier capping applied successfully!")


# ===============================================
# 📌 6. OUTLIER BOXPLOTS
# ===============================================
os.makedirs("plots/outliers", exist_ok=True)

# Before Outliers
plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=df_before[col])
    plt.title(f"{col} (Before)")
plt.tight_layout()
plt.savefig("plots/outliers/before_outliers.png")
plt.close()

# After Outliers
plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f"{col} (After)")
plt.tight_layout()
plt.savefig("plots/outliers/after_outliers.png")
plt.close()


# ===============================================
# 📌 7. DISTRIBUTIONS AFTER OUTLIER CAPPING
# ===============================================
os.makedirs("plots/distribution/after", exist_ok=True)

for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} Distribution (After Outlier Capping)")
    plt.tight_layout()
    plt.savefig(f"plots/distribution/after/{col}_after.png")
    plt.close()


# ===============================================
# 📌 8. ENCODING (SELLER TYPE + NAME TARGET ENCODING)
# ===============================================
print("\nApplying Encoding...")

df['seller_type'] = df['seller_type'].map({'Individual': 0, 'Dealer': 1})

name_mean_price = df.groupby("name")["selling_price"].mean()
df["name"] = df["name"].map(name_mean_price)

df["name"] = df["name"].fillna(df["selling_price"].mean())

print("Encoding applied successfully!")


# ===============================================
# 📌 9. FULL CORRELATION HEATMAP
# ===============================================
os.makedirs("plots/heatmaps", exist_ok=True)

df_corr = df.copy()

# Factorize non-numeric for correlation
non_numeric = df_corr.select_dtypes(include=['object']).columns
for col in non_numeric:
    df_corr[col], _ = pd.factorize(df_corr[col])

corr_matrix = df_corr.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f"
)
plt.title("Full Correlation Heatmap (All Columns)")
plt.tight_layout()
plt.savefig("plots/heatmaps/full_correlation_heatmap.png")
plt.close()

print("\nFull correlation heatmap saved.")


# ===============================================
# 📌 10. INSIGHT-BASED PLOTS
# ===============================================
os.makedirs("plots/insights", exist_ok=True)

# Selling Price Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['selling_price'], kde=True)
plt.title("Selling Price Distribution")
plt.tight_layout()
plt.savefig("plots/insights/selling_price_distribution.png")
plt.close()

# Premium Brands (Top 20)
plt.figure(figsize=(12,6))
brand_prices = df.groupby("name")["selling_price"].mean().sort_values(ascending=False).head(20)
sns.barplot(x=brand_prices.values, y=brand_prices.index)
for i, v in enumerate(brand_prices.values):
    plt.text(v + 200, i, f"{int(v)}", va='center')
plt.title("Top 20 Models with Highest Resale Value")
plt.tight_layout()
plt.savefig("plots/insights/brand_vs_price.png")
plt.close()

# Year vs Avg Price
year_avg = df[(df["year"] >= 2000) & (df["year"] <= 2020)].groupby("year")["selling_price"].mean()
plt.figure(figsize=(10,6))
sns.lineplot(x=year_avg.index, y=year_avg.values, marker="o")
for x, y in zip(year_avg.index, year_avg.values):
    plt.text(x, y + 500, f"{int(y)}", ha='center')
plt.title("Average Selling Price by Manufacturing Year (2000–2020)")
plt.tight_layout()
plt.savefig("plots/insights/year_vs_price.png")
plt.close()

# KM Driven vs Price (Bins)
df["km_bin"] = pd.qcut(df["km_driven"], q=10, duplicates='drop')
km_avg = df.groupby("km_bin", observed=False)["selling_price"].mean()
plt.figure(figsize=(12,6))
km_avg.plot(marker="o")
for i, (idx, val) in enumerate(km_avg.items()):
    plt.text(i, val + 300, f"{int(val)}", ha='center')
plt.title("Average Selling Price vs KM Driven (Binned)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/insights/km_vs_price.png")
plt.close()

# Owner vs Price
owner_avg = df.groupby("owner")["selling_price"].mean()
plt.figure(figsize=(8,6))
sns.lineplot(x=owner_avg.index, y=owner_avg.values, marker="o")
for x, y in zip(owner_avg.index, owner_avg.values):
    plt.text(int(x), y + 300, f"{int(y)}", ha='center')
plt.title("Average Selling Price vs Number of Owners")
plt.xticks([0, 1, 2, 3])
plt.tight_layout()
plt.savefig("plots/insights/owner_vs_price.png")
plt.close()

# Ex-Showroom Price vs Resale Price
df["ex_bin"] = pd.qcut(df["ex_showroom_price"], q=10, duplicates='drop')
ex_avg = df.groupby("ex_bin", observed=False)["selling_price"].mean()
plt.figure(figsize=(12,6))
ex_avg.plot(kind='bar')
for i, val in enumerate(ex_avg.values):
    plt.text(i, val + 300, f"{int(val)}", ha='center')
plt.title("Avg Selling Price vs Ex-Showroom Price (Binned)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/insights/ex_price_vs_resale.png")
plt.close()


# ===============================================
# 📌 11. SAVE CLEANED DATA (ONLY REQUIRED COLUMNS)
# ===============================================
df = df.drop(columns=["km_bin", "ex_bin"], errors='ignore')

final_cols = ["name", "selling_price", "year", "seller_type",
              "owner", "km_driven", "ex_showroom_price"]

df = df[final_cols]

os.makedirs("./data/Processed", exist_ok=True)
df.to_csv("./data/Processed/cleaned_data.csv", index=False)

print("\nFinal cleaned dataset saved as: data/Processed/cleaned_data.csv")
print("Shape:", df.shape)