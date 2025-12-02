"""
Requirements to run the program are; pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, kagglehub
"""
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression


np.random.seed(42)

# Download datasets
print("Downloading Kaggle datasets")
path_f = kagglehub.dataset_download("thedevastator/domestic-food-prices-after-covid-19")
path_a = kagglehub.dataset_download("mfalfafa/amazon-sales-during-covid19")
print("Path for food dataset", path_f)
print("Path for amazon dataset", path_a)

# Find CSV files for datasets
def find_csv(base_path):
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.endswith(".csv"):
                return os.path.join(root, f)
    return None

food_csv = find_csv(path_f)
amazon_csv = find_csv(path_a)
print("\nFound food CSV", food_csv)
print("Found Amazon CSV", amazon_csv)

# IMPORTANT: Amazon file is semicolon-separated
food_df = pd.read_csv(food_csv)
amazon_df = pd.read_csv(amazon_csv, sep=";")

print("\nFood dataset", food_df.shape)
print("Amazon dataset", amazon_df.shape)
print("\nFood dataset")
print(food_df.head(3))
print("\nAmazon dataset")
print(amazon_df.head(3))

# Clean columns
food_df.columns = [c.strip().lower().replace(" ", "_") for c in food_df.columns]
amazon_df.columns = [c.strip().lower().replace(" ", "_") for c in amazon_df.columns]

# --- Normalize dates/months and prices so merge/plots don't break ---

def ensure_month_column(df):
    # If there's already a 'month' column, try to parse it
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
    # If there is a generic 'date' column
    elif "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"], errors="coerce")
    # Amazon-specific: sometimes only 'date_first_available'
    elif "date_first_available" in df.columns:
        df["month"] = pd.to_datetime(df["date_first_available"], errors="coerce")
    else:
        return df

    # Normalize to month start (e.g., 2020-06-01)
    df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
    return df

# If amazon has a 'date_first_available' column, create a 'date' for consistency
if "date_first_available" in amazon_df.columns and "date" not in amazon_df.columns:
    amazon_df["date"] = pd.to_datetime(amazon_df["date_first_available"], errors="coerce")

# Apply month normalization
food_df = ensure_month_column(food_df)
amazon_df = ensure_month_column(amazon_df)

# ---- Price cleanup ----

# Amazon: use 'sale_price' as numeric 'price' if present
if "sale_price" in amazon_df.columns and "price" not in amazon_df.columns:
    amazon_df["price"] = (
        amazon_df["sale_price"]
        .astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
    )
    amazon_df["price"] = pd.to_numeric(amazon_df["price"], errors="coerce")

# Food: ensure price is numeric if present
if "price" in food_df.columns:
    food_df["price"] = pd.to_numeric(food_df["price"], errors="coerce")

# Drop rows that are missing the key columns we need,
# but ONLY if those columns actually exist.
if "month" in amazon_df.columns:
    if "price" in amazon_df.columns:
        amazon_df = amazon_df.dropna(subset=["month", "price"])
    else:
        amazon_df = amazon_df.dropna(subset=["month"])

if "month" in food_df.columns:
    if "price" in food_df.columns:
        food_df = food_df.dropna(subset=["month", "price"])
    else:
        food_df = food_df.dropna(subset=["month"])

# Filter for US
if "country" in amazon_df.columns:
    amazon_df = amazon_df[amazon_df["country"].str.contains("United", case=False, na=False)]
if "country" in food_df.columns:
    food_df = food_df[food_df["country"].str.contains("United", case=False, na=False)]

#  EDA 
print("\nStatistics for Amazon")
print(amazon_df.describe(include="all"))

print("\nStatistics for Food Prices")
print(food_df.describe(include="all"))

#  missing valued (fully empty rows, if any)
amazon_df = amazon_df.dropna(how="all")
food_df = food_df.dropna(how="all")

if "month" in amazon_df.columns and "month" in food_df.columns:
    merged = pd.merge(amazon_df, food_df, on="month", how="inner", suffixes=("_amazon", "_food"))
else:
    merged = amazon_df.copy()
print("\ndataset shape", merged.shape)
if "price" in amazon_df.columns:
    amazon_df["price_change_pct"] = amazon_df["price"].pct_change()
if "price" in food_df.columns:
    food_df["price_change_pct"] = food_df["price"].pct_change()

# trends 
plt.figure(figsize=(10,5))
if "month" in amazon_df.columns and "price" in amazon_df.columns:
    sns.lineplot(data=amazon_df, x="month", y="price")
    plt.title("Amazon Price Trends")
    plt.xticks(rotation=30)
    plt.show()
plt.figure(figsize=(10,5))
if "month" in food_df.columns and "price" in food_df.columns:
    sns.lineplot(data=food_df, x="month", y="price")
    plt.title("Food Price Trends")
    plt.xticks(rotation=30)
    plt.show()

#  Clustering 
if "month" in amazon_df.columns and "price" in amazon_df.columns:
    cluster_df = amazon_df.groupby("month").agg(avg_price=("price", "mean")).reset_index()
    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_df[["avg_price"]])

    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_df["cluster"] = kmeans.fit_predict(X)

    plt.figure(figsize=(10,5))
    sns.scatterplot(data=cluster_df, x="month", y="avg_price", hue="cluster", palette="tab10")
    plt.title("Amazon Average Prices")
    plt.xticks(rotation=30)
    plt.show()

# Regression 
if "month" in amazon_df.columns and "price" in amazon_df.columns:
    amazon_df = amazon_df.dropna(subset=["price"])
    amazon_df["month_num"] = amazon_df["month"].rank(method="dense").astype(int)

    X = amazon_df[["month_num"]]
    y = amazon_df["price"]

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    plt.figure(figsize=(10,5))
    plt.scatter(X, y, label="Actual", alpha=0.7)
    plt.plot(X, preds, color="red", label="Predicted Trend")
    plt.title("Price vs Time")
    plt.xlabel("Month #")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

print("\n Program Completed Successfuly.")
