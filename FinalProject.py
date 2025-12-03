"""
Requirements to run the program are; pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, kagglehub
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

np.random.seed(42)

print("Downloading Kaggle datasets")
path_f = kagglehub.dataset_download("thedevastator/domestic-food-prices-after-covid-19")
path_a = kagglehub.dataset_download("mfalfafa/amazon-sales-during-covid19")
print("Path for food dataset", path_f)
print("Path for amazon dataset", path_a)


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

food_df = pd.read_csv(food_csv, encoding="latin1")
amazon_df = pd.read_csv(amazon_csv, sep=";")

print("\nFood dataset", food_df.shape)
print("Amazon dataset", amazon_df.shape)
print("\nFood dataset")
print(food_df.head(3))
print("\nAmazon dataset")
print(amazon_df.head(3))

food_df.columns = [c.strip().lower().replace(" ", "_") for c in food_df.columns]
amazon_df.columns = [c.strip().lower().replace(" ", "_") for c in amazon_df.columns]


def ensure_month_column(df):
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
    elif "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"], errors="coerce")
    elif "date_first_available" in df.columns:
        df["month"] = pd.to_datetime(df["date_first_available"], errors="coerce")
    else:
        return df

    df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
    return df


if "date_first_available" in amazon_df.columns and "date" not in amazon_df.columns:
    amazon_df["date"] = pd.to_datetime(amazon_df["date_first_available"], errors="coerce")

food_df = ensure_month_column(food_df)
amazon_df = ensure_month_column(amazon_df)

if "sale_price" in amazon_df.columns and "price" not in amazon_df.columns:
    amazon_df["price"] = (
        amazon_df["sale_price"]
        .astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
    )
    amazon_df["price"] = pd.to_numeric(amazon_df["price"], errors="coerce")

if "price" in food_df.columns:
    food_df["price"] = pd.to_numeric(food_df["price"], errors="coerce")

if "percent" in food_df.columns:
    food_df = food_df.drop(columns=["percent"])

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

print("\nStatistics for Amazon")
print(amazon_df.describe(include="all"))

print("\nStatistics for Food Prices")
print(food_df.describe(include="all"))

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

plt.figure(figsize=(10, 5))
if "month" in amazon_df.columns and "price" in amazon_df.columns:
    sns.lineplot(data=amazon_df, x="month", y="price")
    plt.title("Amazon Price Trends")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(10, 5))
if "month" in food_df.columns and "price" in food_df.columns:
    sns.lineplot(data=food_df, x="month", y="price")
    plt.title("Food Price Trends")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

if "month" in amazon_df.columns and "price" in amazon_df.columns:
    cluster_df = amazon_df.groupby("month").agg(avg_price=("price", "mean")).reset_index()
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(cluster_df[["avg_price"]])

    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_df["cluster"] = kmeans.fit_predict(X_cluster)

    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=cluster_df, x="month", y="avg_price", hue="cluster", palette="tab10")
    plt.title("Amazon Average Prices by Cluster")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

if "month" in amazon_df.columns and "price" in amazon_df.columns:
    amazon_df = amazon_df.dropna(subset=["price"])
    amazon_df["month_num"] = amazon_df["month"].rank(method="dense").astype(int)

    X_reg = amazon_df[["month_num"]]
    y_reg = amazon_df["price"]

    lin_model = LinearRegression()
    lin_model.fit(X_reg, y_reg)

    preds = lin_model.predict(X_reg)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_reg, y_reg, label="Actual", alpha=0.7)
    plt.plot(X_reg, preds, color="red", label="Predicted Trend")
    plt.title("Amazon Price vs Time (Linear Regression)")
    plt.xlabel("Month #")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

if "number_of_reviews" in amazon_df.columns:
    amazon_df["number_of_reviews"] = pd.to_numeric(amazon_df["number_of_reviews"], errors="coerce")
else:
    amazon_df["number_of_reviews"] = 0.0

feature_cols = ["month_num", "number_of_reviews"]
X_rf = amazon_df[feature_cols].fillna(0.0)
y_rf = amazon_df["price"]

rf = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10]
}
tscv = TimeSeriesSplit(n_splits=4)
grid = GridSearchCV(
    rf,
    param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1
)
grid.fit(X_rf, y_rf)

best_rf = grid.best_estimator_
print("\nBest RandomForest parameters:", grid.best_params_)
print("Best RandomForest R^2 score (TimeSeriesSplit):", grid.best_score_)

importances = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances (RandomForest):")
print(importances)

plt.figure(figsize=(6, 4))
importances.plot(kind="bar")
plt.title("Feature Importance (Random Forest Regression)")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

def label_phase(dt):
    if pd.isna(dt):
        return "Unknown"
    if dt < pd.Timestamp(2020, 2, 1):
        return "Pre"
    elif dt <= pd.Timestamp(2020, 3, 31):
        return "During"
    else:
        return "Post"

if "month" in amazon_df.columns:
    amazon_df["phase"] = amazon_df["month"].apply(label_phase)
    amazon_df = amazon_df[amazon_df["phase"] != "Unknown"]

    X_cls = X_rf.copy()
    le = LabelEncoder()
    y_cls = le.fit_transform(amazon_df["phase"])

    if len(np.unique(y_cls)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls
        )

        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)
        y_pred_lr = log_reg.predict(X_test)
        print("\nLogistic Regression classification report:")
        print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

        tree = DecisionTreeClassifier(random_state=42)
        tree.fit(X_train, y_train)
        y_pred_tree = tree.predict(X_test)
        print("Decision Tree classification report:")
        print(classification_report(y_test, y_pred_tree, target_names=le.classes_))


print("\n Program Completed Successfuly.")
