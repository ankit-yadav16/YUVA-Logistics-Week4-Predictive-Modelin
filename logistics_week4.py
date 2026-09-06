import pandas as pd
import numpy as np

df = pd.read_csv("supply_chain_data.csv")

print(df.head())
print(df.shape)
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

target = "Shipping times"

features = [
    "Shipping costs",
    "Lead time",
    "Manufacturing lead time",
    "Order quantities",
    "Stock levels",
    "Production volumes",
    "Defect rates"
]

X = df[features]
y = df[target]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nLinear Regression Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=5
)

tree_model.fit(
    X_train,
    y_train
)

tree_predictions = tree_model.predict(X_test)

tree_mae = mean_absolute_error(
    y_test,
    tree_predictions
)

tree_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tree_predictions
    )
)

tree_r2 = r2_score(
    y_test,
    tree_predictions
)

print("\nDecision Tree Results")
print("MAE:", tree_mae)
print("RMSE:", tree_rmse)
print("R2:", tree_r2)

from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

forest_model.fit(
    X_train,
    y_train
)

forest_predictions = forest_model.predict(X_test)

forest_mae = mean_absolute_error(
    y_test,
    forest_predictions
)

forest_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        forest_predictions
    )
)

forest_r2 = r2_score(
    y_test,
    forest_predictions
)

print("\nRandom Forest Results")
print("MAE:", forest_mae)
print("RMSE:", forest_rmse)
print("R2:", forest_r2)

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "MAE": [
        mae,
        tree_mae,
        forest_mae
    ],

    "RMSE": [
        rmse,
        tree_rmse,
        forest_rmse
    ],

    "R2": [
        r2,
        tree_r2,
        forest_r2
    ]
})

print("\nModel Comparison:")
print(results)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["RMSE"]
)

plt.title("Comparison of Predictive Models")
plt.xlabel("Model")
plt.ylabel("RMSE")

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()

best_predictions = forest_predictions

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    best_predictions
)

plt.xlabel("Actual Shipping Time")
plt.ylabel("Predicted Shipping Time")

plt.title(
    "Actual vs Predicted Shipping Time"
)

plt.tight_layout()
plt.show()

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    forest_model,
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

print("\nCross Validation RMSE:")
print(-cv_scores)

print(
    "Average CV RMSE:",
    (-cv_scores).mean()
)

df["Predicted Shipping Time"] = forest_model.predict(X)

df["Risk Level"] = pd.cut(
    df["Predicted Shipping Time"],
    bins=[
        -float("inf"),
        3,
        6,
        float("inf")
    ],
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)

print(
    df[
        [
            "SKU",
            "Shipping times",
            "Predicted Shipping Time",
            "Risk Level"
        ]
    ].head(20)
)

