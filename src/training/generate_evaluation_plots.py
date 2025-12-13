import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Global plot styling (IMPORTANT)
# ============================
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# ============================
# Load cleaned dataset
# ============================
df = pd.read_csv("data/processed/cleaned_data.csv")
df = df.astype(float)

y = df["selling_price"].values
X = df.drop(columns=["selling_price"]).values

# ============================
# SAME train-test split as training.py
# (order-based, deterministic)
# ============================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ============================
# Load trained models
# ============================
MODELS_DIR = "models"

model_files = {
    "KNN": "knn.pkl",
    "DecisionTree": "decision_tree.pkl",
    "RandomForest": "random_forest.pkl",
    "GradientBoosting": "gradient_boosting.pkl"
}

loaded_models = {}

for name, filename in model_files.items():
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "rb") as f:
        saved_obj = pickle.load(f)

    # Extract actual model from saved dictionary
    if isinstance(saved_obj, dict) and "model" in saved_obj:
        loaded_models[name] = saved_obj["model"]
    else:
        loaded_models[name] = saved_obj

# ============================
# Sanity check
# ============================
for name, model in loaded_models.items():
    print(f"{name} loaded as -> {type(model)}")

# ============================
# Create output directory
# ============================
OUTPUT_DIR = "plots/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# A. Actual vs Predicted
# ============================
for name, model in loaded_models.items():
    y_pred = model.predict(X_test)

    plt.figure(figsize=(7, 6))
    plt.scatter(
        y_test,
        y_pred,
        alpha=0.6,
        edgecolors="black",
        label="Predictions"
    )

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=2,
        label="Ideal Fit"
    )

    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title(f"Actual vs Predicted Selling Price ({name})")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}_actual_vs_pred.png")
    plt.close()

# ============================
# B. Residuals vs Predicted
# ============================
for name, model in loaded_models.items():
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(7, 6))
    plt.scatter(
        y_pred,
        residuals,
        alpha=0.6,
        edgecolors="black"
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=2,
        label="Zero Error Line"
    )

    plt.xlabel("Predicted Selling Price")
    plt.ylabel("Residual (Actual − Predicted)")
    plt.title(f"Residuals vs Predicted Price ({name})")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}_residuals_vs_pred.png")
    plt.close()

print("\n✅ All evaluation plots generated successfully.")
