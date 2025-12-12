import pandas as pd
import numpy as np

# Import utilities
from src.utils.metrics import evaluate

# Import models
from src.models.KNN import KNNRegressor
from src.models.decision_tree import DecisionTreeRegressorScratch
from src.models.random_forest import RandomForestRegressorScratch
from src.models.gradient_boosting import GradientBoostingRegressorScratch

# Import model saver
from src.models.save_model import save_model


# ============================
# Load Encoded Dataset
# ============================
df = pd.read_csv("data/processed/cleaned_data.csv")
df = df.astype(float)

y = df["selling_price"].values
X = df.drop(columns=["selling_price"]).values


# ============================
# Train-Test Split
# ============================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Training Columns:", df.columns.tolist())
print("Number of Columns:", len(df.columns))


# ============================
# Train + Evaluate Helper
# ============================
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse, rmse, mae, r2 = evaluate(y_test, preds)

    print(f"\n➡ {name}")
    print("MSE        :", mse)
    print("RMSE       :", rmse)
    print("MAE        :", mae)
    print("R² Score   :", r2)
    print("R² Score % :", r2 * 100)

    return model, preds, mse, rmse, mae, r2


# ============================
# Train All Models
# ============================

# 1. KNN Regressor
model_knn = KNNRegressor(k=7)
model_knn, knn_preds, knn_mse, knn_rmse, knn_mae, knn_r2 = evaluate_model(
    "KNN Regression (k=7)", model_knn, X_train, y_train, X_test, y_test
)

# 2. Decision Tree
model_tree = DecisionTreeRegressorScratch(max_depth=7, min_samples_split=20)
model_tree, tree_preds, tree_mse, tree_rmse, tree_mae, tree_r2 = evaluate_model(
    "Decision Tree Regression", model_tree, X_train, y_train, X_test, y_test
)

# 3. Random Forest
model_rf = RandomForestRegressorScratch(
    n_trees=80,
    max_depth=10,
    min_samples_split=5,
    max_features=None,
    random_state=42
)

model_rf, rf_preds, rf_mse, rf_rmse, rf_mae, rf_r2 = evaluate_model(
    "Random Forest Regression", model_rf, X_train, y_train, X_test, y_test
)

# 4. Gradient Boosting
model_gb = GradientBoostingRegressorScratch(
    n_estimators=80,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=2
)

model_gb, gb_preds, gb_mse, gb_rmse, gb_mae, gb_r2 = evaluate_model(
    "Gradient Boosting Regression", model_gb, X_train, y_train, X_test, y_test
)


# ================================
# Save All Models
# ================================
save_model(model_knn, None, "knn")
save_model(model_tree, None, "decision_tree")
save_model(model_rf, None, "random_forest")
save_model(model_gb, None, "gradient_boosting")

print("\nAll models saved successfully!")


# ================================
# Save Evaluation Results
# ================================
results = {
    "Model": ["KNN", "Decision Tree", "Random Forest", "Gradient Boosting"],
    "MSE": [knn_mse, tree_mse, rf_mse, gb_mse],
    "RMSE": [knn_rmse, tree_rmse, rf_rmse, gb_rmse],
    "MAE": [knn_mae, tree_mae, rf_mae, gb_mae],
    "R2 Score": [knn_r2, tree_r2, rf_r2, gb_r2],
    "R2 Score %": [knn_r2 * 100, tree_r2 * 100, rf_r2 * 100, gb_r2 * 100]
}

df_results = pd.DataFrame(results)
df_results.to_csv("data/model_evaluation.csv", index=False)

print("\nSaved model_evaluation.csv")
