import numpy as np

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    """Coefficient of Determination"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate(y_true, y_pred):
    """
    Returns all metrics:
    - MSE
    - RMSE
    - MAE
    - R² Score
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse_val = rmse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)

    return mse, rmse_val, mae_val, r2_val
