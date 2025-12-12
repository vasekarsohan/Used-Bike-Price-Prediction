import pickle
import os

def save_model(model, scaler, name):
    os.makedirs("models", exist_ok=True)

    model_data = {
        "model": model,
        "scaler": None    # No scaler needed now
    }

    path = f"models/{name}.pkl"

    with open(path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved at: {path}")
