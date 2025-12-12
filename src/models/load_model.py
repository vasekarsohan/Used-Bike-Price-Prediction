import pickle

def load_model(path):
    """
    Load a trained ML model from a pickle file.
    Compatible with model format used in this project.
    
    Expected pickle structure:
    {
        "model": trained_model
    }
    """

    with open(path, "rb") as f:
        data = pickle.load(f)

    # Extract model
    model = data.get("model")

    if model is None:
        raise ValueError("ERROR: 'model' key not found in saved file.")

    print(f"Model loaded successfully from: {path}")
    return model
