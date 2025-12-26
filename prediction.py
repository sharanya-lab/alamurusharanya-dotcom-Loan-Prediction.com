import pickle
import numpy as np

def load_model(model_path="loan_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_scaler(scaler_path="scaler.pkl"):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler

def predict(model, scaler, sample):
    """
    sample: list or array of feature values in same order as trained
    """
    sample_scaled = scaler.transform([sample])
    return model.predict(sample_scaled)
