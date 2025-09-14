import pandas as pd
import numpy as np
from feature_extraction import extract_features_from_sequence
from models import get_model
import pickle
import os

MAX_LEN = 50

def load_test_data():
    if not os.path.exists("test_data.csv"):
        raise FileNotFoundError("test_data.csv not found. Create it with 'sequence' column.")
    
    df = pd.read_csv("test_data.csv")
    X = np.array([extract_features_from_sequence(seq, max_len=MAX_LEN) for seq in df["sequence"]])
    return X, df["sequence"].values

def load_model(model_type):
    path = f"{model_type}_model.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Train models first.")
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_test_data():
    X_test, sequences = load_test_data()
    for model_type in ['SVM', 'NN', 'RF']:
        model = load_model(model_type)
        preds = model.predict(X_test)
        print(f"\n{model_type} Predictions:")
        for i, (seq, pred) in enumerate(zip(sequences, preds), start=1):
            label = "Driver Gene (1)" if pred == 1 else "Non-driver Gene (0)"
            print(f"{i}. {seq} → {label}")

if __name__ == "__main__":
    predict_test_data()
