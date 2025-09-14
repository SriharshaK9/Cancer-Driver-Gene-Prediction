import pandas as pd
import numpy as np
from feature_extraction import extract_features_from_sequence
from models import get_model
import pickle
import os

MAX_LEN = 50  # DNA-AAPIV feature length

def load_training_data():
    if not os.path.exists("train_data.csv"):
        raise FileNotFoundError("train_data.csv not found. Please create it with 'sequence' and 'label' columns.")
    
    df = pd.read_csv("train_data.csv")
    X = np.array([extract_features_from_sequence(seq, max_len=MAX_LEN) for seq in df["sequence"]])
    y = df["label"].values
    return X, y

def train_and_save_models():
    X, y = load_training_data()
    
    for model_type in ['SVM', 'NN', 'RF']:
        model = get_model(model_type)
        print(f"Training {model_type}...")
        model.fit(X, y)
        with open(f"{model_type}_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"{model_type} trained and saved ✅")

if __name__ == "__main__":
    train_and_save_models()
