from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def get_model(model_type):
    if model_type == "SVM":
        return SVC(kernel='linear', probability=True, random_state=42)
    elif model_type == "NN":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    elif model_type == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
