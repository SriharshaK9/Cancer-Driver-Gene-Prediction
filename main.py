import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_extraction import extract_features_from_sequence
from models import get_model
import pickle
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MAX_LEN = 50

# -------------------- Model Loading -------------------- #
def load_model(model_type):
    path = f"{model_type}_model.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Train models first.")
    with open(path, "rb") as f:
        return pickle.load(f)

# -------------------- Training -------------------- #
def train_models_gui():
    try:
        df = pd.read_csv("train_data.csv")
        X_train = np.array([extract_features_from_sequence(seq) for seq in df["sequence"]])
        y_train = df["label"].values

        progress_win = tk.Toplevel(root)
        progress_win.title("Training Progress")
        progress_win.geometry("450x350")
        progress_win.configure(bg="#ffffff")

        tk.Label(progress_win, text="📊 Model Training Progress",
                 font=("Helvetica", 14, "bold"), bg="#ffffff", fg="#007acc").pack(pady=15)

        text_area = tk.Text(progress_win, font=("Courier", 11),
                            bg="#f8f9fa", fg="#333", wrap="word",
                            relief="solid", bd=1, height=12)
        text_area.pack(expand=True, fill="both", padx=10, pady=10)

        def log(msg):
            text_area.insert("end", msg + "\n")
            text_area.see("end")
            progress_win.update()

        for model_type in ['SVM', 'NN', 'RF']:
            log(f"Training {model_type}...")
            model = get_model(model_type)
            model.fit(X_train, y_train)
            with open(f"{model_type}_model.pkl", "wb") as f:
                pickle.dump(model, f)
            log(f"{model_type} training completed ✅")
        log("\nAll models trained successfully!")

        ttk.Button(progress_win, text="Close", command=progress_win.destroy).pack(pady=10)

    except Exception as e:
        messagebox.showerror("Training Error", str(e))

# -------------------- Testing -------------------- #
test_results = []  # store predictions for comparison

def test_models_gui():
    global test_results
    try:
        df_test = pd.read_csv("test_data.csv")
        X_test = np.array([extract_features_from_sequence(seq) for seq in df_test["sequence"]])
        sequences = df_test["sequence"].values

        result_window = tk.Toplevel(root)
        result_window.title("Test Predictions")
        result_window.geometry("950x600")
        result_window.configure(bg="#f8f9fa")

        canvas = tk.Canvas(result_window, bg="#f8f9fa")
        frame = tk.Frame(canvas, bg="#f8f9fa")
        scrollbar = tk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=frame, anchor="nw")

        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        frame.bind("<Configure>", on_configure)

        all_predictions = []
        row = 0
        for model_type in ['SVM', 'NN', 'RF']:
            model = load_model(model_type)
            preds = model.predict(X_test)

            tk.Label(frame, text=f"{model_type} Predictions",
                     font=("Helvetica", 13, "bold"), bg="#f8f9fa", fg="#007acc").grid(row=row, column=0, columnspan=2, pady=10)
            row += 1

            for i, (seq, pred) in enumerate(zip(sequences, preds), start=1):
                label_text = "Driver Gene (1)" if pred==1 else "Non-driver Gene (0)"
                color = "#dc3545" if pred==1 else "#28a745"
                bg_color = "#ffffff" if i % 2 == 0 else "#f8f9fa"

                tk.Label(frame, text=f"{i}. {seq}", font=("Courier", 10),
                         bg=bg_color, fg="#333").grid(row=row, column=0, sticky="w", padx=10, pady=2)
                tk.Label(frame, text=label_text, font=("Helvetica", 10, "bold"),
                         bg=bg_color, fg=color).grid(row=row, column=1, sticky="w", padx=10, pady=2)
                row += 1
                all_predictions.append([model_type, seq, pred])

        test_results = all_predictions  # save globally

        def save_results():
            df = pd.DataFrame(all_predictions, columns=["Model","Sequence","Prediction"])
            df.to_csv("predictions.csv", index=False)
            messagebox.showinfo("Saved", "Predictions saved to predictions.csv")

        ttk.Button(result_window, text="💾 Save Results", command=save_results).pack(pady=10)
        ttk.Button(result_window, text="📊 Compare Models", command=compare_all_models).pack(pady=5)

    except Exception as e:
        messagebox.showerror("Testing Error", str(e))

# -------------------- Compare Models -------------------- #
def compare_all_models():
    global test_results
    if not test_results:
        messagebox.showerror("Error", "No test predictions found. Run Test Predictions first.")
        return

    # Load confirmation labels
    if not os.path.exists("confirmation.csv"):
        messagebox.showerror("Error", "confirmation.csv not found. Create with 'sequence' and 'label'.")
        return
    conf_df = pd.read_csv("confirmation.csv")

    # Merge predictions with actual labels
    df_pred = pd.DataFrame(test_results, columns=["Model","Sequence","Prediction"])
    df_pred["Sequence"] = df_pred["Sequence"].str.upper()
    conf_df["sequence"] = conf_df["sequence"].str.upper()
    merged = pd.merge(df_pred, conf_df, left_on="Sequence", right_on="sequence", how="inner")
    if merged.empty:
        messagebox.showerror("Error", "No matching sequences between test predictions and confirmation.csv")
        return

    accuracies = {}
    fig, axes = plt.subplots(2,3, figsize=(15,8))

    for idx, model_type in enumerate(['SVM','NN','RF']):
        df_model = merged[merged["Model"]==model_type]
        y_true = df_model["label"].values
        y_pred = df_model["Prediction"].values
        acc = np.mean(y_true == y_pred)
        accuracies[model_type] = acc

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-driver","Driver"])
        disp.plot(ax=axes[0,idx], colorbar=False)
        axes[0,idx].set_title(f"{model_type}\nAccuracy={acc:.2f}")

    # Accuracy bar chart
    axes[1,0].bar(accuracies.keys(), accuracies.values(), color=['#007acc']*3)
    axes[1,0].set_ylim(0,1)
    axes[1,0].set_title("Accuracy Comparison")
    for i,(m,acc) in enumerate(accuracies.items()):
        axes[1,0].text(i, acc+0.02, f"{acc:.2f}", ha='center', fontweight="bold")
    axes[1,1].axis("off")
    axes[1,2].axis("off")

    fig.suptitle("Model Comparison (Predictions vs Confirmation)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

# -------------------- Single Sequence Prediction -------------------- #
def predict_sequence(model_type, sequence):
    features = extract_features_from_sequence(sequence).reshape(1,-1)
    model = load_model(model_type)
    return model.predict(features)[0]

def predict_sequence_gui():
    seq = seq_entry.get().strip().upper()

    # Minimum length requirement
    MIN_LEN = 10
    if not seq or any(c not in "ATGC" for c in seq):
        messagebox.showerror("Invalid Input", "Enter a valid DNA sequence (A/T/G/C only)")
        return
    if len(seq) < MIN_LEN:
        messagebox.showerror("Invalid Input", f"Sequence must be at least {MIN_LEN} nucleotides long")
        return

    result_window = tk.Toplevel(root)
    result_window.title("Prediction Result")
    result_window.geometry("400x300")
    result_window.configure(bg="#ffffff")

    tk.Label(result_window, text="🔬 Prediction Results",
             font=("Helvetica", 15, "bold"), bg="#ffffff", fg="#333").pack(pady=15)

    for model_type in ['SVM','NN','RF']:
        pred = predict_sequence(model_type, seq)
        label = "🧬 Driver Gene (1)" if pred==1 else "✅ Non-driver Gene (0)"
        color = "#dc3545" if pred==1 else "#28a745"
        tk.Label(result_window, text=f"{model_type}: {label}",
                 font=("Helvetica", 12, "bold"), bg="#ffffff", fg=color).pack(pady=5)
    ttk.Button(result_window, text="OK", command=result_window.destroy).pack(pady=15)

# -------------------- GUI Setup -------------------- #
root = tk.Tk()
root.title("Cancer Driver Gene Predictor")
root.geometry("800x500")
root.configure(bg="#f0f4f8")

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Helvetica",12,"bold"), background="#007acc", foreground="white", padding=8)
style.map("TButton", background=[("active","#005f99")])

tk.Label(root, text="🧬 Cancer Driver Gene Prediction Tool", font=("Helvetica",20,"bold"),
         bg="#f0f4f8", fg="#222").pack(pady=15)

ttk.Button(root, text="Train Models", command=train_models_gui, width=25).pack(pady=10)
ttk.Button(root, text="Testing", command=test_models_gui, width=25).pack(pady=10)

tk.Label(root, text="Enter DNA Sequence (A/T/G/C):", font=("Helvetica",13), bg="#f0f4f8", fg="#444").pack(pady=10)
seq_entry = tk.Entry(root, font=("Courier",13), width=50, bd=2, relief="groove")
seq_entry.pack(pady=5)

ttk.Button(root, text="Predict Sequence", command=predict_sequence_gui, width=25).pack(pady=15)
tk.Label(root, text="Developed by Harsha", font=("Helvetica",10,"italic"),
         bg="#f0f4f8", fg="#666").pack(side="bottom", pady=10)

root.mainloop()
