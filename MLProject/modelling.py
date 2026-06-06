import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn

IS_GITHUB = "GITHUB_ACTIONS" in os.environ

if not IS_GITHUB:
    import dagshub
    dagshub.init(repo_owner='dezssertsoul', repo_name='Eksperimen_SML_Reza-Rahmawati', mlflow=True)
    mlflow.sklearn.autolog()
    mlflow.set_experiment("Eksperimen_SML_Reza-Rahmawati")
    print("[INFO] Berjalan di Laptop. Tracking DagsHub Online AKTIF.")

def main():
    print("[INFO] Memulai proses re-training model...")

    data_path = 'Midterm_53_group_preprocessed.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] File {data_path} tidak ditemukan!")
        return

    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Proses Training
    print("[INFO] Melatih model RandomForest...")
    model = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Akurasi Model: {acc:.4f}")

    summary_path = "summary_model.txt"
    with open(summary_path, "w") as f:
        f.write("=== Ringkasan Eksperimen Reza Rahmawati ===\n")
        f.write(f"Model Utama: RandomForestClassifier\n")
        f.write(f"Akurasi Akhir: {acc:.4f}\n")
    
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_dict["meta_siswa"] = {"nama": "Reza Rahmawati"}
    json_path = "dataset_info.json"
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=4)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig("training_confusion_matrix.png")
    plt.close()

    if not IS_GITHUB:
        mlflow.log_artifact(summary_path)
        mlflow.log_artifact(json_path)
        mlflow.log_artifact("training_confusion_matrix.png")

    print("[SUKSES] Seluruh proses training dan pembuatan file artefak selesai!")

if __name__ == "__main__":
    main()
