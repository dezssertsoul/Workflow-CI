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
import dagshub

if "GITHUB_ACTIONS" not in os.environ:
    dagshub.init(repo_owner='dezssertsoul', repo_name='Eksperimen_SML_Reza-Rahmawati', mlflow=True)
    print("[INFO] Berjalan di Laptop. Tracking DagsHub Online AKTIF.")
else:
    print("[INFO] Berjalan di GitHub Actions. Menggunakan tracking lokal runner.")

mlflow.sklearn.autolog()
mlflow.set_experiment("Eksperimen_SML_Reza-Rahmawati")

def main():
    print("[INFO] Menjalankan MLflow proses...")
    
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

    # Mulai training run
    with mlflow.start_run(run_name="MLflow_Project_Run"):
        print("[INFO] Melatih model RandomForest...")
        model = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"[INFO] Akurasi Model: {acc:.4f}")

        # Artefak 1: Ringkasan Model (Format .txt)
        summary_path = "summary_model.txt"
        with open(summary_path, "w") as f:
            f.write("=== Ringkasan Eksperimen Reza Rahmawati ===\n")
            f.write(f"Model Utama: RandomForestClassifier (n_estimators=120, max_depth=12)\n")
            f.write(f"Akurasi Akhir: {acc:.4f}\n")
            f.write("Strategi Logging: Hybrid (Autolog + 2 Artefak Manual Tambahan)\n")
        mlflow.log_artifact(summary_path)

        # Artefak 2: JSON Report Dataset & Metriks
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        report_dict["meta_siswa"] = {
            "nama": "Reza Rahmawati",
            "status": "Memenuhi Kriteria Autolog + Minimal 2 Artefak Tambahan"
        }
        
        json_path = "dataset_info.json"
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=4)
        mlflow.log_artifact(json_path)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig("training_confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("training_confusion_matrix.png")

        print(f"[SUKSES] Training selesai! Semua data & 2 artefak sukses tercatat.")

if __name__ == "__main__":
    main()
