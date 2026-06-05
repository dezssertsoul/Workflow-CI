import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

import dagshub
import mlflow
import mlflow.sklearn

def main():
   
    REPO_OWNER = 'dezssertsoul'  
    REPO_NAME = 'Eksperimen_SML_Reza-Rahmawati'

    print("[INFO] Menghubungkan ke DagsHub...")
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

    # Membaca dataset lokal
    data_path = 'Midterm_53_group_preprocessed.csv'
    df = pd.read_csv(data_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("CI_Automation_Project")

    with mlflow.start_run(run_name="MLflow_Project_Run"):
        model = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "logged_model")

        # Buat Artefak Tambahan 1 (Confusion Matrix)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig("training_confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("training_confusion_matrix.png")

        # Buat Artefak Tambahan 2 (JSON Report)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        with open("metric_info.json", "w") as f:
            json.dump(report_dict, f, indent=4)
        mlflow.log_artifact("metric_info.json")

        print(f"[SUKSES] Re-training selesai! Akurasi: {acc:.4f}")

if __name__ == "__main__":
    main()