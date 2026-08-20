import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Dataset 
def load_dataset(path):
    df = pd.read_csv(path)
    return df

# Dataset Preprocessing
def preprocess(df):
    df = df.dropna()

    X = df.drop("tsunami", axis=1)
    y = df["tsunami"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Regression Baseline
def run_logistic_regression(X_train, X_test, y_train, y_test):

    baseline = LogisticRegression(max_iter=300)
    baseline.fit(X_train, y_train)

    y_pred = baseline.predict(X_test)
    y_prob = baseline.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    print("\nBaseline Model Complete.\n")


# FNN Architecture
def build_fnn(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# FNN
def train_fnn_no_es(model, X_train, y_train):
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=300,
        batch_size=32,
        verbose=1
    )
    return history

# FNN with Early Stopping
def train_fnn_with_es(model, X_train, y_train):
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=300,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    return history

# Model Evaluation
def evaluate_model(model, X_test, y_test, title="Model"):
    print("Model Eval")

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    y_prob = model.predict(X_test).flatten()

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix — {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Plot Learning Curves
def plot_learning_curves(history, title="Training Curves"):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Start Main
def main():

    print("Tsunami Prediction using FNN")

    dataset_path = "earthquake_data_tsunami.csv"

    df = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = preprocess(df)

# Baseline
    run_logistic_regression(X_train, X_test, y_train, y_test)

# FNN
    print("FNN")

    model_no_es = build_fnn(X_train.shape[1])
    history_no_es = train_fnn_no_es(model_no_es, X_train, y_train)

    evaluate_model(model_no_es, X_test, y_test)
    plot_learning_curves(history_no_es)

# FNN with Early Stopping
    print("FNN WITH EARLY STOPPING")

    model_es = build_fnn(X_train.shape[1])
    history_es = train_fnn_with_es(model_es, X_train, y_train)

    evaluate_model(model_es, X_test, y_test, title="FNN with EarlyStopping")
    plot_learning_curves(history_es, title="FNN (EarlyStopping)")

# Run main
if __name__ == "__main__":
    main()
