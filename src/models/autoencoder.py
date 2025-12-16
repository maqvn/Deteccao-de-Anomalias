import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report
)

# ==========================================================
# CONFIGURAÇÕES GERAIS
# ==========================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

DATA_PATH = "data/processed"
OUTPUT_PATH = "outputs"

N_JOBS = -1  # usa todos os núcleos da CPU

# ==========================================================
# 1. PREPARAÇÃO DOS DADOS
# ==========================================================

def load_and_split_data(data_path, test_size=0.3):
    X = pd.read_csv(os.path.join(data_path, "X_train_processed.csv"))
    y = pd.read_csv(os.path.join(data_path, "y_train.csv"))["Class"]

    mask_normal = y == 0

    X_normal = X[mask_normal]
    y_normal = y[mask_normal]

    X_anomaly = X[~mask_normal]
    y_anomaly = y[~mask_normal]

    X_train_pure, X_val_normal, y_train_pure, y_val_normal = train_test_split(
        X_normal,
        y_normal,
        test_size=test_size,
        random_state=RANDOM_SEED
    )

    X_val_combined = pd.concat([X_val_normal, X_anomaly], ignore_index=True)
    y_val_combined = pd.concat([y_val_normal, y_anomaly], ignore_index=True)

    print(f"Treino puro (classe 0): {X_train_pure.shape}")
    print(f"Validação interna (0 + 1): {X_val_combined.shape}")

    return X_train_pure, X_val_combined, y_val_combined


# ==========================================================
# 2. MODELO
# ==========================================================

def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))

    encoded = Dense(int(input_dim * 0.75), activation="relu")(input_layer)
    bottleneck = Dense(encoding_dim, activation="relu")(encoded)
    decoded = Dense(int(input_dim * 0.75), activation="relu")(bottleneck)

    output_layer = Dense(input_dim, activation="sigmoid")(decoded)

    return Model(inputs=input_layer, outputs=output_layer)


# ==========================================================
# 3. TREINAMENTO INDIVIDUAL (PARALELIZÁVEL)
# ==========================================================

def train_single_model(params, X_train_pure, X_val, y_val):
    start_time = time.time()

    model = build_autoencoder(X_train_pure.shape[1], params["encoding_dim"])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        loss="mean_squared_error"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=0
    )

    model.fit(
        X_train_pure,
        X_train_pure,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        shuffle=True,
        validation_data=(X_val, X_val),
        callbacks=[early_stop],
        verbose=0
    )

    reconstructions = model.predict(X_val, verbose=0)
    mse = np.mean(np.square(X_val - reconstructions), axis=1)

    auc_pr = average_precision_score(y_val, mse)
    auc_roc = roc_auc_score(y_val, mse)

    elapsed = time.time() - start_time

    return auc_pr, auc_roc, params, model, elapsed


# ==========================================================
# 4. PLOTS
# ==========================================================

def save_error_distribution_plot(model, X_val, y_val):
    recon = model.predict(X_val, verbose=0)
    mse = np.mean(np.square(X_val - recon), axis=1)

    plt.figure(figsize=(10, 6))
    for cls in [0, 1]:
        plt.hist(
            mse[y_val == cls],
            bins=50,
            alpha=0.6,
            density=True,
            label=f"Classe {cls}"
        )

    plt.legend()
    plt.xlabel("Erro de Reconstrução (MSE)")
    plt.ylabel("Densidade")
    plt.title("Distribuição do Erro de Reconstrução")

    plt.savefig(os.path.join(OUTPUT_PATH, "ae_error_distribution.png"))
    plt.close()


def save_pr_curve_plot(precision, recall, best_idx, threshold):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="Autoencoder")
    plt.scatter(
        recall[best_idx],
        precision[best_idx],
        color="red",
        label=f"Threshold={threshold:.4f}"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_PATH, "ae_precision_recall_curve.png"))
    plt.close()


# ==========================================================
# 5. AVALIAÇÃO FINAL
# ==========================================================

def generate_final_scores(model):
    X_test = pd.read_csv(os.path.join(DATA_PATH, "X_test_processed.csv"))
    y_test = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv"))["Class"]
    ids_test = pd.read_csv(os.path.join(DATA_PATH, "ids_test.csv"))["id"]

    recon = model.predict(X_test, verbose=0)
    scores = np.mean(np.square(X_test - recon), axis=1)

    precision, recall, thresholds = precision_recall_curve(y_test, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx]

    preds = (scores > best_threshold).astype(int)

    print("\nRelatório Final:")
    print(classification_report(y_test, preds))

    pd.DataFrame({
        "id": ids_test,
        "anomaly_score": scores,
        "is_anomaly": preds
    }).to_csv(os.path.join(OUTPUT_PATH, "autoencoder_predictions.csv"), index=False)

    return precision, recall, best_idx, best_threshold


# ==========================================================
# 6. MAIN
# ==========================================================

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    X_train, X_val, y_val = load_and_split_data(DATA_PATH)

    param_grid = [
        {"encoding_dim": ed, "learning_rate": lr, "batch_size": bs, "epochs": 50}
        for ed in [4, 8, 16]
        for lr in [0.01, 0.001]
        for bs in [32, 64]
    ]

    print("\nIniciando Grid Search Paralelizado...")
    print(f"Total de combinações: {len(param_grid)}\n")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(train_single_model)(p, X_train, X_val, y_val)
        for p in param_grid
    )

    print("\nResultados do Grid Search:")
    for auc_pr, auc_roc, params, _, t in results:
        print(f"AUC-PR={auc_pr:.4f} | AUC-ROC={auc_roc:.4f} | {params} | {t:.1f}s")

    best = max(results, key=lambda x: x[0])
    best_auc_pr, _, best_params, best_model, _ = best

    print("\n🏆 Melhor Modelo:")
    print(f"AUC-PR: {best_auc_pr:.4f}")
    print(f"Parâmetros: {best_params}")

    save_error_distribution_plot(best_model, X_val, y_val)

    precision, recall, best_idx, threshold = generate_final_scores(best_model)
    save_pr_curve_plot(precision, recall, best_idx, threshold)


main()
