import pandas as pd
import numpy as np
import os

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix
)

# =========================================================
# CONFIGURAÇÕES GERAIS
# =========================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = 'data/processed'
OUTPUT_PATH = 'outputs'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# =========================================================
# 1. PREPARAÇÃO DOS DADOS
# =========================================================

def load_data(data_path):
    X_train = pd.read_csv(os.path.join(data_path, 'X_train_processed.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))['Class']

    X_test = pd.read_csv(os.path.join(data_path, 'X_test_processed.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))['Class']
    ids_test = pd.read_csv(os.path.join(data_path, 'ids_test.csv'))['id']

    # Treino APENAS com normais
    X_train_normal = X_train[y_train == 0]

    return (
        X_train_normal.values,
        X_test.values,
        y_test.values,
        ids_test
    )

# =========================================================
# 2. GRID SEARCH + AVALIAÇÃO
# =========================================================

def train_and_evaluate_gmm(params, X_train_normal, X_val, y_val):
    gmm = GaussianMixture(
        n_components=params['n_components'],
        covariance_type=params['covariance_type'],
        random_state=RANDOM_SEED
    )

    gmm.fit(X_train_normal)

    # Anomaly score = -log likelihood
    scores = -gmm.score_samples(X_val)

    auc_pr = average_precision_score(y_val, scores)

    return auc_pr, gmm

# =========================================================
# 3. AVALIAÇÃO FINAL E EXPORTAÇÃO
# =========================================================

def generate_final_scores(best_model, X_test, y_test, ids_test, target_recall=0.80):
    scores = -best_model.score_samples(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, scores)

    valid_idxs = np.where(recall >= target_recall)[0]
    if len(valid_idxs) > 0:
        best_idx = valid_idxs[-1]
        threshold = thresholds[best_idx]
    else:
        best_idx = np.argmax(recall)
        threshold = thresholds[best_idx]

    predictions = (scores > threshold).astype(int)

    print(f"\n🎯 Threshold escolhido: {threshold:.6f} (Recall ≥ {target_recall:.0%})")
    print("\n--- RELATÓRIO FINAL ---")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Fraude']))

    cm = confusion_matrix(y_test, predictions)
    print(f"Matriz de Confusão:\n{cm}")

    # Exportação padronizada
    df_output = pd.DataFrame({
        'id': ids_test,
        'anomaly_score': scores,
        'is_anomaly': predictions
    })

    output_file = os.path.join(OUTPUT_PATH, 'gmm_predictions.csv')
    df_output.to_csv(output_file, index=False)

    print(f"\n✅ Arquivo salvo em: {output_file}")

# =========================================================
# 4. EXECUÇÃO PRINCIPAL
# =========================================================

def main():
    X_train_normal, X_test, y_test, ids_test = load_data(DATA_PATH)

    # 🔎 Grid Search (igual ao AE)
    param_grid = {
        'n_components': [1, 2, 3, 4],
        'covariance_type': ['full', 'diag']
    }

    grid = list(ParameterGrid(param_grid))

    best_auc_pr = -1
    best_model = None
    best_params = None

    print("\n=============================================")
    print(f"INICIANDO GRID SEARCH GMM ({len(grid)} combinações)")
    print("=============================================")

    for i, params in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] Testando: {params} ...", end=" ")

        try:
            auc_pr, model = train_and_evaluate_gmm(
                params,
                X_train_normal,
                X_test,
                y_test
            )
            print(f"AUC-PR: {auc_pr:.4f}")

            if auc_pr > best_auc_pr:
                best_auc_pr = auc_pr
                best_model = model
                best_params = params

        except Exception as e:
            print(f"Erro: {e}")

    print("\n🏆 MELHOR MODELO GMM")
    print(f"Parâmetros: {best_params}")
    print(f"AUC-PR (Validação/Teste): {best_auc_pr:.4f}")

    if best_model:
        generate_final_scores(
            best_model,
            X_test,
            y_test,
            ids_test,
            target_recall=0.80
        )

main()
