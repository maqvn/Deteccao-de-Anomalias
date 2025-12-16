import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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
# MODO DE EXECUÇÃO
# 0 = Execução normal (apenas melhores hiperparâmetros)
# 1 = Grid Search (tunagem completa)
# =========================================================
RUN_TUNING = 0

# =========================================================
# 1. PREPARAÇÃO DOS DADOS
# =========================================================

def load_data(data_path):
    # Carrega os dados (certifique-se que os arquivos existem no caminho)
    try:
        X_train = pd.read_csv(os.path.join(data_path, 'X_train_processed.csv'))
        y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))['Class']

        X_test = pd.read_csv(os.path.join(data_path, 'X_test_processed.csv'))
        y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))['Class']
        ids_test = pd.read_csv(os.path.join(data_path, 'ids_test.csv'))['id']
        
        # Treino APENAS com normais
        X_train_normal = X_train[y_train == 0]
        
        return X_train_normal, X_test, y_test, ids_test
    except FileNotFoundError as e:
        print(f"Erro ao carregar arquivos: {e}")
        exit()

# =========================================================
# 2. TREINAMENTO E AVALIAÇÃO
# =========================================================

def train_and_evaluate_gmm(params, X_train, X_test, y_test):
    # Instancia o modelo
    gmm = GaussianMixture(
        n_components=params['n_components'],
        covariance_type=params['covariance_type'],
        random_state=RANDOM_SEED
    )
    
    # Treina apenas com dados normais
    gmm.fit(X_train)
    
    # Avalia no conjunto de teste (Score: Log-likelihood negativo)
    # Quanto menor o log-likelihood, maior a chance de ser anomalia
    # Multiplicamos por -1 para que scores ALTOS sejam anomalias
    scores = -gmm.score_samples(X_test)
    
    auc_pr = average_precision_score(y_test, scores)
    
    return auc_pr, gmm, scores

# =========================================================
# 4. EXECUÇÃO PRINCIPAL
# =========================================================

def main():
    X_train_normal, X_test, y_test, ids_test = load_data(DATA_PATH)

    # LÓGICA DE SELEÇÃO DE PARÂMETROS (Igual ao Autoencoder)
    if RUN_TUNING:
        print(">>> MODO: GRID SEARCH ATIVADO")
        param_grid = {
            'n_components': [1, 2, 3, 4],
            'covariance_type': ['full', 'diag']
        }
    else:
        print(">>> MODO: EXECUÇÃO ÚNICA (MELHORES PARÂMETROS)")
        # Melhores parâmetros identificados no Grid Search anterior
        param_grid = {
            'n_components': [3],
            'covariance_type': ['full']
        }

    grid = list(ParameterGrid(param_grid))

    best_auc_pr = -1
    best_model = None
    best_params = None
    best_scores = None

    print("\n=============================================")
    print(f"INICIANDO EXECUÇÃO ({len(grid)} combinações)")
    print("=============================================")

    for i, params in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] Testando: {params} ...", end=" ")

        try:
            auc_pr, model, scores = train_and_evaluate_gmm(
                params,
                X_train_normal,
                X_test,
                y_test
            )
            print(f"AUC-PR: {auc_pr:.4f}")

            # Salva o melhor modelo (ou o único, se RUN_TUNING=0)
            if auc_pr > best_auc_pr:
                best_auc_pr = auc_pr
                best_model = model
                best_params = params
                best_scores = scores

        except Exception as e:
            print(f"Erro: {e}")

    print("\n🏆 MELHOR RESULTADO")
    print(f"Parâmetros: {best_params}")
    print(f"AUC-PR Final: {best_auc_pr:.4f}")

    # =========================================================
    # GERAÇÃO DE RESULTADOS FINAIS (Do melhor modelo)
    # =========================================================
    
    # 1. Definir Threshold para Recall ~0.80
    precision, recall, thresholds = precision_recall_curve(y_test, best_scores)
    
    # Busca o limiar onde recall é o mais próximo possível de 0.80
    target_recall = 0.80
    idx = (np.abs(recall - target_recall)).argmin()
    final_threshold = thresholds[idx]
    
    print(f"\n🎯 Threshold escolhido: {final_threshold:.6f} (Recall aprox {recall[idx]:.2f})")
    
    # 2. Gerar predições binárias
    y_pred = (best_scores >= final_threshold).astype(int)
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    # 3. Salvar CSV de predições
    results_df = pd.DataFrame({
        'id': ids_test,
        'anomaly_score': best_scores,
        'is_anomaly': y_pred
    })
    
    csv_path = os.path.join(OUTPUT_PATH, 'gmm_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nArquivo salvo em: {csv_path}")


main()