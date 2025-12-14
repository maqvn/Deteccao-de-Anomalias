import numpy as np
import pandas as pd
import os

# --- Configurações ---
N_SAMPLES_TEST = 1000   # Amostras para o conjunto de teste
N_SAMPLES_TRAIN = 4000  # Amostras para o conjunto de treino (simula ser maior)
N_FEATURES = 30         # V1-V28, Time, Amount
FRAUD_RATE = 0.01       # Simula 1% de fraude (desbalanceamento)
SEED = 42

# --- Caminhos e Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'mocks')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Pasta criada: {OUTPUT_DIR}")

np.random.seed(SEED)

def generate_labels(n_samples, fraud_rate):
    """Cria um array de labels com a taxa de fraude especificada."""
    labels = np.zeros(n_samples, dtype=int)
    num_anomalies = int(n_samples * fraud_rate)
    labels[:num_anomalies] = 1
    np.random.shuffle(labels)
    return labels, num_anomalies

# --- 1. Geração do Conjunto de Treino (X_train e Y_train) ---

# Gera dados aleatórios normalizados (entre 0 e 1)
X_train_mock = np.random.rand(N_SAMPLES_TRAIN, N_FEATURES)
feature_names = [f'V{i}' for i in range(1, 29)] + ['Time_norm', 'Amount_norm']
df_X_train = pd.DataFrame(X_train_mock, columns=feature_names)

# Cria o gabarito Y_train
y_train_mock, num_frauds_train = generate_labels(N_SAMPLES_TRAIN, FRAUD_RATE)

# SALVAR X_train_processed.csv
df_X_train.to_csv(os.path.join(OUTPUT_DIR, 'X_train_processed.csv'), index=False)
print(f"\nGerado: X_train_processed.csv ({N_SAMPLES_TRAIN} linhas)")

# SALVAR Y_train.csv
pd.Series(y_train_mock, name='Class').to_csv(os.path.join(OUTPUT_DIR, 'Y_train.csv'), index=False)
print(f"Gerado: Y_train.csv (Fraudes simuladas: {num_frauds_train})")

# --- 2. Geração do Conjunto de Teste (X_test, Y_test, IDs) ---

X_test_mock = np.random.rand(N_SAMPLES_TEST, N_FEATURES)
df_X_test = pd.DataFrame(X_test_mock, columns=feature_names)

y_test_mock, num_frauds_test = generate_labels(N_SAMPLES_TEST, FRAUD_RATE)

# SALVAR X_test_processed.csv
df_X_test.to_csv(os.path.join(OUTPUT_DIR, 'X_test_processed.csv'), index=False)
print(f"Gerado: X_test_processed.csv ({N_SAMPLES_TEST} linhas)")

# SALVAR Y_test.csv
pd.Series(y_test_mock, name='Class').to_csv(os.path.join(OUTPUT_DIR, 'Y_test.csv'), index=False)
print(f"Gerado: Y_test.csv (Fraudes simuladas: {num_frauds_test})")

# SALVAR ids_test.csv
ids_test_mock = np.arange(10000, 10000 + N_SAMPLES_TEST)
pd.Series(ids_test_mock, name='id').to_csv(os.path.join(OUTPUT_DIR, 'ids_test.csv'), index=False)
print(f"Gerado: ids_test.csv")

print("\nMocks prontos no caminho 'data/mocks/'.")