import numpy as np
import pandas as pd
import os

# --- 1. Definir o Contrato de Entrada (Deve ser o mesmo para todos) ---
N_SAMPLES_TEST = 1000  # Número razoável de linhas para teste rápido
N_FEATURES = 28 + 2    # V1-V28, Time, Amount = 30 features
SEED = 42              # Para reprodutibilidade
np.random.seed(SEED)

OUTPUT_DIR = 'data/mocks'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. Gerar Features Mock (X_test_processed.csv) ---

# Gera dados aleatórios seguindo a normalização (entre 0 e 1)
# Simulamos 30 features (V1-V28, Time, Amount)
X_test_mock = np.random.rand(N_SAMPLES_TEST, N_FEATURES)

# Criar DataFrame para salvar
feature_names = [f'V{i}' for i in range(1, 29)] + ['Time_norm', 'Amount_norm']
df_X_test = pd.DataFrame(X_test_mock, columns=feature_names)

# SALVAR X_test_processed.csv
df_X_test.to_csv(os.path.join(OUTPUT_DIR, 'X_test_processed.csv'), index=False)
print(f"Criado: {os.path.join(OUTPUT_DIR, 'X_test_processed.csv')}")


# --- 3. Gerar Gabarito (y_test.csv) ---

# Cria um gabarito desbalanceado (ex: 99% Normal, 1% Anomalia)
# Isso testa se os modelos conseguem lidar com o desbalanceamento
y_test_mock = np.zeros(N_SAMPLES_TEST, dtype=int)
# Insere anomalias aleatoriamente (1% das amostras)
num_anomalies = int(N_SAMPLES_TEST * 0.01)
y_test_mock[:num_anomalies] = 1 
np.random.shuffle(y_test_mock) # Embaralha para não ficarem só no início

# SALVAR y_test.csv
pd.Series(y_test_mock, name='Class').to_csv(os.path.join(OUTPUT_DIR, 'y_test.csv'), index=False)
print(f"Criado: {os.path.join(OUTPUT_DIR, 'y_test.csv')}")


# --- 4. Gerar IDs (ids_test.csv) ---

# IDs sequenciais simples para cruzamento
ids_test_mock = np.arange(10000, 10000 + N_SAMPLES_TEST)
# SALVAR ids_test.csv
pd.Series(ids_test_mock, name='id').to_csv(os.path.join(OUTPUT_DIR, 'ids_test.csv'), index=False)
print(f"Criado: {os.path.join(OUTPUT_DIR, 'ids_test.csv')}")


# --- 5. Gerar X_train_processed.csv (Apenas para simular a presença) ---

# O treino mock terá 4x mais dados, mas o mesmo formato
X_train_mock = np.random.rand(N_SAMPLES_TEST * 4, N_FEATURES)
df_X_train = pd.DataFrame(X_train_mock, columns=feature_names)
df_X_train.to_csv(os.path.join(OUTPUT_DIR, 'X_train_processed.csv'), index=False)
print(f"Criado: {os.path.join(OUTPUT_DIR, 'X_train_processed.csv')}")