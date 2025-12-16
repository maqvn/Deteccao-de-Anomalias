import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, classification_report
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÕES GERAIS
# =========================================================
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_PATH = 'data/processed'
OUTPUT_PATH = 'outputs'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


# =========================================================
# MODO DE EXECUÇÃO
# 0 = Execução normal (hiperparâmetros fixos)
# 1 = Grid Search (tunagem)
# =========================================================
RUN_TUNING = 0


# =========================================================
# 1. PREPARAÇÃO DOS DADOS (Mantido similar, com ajustes de tipo)
# =========================================================

def load_and_split_data(data_path, test_size=0.2): # Reduzi test_size para ter mais dados de treino
    # Carregamento seguro
    try:
        X_train = pd.read_csv(os.path.join(data_path, 'X_train_processed.csv'))
        y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))['Class'] # Atenção ao Y maiúsculo se for mock
    except FileNotFoundError:
        print("Arquivos não encontrados. Verifique o caminho.")
        return None

    mask_normal = y_train == 0
    X_normal = X_train[mask_normal]
    y_normal = y_train[mask_normal]

    X_anomaly = X_train[~mask_normal]
    y_anomaly = y_train[~mask_normal]

    # Divisão
    X_train_pure, X_val_normal, _, y_val_normal = train_test_split(
        X_normal, y_normal, test_size=test_size, random_state=RANDOM_SEED
    )

    # Validação combinada
    X_val_combined = pd.concat([X_val_normal, X_anomaly], ignore_index=True)
    y_val_combined = pd.concat([y_val_normal, y_anomaly], ignore_index=True)

    print(f"Treino Puro (Normal): {X_train_pure.shape}")
    print(f"Validação (Normal+Fraude): {X_val_combined.shape}")

    return (
        X_train_pure.values.astype(np.float32),
        X_val_normal.values.astype(np.float32),
        X_val_combined.values.astype(np.float32),
        y_val_combined.values
    )

# =========================================================
# 2. MODELO DEEP & SPARSE AUTOENCODER
# =========================================================

def build_deep_autoencoder(input_dim, encoding_dim, dropout_rate=0.2):
    input_layer = Input(shape=(input_dim,))
    
    # --- Encoder ---
    # Camada de Denoising via Dropout (substitui o ruído gaussiano manual)
    x = Dropout(dropout_rate)(input_layer)
    
    # Camadas Profundas para aprender não-linearidade
    x = Dense(24, activation='relu')(x)
    x = BatchNormalization()(x) # Estabiliza o treino
    x = Dense(16, activation='relu')(x)
    
    # --- Bottleneck (Gargalo) ---
    # Regularização L1 força "esparsidade" (o modelo aprende características essenciais)
    bottleneck = Dense(encoding_dim, activation='relu', activity_regularizer=l1(10e-5))(x)
    
    # --- Decoder ---
    x = Dense(16, activation='relu')(bottleneck)
    x = Dense(24, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Saída (Sigmoid pois dados estão entre 0 e 1)
    output = Dense(input_dim, activation='sigmoid')(x)
    
    return Model(input_layer, output)

def train_and_evaluate_run(params, X_train_pure, X_val_pure, X_val_combined, y_val_combined):
    # Desempacotar parâmetros
    encoding_dim = params['encoding_dim']
    lr = params['learning_rate']
    batch_size = params['batch_size']
    epochs = params['epochs']
    
    autoencoder = build_deep_autoencoder(X_train_pure.shape[1], encoding_dim)

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mean_squared_error' # MSE foca em penalizar grandes erros (fraudes)
    )

    # Callbacks para parar treino se não melhorar e reduzir LR se estagnar
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0)
    ]

    # Treino
    # Nota: Passamos X_train_pure como entrada E saída. 
    # O Dropout na primeira camada cuida do "ruído".
    autoencoder.fit(
        X_train_pure, X_train_pure,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_val_pure, X_val_pure),
        callbacks=callbacks,
        verbose=0
    )

    # Avaliação
    reconstructions = autoencoder.predict(X_val_combined, verbose=0)
    mse = np.mean(np.square(X_val_combined - reconstructions), axis=1)

    auc_pr = average_precision_score(y_val_combined, mse)
    # auc_roc = roc_auc_score(y_val_combined, mse) # Opcional

    return auc_pr, autoencoder

# =========================================================
# 3. AVALIAÇÃO FINAL
# =========================================================

def generate_final_scores(best_model, data_path, target_recall=0.80):
    # Carrega dados de teste
    try:
        X_test = pd.read_csv(os.path.join(data_path, 'X_test_processed.csv')).values.astype(np.float32)
        y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))['Class'].values
        ids_test = pd.read_csv(os.path.join(data_path, 'ids_test.csv'))['id']
    except Exception as e:
        print(f"Erro ao carregar teste: {e}")
        return

    # Gera scores
    reconstructions = best_model.predict(X_test, verbose=0)
    anomaly_scores = np.mean(np.square(X_test - reconstructions), axis=1)

    # Curva PR
    precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)

    # Estratégia de Threshold: Buscar o limiar que garante X% de Recall (captura de fraude)
    # Em fraude, geralmente preferimos Recall alto (pegar a fraude) mesmo que Precision caia um pouco
    valid_idxs = np.where(recall >= target_recall)[0]
    
    if len(valid_idxs) > 0:
        best_idx = valid_idxs[-1] # O último índice onde recall >= target (maior precision possível)
        threshold = thresholds[best_idx]
    else:
        best_idx = np.argmax(recall) # Fallback
        threshold = thresholds[best_idx]

    predictions = (anomaly_scores > threshold).astype(int)

    print(f"\n🎯 Threshold escolhido: {threshold:.6f} (Para Recall ~{target_recall:.0%})")
    print("\n--- RELATÓRIO FINAL ---")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Fraude']))
    
    # Matriz de Confusão simplificada
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predictions)
    print(f"Matriz de Confusão:\n{cm}")

    # =========================================================
    # EXPORTAÇÃO FINAL (CONTRATO DE SAÍDA)
    # =========================================================
    df_output = pd.DataFrame({
        'id': ids_test,
        'anomaly_score': anomaly_scores,
        'is_anomaly': predictions
    })

    output_file = os.path.join(OUTPUT_PATH, 'autoencoder_predictions.csv')
    df_output.to_csv(output_file, index=False)

    print(f"\n✅ Arquivo de predições salvo em: {output_file}")


# =========================================================
# 4. EXECUÇÃO PRINCIPAL
# =========================================================

def main():
    if not os.path.exists(DATA_PATH):
        print("Gere os mocks primeiro!")
        return
        
    data = load_and_split_data(DATA_PATH)
    if data is None: return
    X_train_pure, X_val_pure, X_val_combined, y_val_combined = data

    if RUN_TUNING == 1:
        # =========================
        # MODO TUNAGEM (GRID SEARCH)
        # =========================
        param_grid = {
            'encoding_dim': [4, 8],
            'learning_rate': [0.01, 0.001],
            'batch_size': [64, 128],
            'epochs': [50]
        }
    else:
        # =========================
        # MODO NORMAL / PRODUÇÃO
        # =========================
        param_grid = {
            'encoding_dim': [8],
            'learning_rate': [0.001],
            'batch_size': [128],
            'epochs': [50]
        }


    # Transforma o dicionário em lista de combinações
    grid = list(ParameterGrid(param_grid))

    best_auc_pr = -1
    best_model = None
    best_params = None

    print("\n=============================================")
    print(f"INICIANDO GRID SEARCH ({len(grid)} combinações)")
    print("=============================================")

    for i, params in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] Testando: {params} ...", end=" ")
        
        try:
            auc_pr, model = train_and_evaluate_run(
                params, X_train_pure, X_val_pure, X_val_combined, y_val_combined
            )
            print(f"AUC-PR: {auc_pr:.4f}")

            if auc_pr > best_auc_pr:
                best_auc_pr = auc_pr
                best_model = model
                best_params = params
        except Exception as e:
            print(f"Erro: {e}")

    print("\n🏆 MELHOR MODELO ENCONTRADO")
    print(f"Params: {best_params}")
    print(f"AUC-PR (Validação): {best_auc_pr:.4f}")

    if best_model:
        # Avalia no Teste (Simulando produção)
        # Definimos target_recall=0.8 (queremos pegar 80% das fraudes)
        generate_final_scores(best_model, DATA_PATH, target_recall=0.8)

main()