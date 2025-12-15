import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# --- CONFIGURAÇÕES DE INTEGRAÇÃO ---
INPUT_X_TEST = 'data/processed/X_test_processed.csv'
INPUT_IDS = 'data/processed/ids_test.csv'
OUTPUT_DIR = 'outputs'
OUTPUT_FILE = 'dbscan_predictions.csv'

# Hiperparâmetros
EPS_OTIMO = 2.9619
MIN_SAMPLES = 14

def main():
    print("--- INICIANDO MODELO DBSCAN (MODO INTEGRAÇÃO) ---")
    
    # 1. Verificar e Ler Dados Processados
    if not os.path.exists(INPUT_X_TEST) or not os.path.exists(INPUT_IDS):
        print(f"ERRO CRÍTICO: Arquivos processados não encontrados em 'data/processed/'.")
        print("Certifique-se de que o pré-processamento (Integrante 1) foi rodado antes.")
        return

    print(f"Lendo dados de: {INPUT_X_TEST}")
    X_input = pd.read_csv(INPUT_X_TEST)
    ids_test = pd.read_csv(INPUT_IDS)
    
    # Garantir que IDs sejam uma série unidimensional
    if isinstance(ids_test, pd.DataFrame):
        ids_test = ids_test.iloc[:, 0]

    # 2. Aplicação do PCA
    print("Aplicando PCA (Redução para 10 componentes)...")
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_input)
    
    # 3. Rodar DBSCAN
    print(f"Rodando DBSCAN (eps={EPS_OTIMO}, min_samples={MIN_SAMPLES})...")
    db = DBSCAN(eps=EPS_OTIMO, min_samples=MIN_SAMPLES, n_jobs=-1)
    labels = db.fit_predict(X_pca)
    
    # 4. Formatar Saída
    is_anomaly = [1 if x == -1 else 0 for x in labels]
    anomaly_score = [1.0 if x == 1 else 0.0 for x in is_anomaly]
    
    df_out = pd.DataFrame({
        'id': ids_test,
        'anomaly_score': anomaly_score,
        'is_anomaly': is_anomaly
    })
    
    # Validação de Tamanho
    print(f"\n--- RESULTADO FINAL ---")
    print(f"Total de linhas processadas: {len(df_out)}")
    print(f"Anomalias detectadas: {sum(is_anomaly)}")
    
    # 5. Salvar
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df_out.to_csv(save_path, index=False)
    print(f"Sucesso! Arquivo salvo em: {save_path}")

if __name__ == "__main__":
    main()