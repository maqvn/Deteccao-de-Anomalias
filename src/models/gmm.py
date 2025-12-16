import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# GMM Model
# Prepare the data for GMM
# Exclude 'Time' and 'Class' columns for clustering

# 1. Carregamento dos Dados
X_train = pd.read_csv('data/processed/X_train_processed.csv')
X_test = pd.read_csv('data/processed/X_test_processed.csv')
ids_test = pd.read_csv('data/processed/ids_test.csv')


CONTAMINATION_RATE = 0.01 

# 2. Configuração e Treinamento do Modelo
print("Treinando GMM...")
# n_components=2 conforme seu código original
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_train)

# 3. Geração dos Scores (Inyferência)
# score_samples retorna o log da verossimilhança (Log-Likelihood)
# Valores ALTOs (perto de 0 ou positivos) = Normal
# Valores BAIXOS (muito negativos) = Anomalia
log_likelihood = gmm.score_samples(X_test)

# Para transformar em "Grau de Anomalia" (onde quanto maior, mais anômalo), 
# invertemos o sinal do log-likelihood.
anomaly_scores = -log_likelihood

# 4. Definição do Threshold para Classificação Binária (is_anomaly)
# Aqui definimos o limite para considerar algo como anomalia (1) ou normal (0).
# Uma estratégia comum é usar um percentil dos scores de treino ou definir manualmente.
# Vamos calcular o threshold que separa o top X% mais anômalos
threshold = np.percentile(anomaly_scores, 100 * (1 - CONTAMINATION_RATE))

# Gera a classificação binária baseada no score e no threshold
predictions = (anomaly_scores > threshold).astype(int)

# 5. Montagem do DataFrame de Saída
output_df = pd.DataFrame({
    'id': ids_test.iloc[:, 0],  # Pega a primeira coluna do arquivo de IDs
    'anomaly_score': anomaly_scores,
    'is_anomaly': predictions
})

# 6. Validação e Salvamento
print("\n--- Amostra do Output ---")
print(output_df.head())

print(f"\nDistribuição de Classes Preditas:\n{output_df['is_anomaly'].value_counts()}")

output_path = 'outputs/gmm_predictions.csv'
output_df.to_csv(output_path, index=False)
print(f"\nArquivo salvo com sucesso em: {output_path}")

# -----------------------------------------------

# # Hyperparameter Tuning for GMM
# # 1. Carregar dados (apenas X é necessário para o treino do GMM)
# X = pd.read_csv('data/processed/X_train_processed.csv')

# # Opcional: usar apenas uma amostra se o dataset for gigante para ser mais rápido
# # X_sample = X.sample(frac=0.5, random_state=42) 

# # Definição dos parâmetros que queremos testar
# n_components_range = range(1, 10) # Testar de 1 a 9 componentes
# covariance_types = ['full', 'tied', 'diag', 'spherical']

# best_gmm = None
# best_bic = np.inf

# results = []

# print("Iniciando busca pelos melhores parâmetros do GMM...")

# for n_components in n_components_range:
#     for cv_type in covariance_types:
#         # Instancia e treina o modelo
#         gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=42)
#         gmm.fit(X)
        
#         # Calcula o BIC (quanto menor, melhor)
#         bic = gmm.bic(X)
        
#         results.append({
#             'n_components': n_components,
#             'covariance_type': cv_type,
#             'bic': bic
#         })
        
#         # Salva se for o melhor até agora
#         if bic < best_bic:
#             best_bic = bic
#             best_gmm = gmm
#             print(f"Novo melhor encontrado: {n_components} componentes, tipo {cv_type} (BIC: {bic:.2f})")

# # Converte resultados para DataFrame para visualização
# results_df = pd.DataFrame(results)
# print("\n--- Top 5 Melhores Configurações ---")
# print(results_df.sort_values('bic').head())

# print("\nMelhor modelo final selecionado:")
# print(best_gmm)

# -----------------------------------------

# # Aplicação do Melhor Modelo GMM nos Dados de Teste
# # 1. Carregar os dados de TESTE e IDs (que não foram usados no treino)
# # Certifique-se de que os caminhos estão corretos
# X_test = pd.read_csv('data/processed/X_test_processed.csv')
# ids_test = pd.read_csv('data/processed/ids_test.csv')

# print(f"\nAplicando o melhor modelo (n_comp={best_gmm.n_components}, type={best_gmm.covariance_type}) nos dados de teste...")

# # 2. Definição do Threshold (Limite de Anomalia)
# # A melhor prática é definir o threshold olhando para o treino (o que é "normal").
# # Vamos assumir uma taxa de contaminação (ex: 1% dos dados são anomalias).
# CONTAMINATION_RATE = 0.01  # Ajuste isso conforme seu conhecimento do negócio (ex: 0.005, 0.05)

# # Calculamos os scores do TREINO para saber onde fica o corte dos 99% mais normais
# scores_train = best_gmm.score_samples(X) 
# anomaly_scores_train = -scores_train # Invertemos o sinal: quanto maior, mais anômalo
# threshold = np.percentile(anomaly_scores_train, 100 * (1 - CONTAMINATION_RATE))

# print(f"Threshold calculado baseada no treino (Top {CONTAMINATION_RATE*100}%): {threshold:.4f}")

# # 3. Aplicação no Teste
# # Calculamos a log-verossimilhança do teste
# log_likelihood_test = best_gmm.score_samples(X_test)

# # Convertemos para "Grau de Anomalia" (positivo e crescente)
# anomaly_scores_test = -log_likelihood_test

# # 4. Classificação Binária (0 ou 1)
# # Se o score do teste for maior que o limite definido no treino, é anomalia.
# is_anomaly_pred = (anomaly_scores_test > threshold).astype(int)

# # 5. Montagem do DataFrame Final
# output_df = pd.DataFrame({
#     'id': ids_test.iloc[:, 0], # Pega a coluna de ID
#     'anomaly_score': anomaly_scores_test,
#     'is_anomaly': is_anomaly_pred
# })

# # 6. Salvar Resultados
# output_filename = 'outputs/gmm_tuned_predictions.csv'
# output_df.to_csv(output_filename, index=False)

# print("\n--- Resultado Final ---")
# print(output_df.head())
# print(f"\nContagem de Anomalias no Teste:\n{output_df['is_anomaly'].value_counts()}")
# print(f"\nArquivo salvo em: {output_filename}")