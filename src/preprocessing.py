# importação de bibliotecas e config. de visualização

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("Bibliotecas importadas com sucesso!")
df = pd.read_csv('data/raw/creditcard.csv')

"""## SPLIT DOS DADOS EM TREINO, VALIDAÇÃO E TESTE

Split HOLDOUT 70-15-15 com estratificação (mantém proporção das classes)

Primeiro split: 70% treino, 30% temporário (val+test)

Segundo split: dividir o temporário em validação (15%) e teste (15%).
"""

# criar um identificador da transação usando o índice como ID.
df = df.copy()
df['id'] = df.index

y = df['Class'] #target

# Features: remover Class e id (id não deve entrar no modelo)
X = df.drop(columns=['Class', 'id'])

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(f"Treino: {X_train.shape[0]} amostras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validação: {X_val.shape[0]} amostras ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Teste: {X_test.shape[0]} amostras ({X_test.shape[0]/len(X)*100:.1f}%)")
print("\nDistribuição do target por conjunto:")
print(f"Treino: {y_train.value_counts(normalize=True).to_dict()}")
print(f"Validação: {y_val.value_counts(normalize=True).to_dict()}")
print(f"Teste: {y_test.value_counts(normalize=True).to_dict()}")

# Separar os IDs do conjunto de teste (para salvar ids_test.csv depois)
ids_test = df.loc[X_test.index, 'id']

"""## Engenharia de features

Apesar de poder capturar padrões temporais, o uso direto de Time pode introduzir um componente posicional que depende do recorte de coleta do conjunto (janela temporal específica), prejudicando a generalização do modelo para novos períodos. Além disso, como as variáveis V1 a V28 já foram transformadas por PCA, o atributo Time não está no mesmo espaço transformado e pode atuar como uma fonte adicional de variação não relacionada ao comportamento transacional em si. Dessa forma, optou-se por remover Time do conjunto de features antes da normalização e modelagem, reduzindo o risco de o modelo aprender padrões temporais específicos do dataset em vez de padrões associados à fraude.
A aplicação de transformações adicionais poderia introduzir redundância, aumentar a complexidade do espaço de atributos e elevar o risco de overfitting, especialmente em um cenário de dados altamente desbalanceados e com foco em detecção de anomalias. Dessa forma, foi priorizada uma maior manutenção da estrutura original das features, garantindo maior robustez, interpretabilidade e alinhamento com os modelos adotados no projeto.
"""

if 'Time' in X_train.columns:
    X_train = X_train.drop(columns=['Time'])
    X_val   = X_val.drop(columns=['Time'])
    X_test  = X_test.drop(columns=['Time'])

"""## Escalonamento"""

scaler = StandardScaler()
scaler.fit(X_train)     #apenas no conjunto de treino

# Aplicação da normalização
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index
)

X_val_scaled = pd.DataFrame(
    X_val_scaled,
    columns=X_val.columns,
    index=X_val.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns,
    index=X_test.index
)

X_train_scaled.describe().loc[['mean', 'std']] #verificação

"""Após a divisão dos dados em conjuntos de treino, validação e teste, foi aplicado o
escalonamento das variáveis por meio do método de padronização (StandardScaler).
O scaler foi ajustado exclusivamente sobre o conjunto de treino, evitando vazamento
de informação, e posteriormente aplicado aos conjuntos de validação e teste.

A padronização é essencial neste projeto, uma vez que os modelos utilizados são
sensíveis à escala das variáveis, garantindo que nenhuma feature domine as demais
por diferenças de magnitude.

## Output
"""

os.makedirs('data/processed', exist_ok=True)

# features
X_train_scaled.to_csv('data/processed/X_train_processed.csv', index=False)
X_test_scaled.to_csv('data/processed/X_test_processed.csv', index=False)

# targets
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

# ids dos testes
ids_test.to_csv('data/processed/ids_test.csv', index=False)

os.listdir('data/processed')

"""Ao final do pré-processamento, os conjuntos de dados foram exportados para arquivos
CSV, seguindo o contrato de dados definido no projeto. Os arquivos gerados incluem
as features normalizadas para treino e teste, os respectivos rótulos e os
identificadores das amostras de teste, garantindo compatibilidade com as etapas
posteriores de modelagem e avaliação.

"""