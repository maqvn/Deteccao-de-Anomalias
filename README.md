# Detecção de Anomalias (Projeto AMCD)

Projeto da disciplina de **Aprendizado de Máquina e Ciência de Dados (AMCD)**.

**Objetivo:** Implementar e comparar três abordagens distintas (**Deep Learning**, **Densidade** e **Probabilística**) para a detecção de anomalias/fraudes em um conjunto de dados desbalanceado.

## Modelos Implementados

1. **Autoencoder** — Abordagem de Reconstrução
2. **DBSCAN** — Abordagem de Densidade
3. **Gaussian Mixture Models (GMM)** — Abordagem Probabilística

---

## Sobre o Dataset

Utilizaremos o dataset **Credit Card Fraud Detection**, disponível no Kaggle.

### Características

- **Conteúdo:** Transações de cartões de crédito de clientes europeus em setembro de 2013.
- **Volume:** 284.807 transações.
- **Desbalanceamento:** Apenas 492 fraudes (0,172%). Dataset altamente desbalanceado, com classe positiva rara.
- **Privacidade:** As features `V1`, `V2`, ..., `V28` são o resultado de uma transformação **PCA (Principal Component Analysis)**, aplicada para proteger a identidade dos usuários.
- **Features Originais:** Apenas `Time` (segundos desde a primeira transação) e `Amount` (valor da transação) não foram transformadas.

### Justificativa da Escolha

Optamos por este dataset para concentrar o esforço do projeto na **comparação algorítmica** e na **análise de sensibilidade dos modelos**. Como as principais features já passaram por PCA, elas apresentam propriedades estatísticas relevantes — como descorrelação — que favorecem a convergência de modelos como **GMM** e **Autoencoders**. Isso permite uma análise mais profunda das nuances de cada abordagem, reduzindo o impacto de ruídos típicos de dados brutos não estruturados.

---

## Estrutura do Repositório

O método de organização visa separar dados brutos, código de exploração (notebooks) e código de produção (`src`).

```text
projeto-anomalia/
├── data/
│   ├── raw/                  # Dados originais imutáveis (NÃO commitar arquivos grandes)
│   ├── processed/            # Dados limpos e normalizados (gerados pelo script de limpeza)
│   └── mocks/                # Dados falsos para testes de integração
├── notebooks/                # Área de experimentação e rascunho
│   ├── 01_eda_analise.ipynb
│   ├── 02_proto_autoencoder.ipynb
│   ├── 02_proto_dbscan.ipynb
│   └── 02_proto_gmm.ipynb
├── src/                      # Código final modularizado
│   ├── preprocessing.py      # Funções de limpeza e split
│   ├── evaluation.py         # Funções para curvas ROC e métricas
│   └── models/               # Scripts finais dos modelos
│       ├── autoencoder.py
│       ├── dbscan.py
│       └── gmm.py
├── outputs/                  # Predições salvas pelos modelos (CSV)
├── requirements.txt          # Dependências do projeto
└── README.md                 # Este arquivo
```

---

## Branches

* **main**: Produção. atualização exclusivamente via *Pull Request* (PR).
* **preprocessing**: Limpeza, EDA e split dos dados.
* **model-autoencoder**: Desenvolvimento da Rede Neural.
* **model-dbscan**: Desenvolvimento do DBSCAN e PCA.
* **model-gmm**: Desenvolvimento do GMM e análise de distribuição.

### Fluxo de Trabalho

1. Crie sua branch a partir da `main`.
2. Desenvolva e teste no seu notebook.
3. Exporte o código limpo para a pasta `src/`.
4. Abra um *Pull Request* para a `main` ao finalizar.

---

## Contrato de Interface de Dados

Para garantir a paralelização do trabalho, os formatos de entrada e saída são pré-definidos.

### Entrada dos modelos

Todos os modelos devem ler os dados da pasta `data/processed/`:

* **X_train_processed.csv**
  Features numéricas normalizadas, sem coluna alvo (`target`) e sem ID.

* **X_test_processed.csv**
  Mesmo formato do conjunto de treino.

* **y_test.csv**
  Gabarito oficial para validação (coluna única binária: `0 = Normal`, `1 = Anomalia`).

* **ids_test.csv**
  IDs correspondentes às linhas de teste (para cruzamento de resultados).

### Saída dos modelos

Todo modelo deve salvar suas predições na pasta `outputs/`, seguindo **exatamente** este formato:

* **Nome do arquivo:** `[nome_modelo]_predictions.csv`
  Exemplo: `autoencoder_predictions.csv`

#### Estrutura do CSV

| Coluna          | Tipo      | Descrição                                               |
| --------------- | --------- | ------------------------------------------------------- |
| `id`            | int / str | Identificador da transação (deve coincidir com o input) |
| `anomaly_score` | float     | Grau de anomalia (quanto maior, mais anômalo)           |
| `is_anomaly`    | int       | Classificação binária baseada no *threshold* (0 ou 1)   |

#### 📌 Exemplo de CSV de Saída

```csv
id,anomaly_score,is_anomaly
1024,0.954,1
1025,0.021,0
1026,0.110,0
```

---

## Como Executar (Ambiente)

Para garantir compatibilidade, todos devem usar as mesmas versões das bibliotecas.

### Clone o repositório

```bash
git clone https://github.com/maqvn/Deteccao-de-Anomalias.git
```

### Crie um ambiente virtual (opcional, mas recomendado)

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

### Instale as dependências

```bash
pip install -r requirements.txt
```

---

## Desenvolvimento com Mocks

Enquanto os dados reais não estiverem prontos (limpeza em andamento), utilize os arquivos da pasta `data/mocks/`.

* Possuem a **mesma estrutura de colunas e tipos de dados** dos arquivos reais.
* Seu código deve funcionar alterando apenas o caminho de leitura de `data/processed/` para `data/mocks/`.
