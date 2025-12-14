# Detecção de Anomalias (Projeto AMCD)

Projeto da disciplina de **Aprendizado de Máquina e Ciência de Dados (AMCD)**.

**Objetivo:** Implementar e comparar três abordagens distintas (**Deep Learning**, **Densidade** e **Probabilística**) para a detecção de anomalias/fraudes em um conjunto de dados desbalanceado.

---

## Modelos Implementados

1. **Autoencoder** — Abordagem de Reconstrução
2. **DBSCAN** — Abordagem de Densidade
3. **Gaussian Mixture Models (GMM)** — Abordagem Probabilística

---

## Dataset

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

## Divisão de Papéis e Responsabilidades

O projeto adota uma estrutura de trabalho **modular e paralela**, permitindo que as atividades avancem simultaneamente com **baixo acoplamento** entre as partes.  
A **Engenharia de Dados** fornece a base comum para a **Modelagem**, reduzindo gargalos de integração.

### Papéis do Projeto

| Papel | Integrante | Foco | Responsabilidades |
|------|------------|------|-------------------|
| **Eng. de Dados & Avaliação** | *Integrante 1* | Infraestrutura e Métricas | • Limpeza, normalização e split dos dados<br>• Geração dos arquivos em `data/processed/`<br>• Implementação do `evaluation.py`<br>• Cálculo de métricas (AUC-ROC, F1, Recall)<br>• Geração de gráficos comparativos |
| **Esp. em Deep Learning** | *Integrante 2* | Autoencoder (Reconstrução) | • Implementação do `autoencoder.py` (Keras/PyTorch)<br>• Ajuste do gargalo (*bottleneck*) e *learning rate*<br>• Geração do `anomaly_score` via **erro de reconstrução** |
| **Esp. em Densidade** | *Integrante 3* | DBSCAN (Geometria / Ruído) | • Aplicação de PCA para otimizar o modelo<br>• Implementação do `dbscan.py`<br>• Ajuste de `epsilon` e `min_samples`<br>• Uso da classe `-1` (ruído) como anomalia |
| **Esp. Probabilístico** | *Integrante 4* | GMM (Distribuição) | • Implementação do `gmm.py`<br>• Ajuste do número de componentes e tipo de covariância<br>• Cálculo do `anomaly_score` via **probabilidade invertida** \\(1 − P(x)\\) |

---

## Contrato de Interface de Dados

O **Contrato de Interface de Dados** define formalmente os **formatos de entrada e saída** utilizados no projeto.  
Seu objetivo é permitir **desenvolvimento paralelo**, garantindo que o **Engenheiro de Dados & Avaliação (Integrante 1)** consiga integrar e comparar os resultados dos três modelos **sem incompatibilidades**.

### Entrada dos Modelos  
**Responsabilidade exclusiva: Integrante 1 (Engenharia de Dados & Avaliação)**

O Integrante 1 é o **único responsável** por gerar, validar e versionar os arquivos abaixo na pasta `data/processed/`.  
Todos os modelos **devem consumir exatamente esses arquivos**, sem exceções.

| Arquivo | Conteúdo Garantido | Responsável | Justificativa |
|-------|------------------|------------|---------------|
| **`X_train_processed.csv`** | Features numéricas **normalizadas**, sem ID e sem `target` | Integrante 1 | Base de treino limpa para aprendizado **não supervisionado**, focada na estrutura dos dados normais |
| **`X_test_processed.csv`** | Features de teste, sem ID e sem `target` | Integrante 1 | Garante avaliação justa, com os três modelos testados no **mesmo conjunto de dados** |
| **`y_test.csv`** | Coluna única binária (`0 = Normal`, `1 = Anomalia`) | Integrante 1 | Gabarito oficial (`y_true`) para cálculo de métricas |
| **`ids_test.csv`** | Coluna única com IDs das transações | Integrante 1 | Permite o alinhamento entre predições e gabarito, assegurando a integridade dos resultados |

---

### Saída dos Modelos  
**Responsabilidade: Integrantes 2, 3 e 4 (Modelagem)**

Cada especialista em modelagem deve salvar suas predições na pasta `outputs/`, **obedecendo rigorosamente** ao formato definido neste contrato.

- **Nome do arquivo:** [nome_modelo]_predictions.csv
Exemplo: `autoencoder_predictions.csv`

### Estrutura Obrigatória do CSV

| Coluna | Tipo | Finalidade |
|------|------|------------|
| **`id`** | int / str | **Chave de cruzamento** usada para alinhar a predição ao `y_test.csv` |
| **`anomaly_score`** | float | **Score contínuo** utilizado para cálculo da **Curva ROC** e do **AUC-ROC** (capacidade máxima do modelo) |
| **`is_anomaly`** | int (0 ou 1) | Classificação binária após aplicação do *threshold*, usada para métricas como **F1-score** e **Recall** |

### 📌 Exemplo de CSV de Saída (Contrato de Interface)

```csv
id,anomaly_score,is_anomaly
1024,0.954,1
1025,0.021,0
1026,0.110,0
```

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
