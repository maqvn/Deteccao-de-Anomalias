# Detecção de Anomalias (Projeto AMCD)

Projeto desenvolvido para a disciplina de **Aprendizado de Máquina e Ciência de Dados (AMCD)**, com foco na comparação de diferentes abordagens para **detecção de anomalias/fraudes** em dados altamnte desbalanceados.

---

## Objetivo do Projeto

Implementar, avaliar e comparar três paradigmas distintos de detecção de anomalias:

* **Deep Learning** (Reconstrução)
* **Modelos Baseados em Densidade**
* **Modelos Probabilísticos**

A comparação é realizada sob um mesmo conjunto de dados, protocolo experimental e métricas, garantindo uma análise justa e reprodutível.

---

## Modelos Implementados

1. **Autoencoder** — Abordagem de Reconstrução
2. **DBSCAN** — Abordagem de Densidade
3. **Gaussian Mixture Models (GMM)** — Abordagem Probabilística

---

## Dataset Utilizado

Será utilizado o dataset **Credit Card Fraud Detection**, disponibilizado publicamente no Kaggle.

### Características Principais

* **Conteúdo:** Transações de cartões de crédito de clientes europeus (setembro de 2013).
* **Volume:** 284.807 transações.
* **Desbalanceamento:** Apenas 492 fraudes (0,172%), caracterizando um cenário altamente desbalanceado.
* **Privacidade:** As features `V1`, `V2`, ..., `V28` resultam de uma transformação por **PCA (Principal Component Analysis)**, aplicada para anonimização.
* **Features Não Transformadas:**

  * `Time`: segundos desde a primeira transação
  * `Amount`: valor da transação

### Justificativa da Escolha

A escolha deste dataset permite concentrar o esforço do projeto na **análise algorítmica** e na **sensibilidade dos modelos**, minimizando problemas oriundos de dados brutos não estruturados.

Como as principais features já passaram por PCA, elas apresentam propriedades estatísticas desejáveis — como descorrelação — que favorecem a convergência e estabilidade de modelos como **GMM** e **Autoencoders**, possibilitando uma comparação mais precisa entre abordagens.

### Instalação do Dataset (Dados Reais)

Devido ao tamanho (~150MB) e às boas práticas de versionamento, o dataset original **não está incluído no repositório**.

1.  **Download:** Acesse a página oficial no Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2.  **Arquivo:** Baixe e extraia o arquivo `creditcard.csv`.
3.  **Localização:** Salve o arquivo exatamente no seguinte caminho:
    ```text
    data/raw/creditcard.csv
    ```
    > **Nota:** O arquivo `.gitignore` deste projeto já está configurado para ignorar qualquer CSV na pasta `data/raw/`, garantindo que dados sensíveis ou pesados não sejam enviados ao GitHub.

---

## Organização do Trabalho

O projeto adota uma estrutura **modular e paralela**, permitindo que diferentes partes avancem simultaneamente com **baixo acoplamento**. Um contrato claro de dados e responsabilidades reduz conflitos de integração.

### Divisão de Papéis e Responsabilidades

| Papel                         | Integrante     | Foco                      | Responsabilidades                                                                                                                                                                                                  |
| ----------------------------- | -------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Eng. de Dados & Avaliação** | *Integrante 1* | Infraestrutura e Métricas | • Limpeza, normalização e split dos dados<br>• Geração de arquivos em `data/processed/`<br>• Implementação do `evaluation.py`<br>• Cálculo de métricas (AUC-ROC, F1, Recall)<br>• Geração de gráficos comparativos |
| **Esp. em Deep Learning**     | *Integrante 2* | Autoencoder               | • Implementação do `autoencoder.py` (Keras/PyTorch)<br>• Ajuste do gargalo (*bottleneck*) e *learning rate*<br>• Geração do `anomaly_score` via erro de reconstrução                                               |
| **Esp. em Densidade**         | *Integrante 3* | DBSCAN                    | • Aplicação de PCA para otimização<br>• Implementação do `dbscan.py`<br>• Ajuste de `epsilon` e `min_samples`<br>• Uso da classe `-1` como anomalia                                                                |
| **Esp. Probabilístico**       | *Integrante 4* | GMM                       | • Implementação do `gmm.py`<br>• Ajuste do número de componentes e covariância<br>• Cálculo do `anomaly_score` via probabilidade invertida (1 − P(x))                                                              |

---

## Contrato de Interface de Dados

### Entrada dos Modelos (`data/processed/`)

* `X_train_processed.csv` — Features normalizadas (sem target e sem ID)
* `y_train.csv` — Gabarito (0 = Normal, 1 = Anomalia)
* `X_test_processed.csv` — Mesmo formato do treino
* `y_test.csv` — Gabarito (0 = Normal, 1 = Anomalia)
* `ids_test.csv` — Identificadores das amostras de teste

### Saída dos Modelos (`outputs/`)

* **Arquivo:** `[nome_modelo]_predictions.csv`

| Coluna          | Tipo      | Descrição                  |
| --------------- | --------- | -------------------------- |
| `id`            | int / str | Identificador da transação |
| `anomaly_score` | float     | Grau de anomalia           |
| `is_anomaly`    | int       | Classificação binária      |

```csv
id,anomaly_score,is_anomaly
1024,0.954,1
1025,0.021,0
1026,0.110,0
```

---

## Estrutura do Repositório

A organização do repositório separa claramente **dados**, **experimentação** e **código de produção**, facilitando manutenção e avaliação.

```text
projeto-anomalia/
├── data/
│   ├── raw/                  # Dados originais imutáveis
│   ├── processed/            # Dados limpos e normalizados
│   └── mocks/                # Dados sintéticos para testes
├── notebooks/                # Exploração e prototipagem
│   ├── _eda_exploratory_data_analysis.ipynb
│   ├── model_autoencoder.ipynb
│   ├── model_dbscan.ipynb
│   ├── model_gmm.ipynb
│   └── models_evaluation.ipynb
├── src/                      # Código final
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── models/
│       ├── autoencoder.py
│       ├── dbscan.py
│       └── gmm.py
├── outputs/                  # Predições dos modelos
├── requirements.txt
└── README.md
```

---

## Branches e Fluxo de Trabalho

### Branches

* **main**: Produção (atualizações apenas via Pull Request)
* **feature/preprocessing**: Limpeza, EDA e split
* **feature/model-autoencoder**: Desenvolvimento do Autoencoder
* **feature/model-dbscan**: Desenvolvimento do DBSCAN
* **feature/model-gmm**: Desenvolvimento do GMM

### Fluxo de Trabalho

1. Criar branch a partir da `main`.
2. Desenvolver e testar no notebook.
3. Exportar o código final para `src/`.
4. Abrir Pull Request para a `main`.

---

## Preparação do Ambiente

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

## 🚀 Guia de Execução (Pipeline)

Para reproduzir os resultados do projeto, siga a ordem de execução abaixo. O pipeline foi desenhado para que a saída de uma etapa sirva de entrada para a próxima.

### 1. Preparação dos Dados (Preprocessing)

Esta etapa carrega o dataset bruto, realiza a limpeza, normalização e a separação em treino/teste. Os arquivos processados serão salvos em `data/processed/`.

```bash
# Certifique-se de que o dataset (creditcard.csv) estejam em data/raw/
python src/preprocessing.py
```

### 2. Treinamento e Inferência dos Modelos

Após o pré-processamento, execute os scripts dos modelos. Cada script treina o modelo, gera as predições no conjunto de teste e salva os resultados (CSVs e gráficos) na pasta `outputs/.`

**Autoencoder (Reconstrução):**

```bash
python src/models/autoencoder.py
```

**Gaussian Mixture Models (Probabilístico):**

```Bash
python src/models/gmm.py
```

**DBSCAN (Densidade):**

```Bash
python src/models/dbscan.py
```

>    Nota: É possível configurar dentro de cada arquivo (variável RUN_TUNING) se deseja rodar a busca de hiperparâmetros (Grid Search) ou a execução rápida com os melhores parâmetros já fixados.

### 3. Avaliação Comparativa

Após gerar as predições de todos os modelos, execute o script de avaliação para gerar as métricas finais e comparações.
Bash

`python src/evaluation.py`

### 4. Análise Exploratória (Opcional)

Os notebooks presentes na pasta notebooks/ (como o EDA) servem para análise visual e estudos preliminares. Eles não são estritamente necessários para rodar o pipeline de produção, mas são recomendados para o entendimento dos dados.
Bash

`jupyter notebook notebooks/`

---

## Desenvolvimento com Mocks

Enquanto os dados reais não estiverem prontos:
* Rode o arquivo **src/generate_mocks.py**
* Utilize o conjunto de dados resultante em `data/mocks/`.
* Os arquivos possuem **mesma estrutura e tipos** dos dados reais.
* O código deve funcionar alterando apenas o caminho de leitura.
