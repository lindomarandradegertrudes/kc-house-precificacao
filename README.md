# Precificação de Imóveis — King County

Projeto avaliativo do Módulo 1 do curso *Desenvolvimento de IA para Análise Preditiva* (Rede SESI/SENAI SC).

## Problema

Uma imobiliária do condado de King County (EUA) deseja estimar o valor de venda de um imóvel a partir de suas características físicas e de localização. A variável-alvo é `price` (valor numérico contínuo, em dólares).

## Dataset

`kc_house_data.csv` — Opção A do projeto, fornecida pelo curso. Aproximadamente 21.600 registros de vendas de imóveis no condado de King County, EUA, com colunas como número de quartos e banheiros, área construída, área do terreno, número de andares, avaliação do estado de conservação e localização (zipcode, latitude e longitude).

## Técnicas e Tecnologias

- **Linguagem:** Python
- **Bibliotecas:** pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels
- **Pipeline:**
  1. Análise Exploratória de Dados (EDA)
  2. Tratamento e limpeza (duplicatas, valores ausentes, outliers)
  3. Feature Engineering (`idade_imovel`, `foi_reformado`, `preco_por_m2`)
  4. Preparação para modelagem (análise de multicolinearidade via VIF, split treino/teste, escalonamento com StandardScaler)
  5. Modelagem com Regressão Linear e diagnóstico de overfitting
  6. Avaliação de métricas, análise gráfica e versionamento do modelo

## Principais Decisões Técnicas

- **Valores ausentes** em `sqft_above` (2 registros) foram tratados por imputação lógica exata, usando a relação `sqft_above = sqft_living - sqft_basement`.
- **Outlier** de 33 quartos foi identificado como erro de digitação (área incompatível com a quantidade informada) e removido.
- **Multicolinearidade:** `sqft_above` e `sqft_basement` foram removidas do conjunto de variáveis preditoras por apresentarem VIF infinito em relação a `sqft_living` (colinearidade perfeita, já que `sqft_living = sqft_above + sqft_basement`).
- **Localização:** optou-se por usar `lat` e `long` como proxy geográfico, em vez de `zipcode` bruto, evitando a necessidade de codificação com dezenas de colunas.

## Resultados (Modelo v1)

| Métrica | Valor |
|---|---|
| MAE | US$ 127.599,03 |
| MSE | 46.005.714.494,72 |
| RMSE | US$ 214.489,43 |
| R² | 0,6935 |

O modelo apresentou diferença pequena entre o RMSE de treino (US$ 200.265,58) e de teste (US$ 214.489,43), indicando boa capacidade de generalização, sem overfitting significativo. O modelo tende a subestimar o valor de imóveis de alto padrão (acima de US$ 2 milhões), onde a dispersão dos erros é maior.

O modelo treinado (v1) e suas métricas estão versionados em `models/v1/`.

## Estrutura do Projeto

```
data/
├── raw/            # dataset original (kc_house_data.csv)
├── processed/       # dataset após limpeza
└── final/           # recorte usado na modelagem
models/
└── v1/
    ├── modelo_regressao_v1.pkl
    └── metricas_v1.json
notebooks/
└── projeto_precificacao.ipynb   # notebook principal com todo o pipeline
outputs/
└── figures/          # gráficos gerados durante a análise
src/                    # modularização do pipeline (diferencial)
├── __init__.py
├── config.py            # caminhos e parâmetros do projeto
├── dataset.py            # carga/salvamento dos dados
├── features.py            # limpeza + colunas derivadas
├── plots.py                # funções de visualização
└── modeling/
    ├── __init__.py
    └── train.py            # preparação, treino, diagnóstico e versionamento
 .gitignore
 LICENSE
 README.md
requirements.txt
```

## Modularização (src/)

Como diferencial de organização, as funções desenvolvidas ao longo do notebook foram reorganizadas em módulos Python dentro da pasta `src/`, que o notebook importa e reutiliza. Isso separa a lógica de cada etapa do pipeline (carga de dados, limpeza, visualização, modelagem) em arquivos próprios, facilitando manutenção e reuso do código. A demonstração de uso desses módulos está na seção final do notebook.

## Melhorias Futuras (v2)

- Tratamento mais elaborado da variável `zipcode` (agrupamento por região antes de encoding), em vez de usar apenas `lat`/`long`.
- Testar modelos não lineares (Árvore de Decisão, KNN, Random Forest) para capturar melhor a subestimação observada em imóveis de alto padrão.
- Validação cruzada para uma estimativa mais robusta de desempenho.

## Como Executar

```bash
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
pip install -r requirements.txt
```

Depois, abra `notebooks/projeto_precificacao.ipynb` no VSCode ou Jupyter e execute as células em ordem.

## Vídeo de Apresentação

[link do Google Drive aqui]

## Autor

Lindomar Andrade Gertrudes
