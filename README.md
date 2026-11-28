# 🧬 Denv_classifier — Classificação Alignment-Free de Sorotipos do Vírus da Dengue (DENV-1 a DENV-4)

Repositório oficial do projeto científico de **classificação alignment-free dos quatro sorotipos epidêmicos do vírus da Dengue (`DENV-1`, `DENV-2`, `DENV-3` e `DENV-4`)** em fragmentos genômicos curtos, amplicons de RT-PCR e leituras simuladas de sequenciamento de nova geração (NGS), implementado em **Python / NumPy puro e scikit-learn** (sem dependência de PyTorch ou GPU).

---

## 🎯 Resumo Científico

* **Curadoria e Zero Vazamento (*Zero Data Leakage*):** A partir de 3.438 sequências brutas do NCBI, a filtragem biológica (`10.000–11.500 pb`, $\le 1\%$ de bases ambíguas `N`) e a deduplicação estrita por hash **SHA-256** (removendo 82 duplicatas intra-sorotipo e 1 conflito inter-sorotipo) resultaram em **3.109 genomas únicos** (`831` DENV-1, `908` DENV-2, `1.105` DENV-3, `265` DENV-4). O particionamento estratificado `80/20` ocorre **em nível de genoma completo** antes do fatiamento (`2.487` genomas de treino $\to$ `22.383` fragmentos de `2.000 pb`; `622` genomas de teste cego $\to$ `3.110` janelas não-sobrepostas).
* **Representação por Hexâmeros ($k = 6$):** Vetores normalizados dos $4^6 = 4.096$ $k$-mers orientados da fita senso ($A, C, G, T$, com $U \to T$), padronizados via `StandardScaler` ajustado exclusivamente no treino.
* **Arquitetura Principal (`LeanMLPDropout`):** Perceptron Multicamadas (PMC) enxuto em NumPy puro com camada oculta única de **32 neurônios ReLU**, ***Inverted Dropout* ($p = 0{,}3$)**, **regularização $L_2$ ($\alpha = 0{,}01$)** e otimizador **Adam** (`131.200` pesos sinápticos / `131.236` parâmetros treináveis — redução de 94% frente ao PMC grande `(512, 128)`).

---

## 📊 Tabela Mestra de Resultados ($k = 6$, Recorte Estritamente Inédito)

Avaliação no teste cego não-sobreposto (`N = 3.110`), reads inéditas (`< 300 pb`, `N = 81`) e amplicons inéditos (`300–999 pb`, `N = 201`) com **Média ± Desvio-Padrão em 5 sementes** sob ruído estocástico NGS (70% substituições + 30% indels):

| Modelo / Arquitetura | Teste 2.000 pb ($N=3.110$) | Reads `< 300 pb` ($N=81$) | Amplicons 0% ($N=201$) | Amplicons 4% ($N=201$) | Amplicons 10% ($N=201$) | Amplicons 15% ($N=201$) | ECE (Amplicons) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reg. Logística Multinomial ($L_2$)** | 99,87% | 86,42% | 88,56% | 87,26% ± 0,74% | 78,61% ± 2,16% | 69,85% ± 3,58% | 0,0784 |
| **PMC Enxuto `(32,)` + Dropout** | 99,90% | 82,72% | **91,04%** | **90,85% ± 0,51%** | 86,47% ± 1,23% | 78,11% ± 3,76% | **0,0513** |
| **PMC Grande `(512, 128)`** | 99,84% | 81,48% | 88,06% | 88,26% ± 1,32% | 80,60% ± 1,83% | 72,14% ± 2,85% | 0,1075 |
| **RBF Clássica (32 centros, Moody-Darken)** | **99,94%** | 81,48% | 90,05% | 90,35% ± 0,81% | **86,67% ± 0,86%** | **80,60% ± 2,48%** | — |
| **Segment-wise Conv1D (`1×1` Posicional)** | 99,90% | **88,89%** | **91,04%** | 89,45% ± 0,73% | 81,69% ± 1,98% | 68,76% ± 1,98% | — |

* **Significância Estatística (Teste Exato de McNemar — PMC Enxuto vs Regressão Logística em Amplicons Inéditos):** $p = 0{,}0156$ (4% ruído), $p = 0{,}00007$ (8% ruído), $p = 0{,}00012$ (10% ruído) e $p = 0{,}00151$ (15% ruído).
* **Teste Cego Externo Intocado (17 isolados independentes NCBI/BV-BRC $\to$ `193` amplicons de `500 pb`):** **95,34%** limpo (`184/193`), **94,61% ± 0,62%** (ruído 4%) e **91,61% ± 1,65%** (ruído 10%).

---

## 📂 Estrutura do Repositório

```text
Denv_classifier/
├── PMC.py                                          # Pipeline oficial, classe LeanMLPDropout e CLI
├── app.py                                          # Interface Web interativa (Streamlit)
├── README.md                                       # Guia rápido e resumo de resultados
├── RELATORIO_TECNICO_PMC_DENV.md                   # Documentação científica completa
├── requirements.txt                                # Dependências do ambiente Python
├── .gitignore                                      # Regras de versionamento Git
│
├── dados/
│   ├── treino/                                     # Genomas completos de referência (denv1..4.fasta)
│   └── testes/                                     # Validação externa e teste cego externo intocado
│
├── modelos/
│   ├── pipeline_pmc_denv.joblib                    # Modelo oficial serializado (StandardScaler + LeanMLPDropout)
│   └── legado_v1/                                  # Artefatos históricos da versão inicial
│
├── experimentos/                                   # Scripts reprodutíveis de ablação, benchmarks e figuras
│   ├── avaliar_validacao_externa.py                # Avaliação estratificada na base externa heterogênea
│   ├── experimento_comparativo_k5_k6.py            # Comparativo K=5 vs K=6
│   ├── experimento_arquiteturas_k6.py              # Ablação de topologias PMC em K=6
│   ├── experimento_arquiteturas_enxutas_ruido.py   # PMC Enxuto + Dropout vs Regressão Logística
│   ├── experimento_stress_ruido.py                 # Curva de estresse sob ruído NGS (0% a 15%)
│   ├── experimento_regressao_logistica_k4_k6.py    # Ablação K=4, 5 e 6 (Regressão Logística vs PMC)
│   ├── experimento_rbf_comparativo.py              # Rede Neural RBF (Moody-Darken) & Kernel RBF-LR
│   ├── experimento_cnn1d_posicao_genomica.py       # Segment-wise Conv1D 1x1 e varredura nas 10 regiões DENV
│   ├── experimento_validacao_final_sbc.py          # Validação SBC (janelas não-sobrepostas, ECE, 5 sementes)
│   └── gerar_graficos_artigo_600dpi.py             # Gerador de todas as figuras do artigo em 600 DPI
│
├── graficos/                                       # Figuras científicas oficiais em 600 DPI (.png)
├── resultados/                                     # Auditorias biológicas e tabelas de benchmark (.csv)
└── tests/                                          # Suíte com 12 testes unitários e de integração (unittest)
```

---

## 🚀 Instalação e Reprodução Passo a Passo

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Executar a Suíte de Testes Unitários e de Integração
```bash
python -m unittest discover -s tests -v
```

### 3. Treinar o Modelo Oficial e Gerar Auditoria
```bash
python PMC.py
```

### 4. Classificar um Arquivo FASTA via Linha de Comando (CLI)
```bash
python PMC.py amostra_referencia_denv1.fasta
```

### 5. Executar o Painel Web Interativo (Streamlit)
```bash
streamlit run app.py
```

### 6. Reproduzir os Experimentos e Figuras do Artigo (600 DPI)
```bash
python experimentos/experimento_regressao_logistica_k4_k6.py
python experimentos/experimento_rbf_comparativo.py
python experimentos/experimento_cnn1d_posicao_genomica.py
python experimentos/experimento_validacao_final_sbc.py
python experimentos/gerar_graficos_artigo_600dpi.py
```

---

## 📖 Citação

```bibtex
@misc{florindo2026denvclassifier,
  author       = {Filipe Matias Florindo and L{\'a}zaro Eduardo da Silva and Marcelo C. Mussel},
  title        = {Classifica{\c{c}}{\~a}o Alignment-Free de Sorotipos do V{\'i}rus da Dengue (DENV-1 a DENV-4): Influ{\^e}ncia do Comprimento da Sequ{\^e}ncia, Esparsidade de k-mers e Calibra{\c{c}}{\~a}o de Modelos Neurais},
  year         = {2026},
  howpublished = {\url{https://github.com/filipimatiasflorindo-arch/Denv_classifier}}
}
```
