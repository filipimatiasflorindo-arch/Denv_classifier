"""
================================================================================
CLASSIFICADOR DE SOROTIPOS DO VÍRUS DA DENGUE VIA PERCEPTRON MULTICAMADAS (PMC)
Trabalho Acadêmico de Graduação em Inteligência Artificial / Bioinformática
================================================================================

"""

import hashlib
import itertools
import os
import sys
from collections import Counter, defaultdict
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

# ==============================================================================
# 1. CONSTANTES E CONFIGURAÇÕES DE REFERÊNCIA
# ==============================================================================

BASES = ["A", "C", "G", "T"]
# Ordem canônica invariável dos 64 trímeros: AAA, AAC, ..., TTT
TRIMEROS_64 = ["".join(p) for p in itertools.product(BASES, repeat=3)]

ARQUIVOS_SOROTIPOS = {
    "denv1.fasta": "DENV-1",
    "denv2.fasta": "DENV-2",
    "denv3.fasta": "DENV-3",
    "denv4.fasta": "DENV-4",
}

RANDOM_STATE = 42

# ==============================================================================
# 2. DOCUMENTAÇÃO DOS HIPERPARÂMETROS DO PMC (MLPClassifier)
# ==============================================================================
"""
DOCUMENTAÇÃO TEÓRICA DOS HIPERPARÂMETROS:
- hidden_layer_sizes: Quantidade de camadas ocultas e número de neurônios por camada.
  Controla a capacidade de representação e mapeamento não-linear da rede neural.
- activation='relu': Função de ativação f(x) = max(0, x). Permite aprender
  relações não-lineares evitando a saturação do gradiente (vanishing gradient).
- solver='adam': Otimizador estocástico baseado em momentos de 1ª e 2ª ordem dos
  gradientes. Ajusta taxas de aprendizado individuais por peso sináptico.
- alpha: Coeficiente de regularização L2 (Weight Decay). Penaliza pesos
  excessivamente grandes adicionando penalidade à função de perda para prevenir sobreajuste.
- batch_size=32: Tamanho dos minilotes utilizados a cada passo de retropropagação.
- learning_rate_init=0.001: Taxa de aprendizado inicial do otimizador Adam.
- max_iter=500: Limite teto de épocas permitidas para o treinamento.
- early_stopping=True: Parada precoce caso a acurácia de validação interna não
  melhore após 10 épocas consecutivas (patience).
- validation_fraction=0.15: Fração dos dados de treino (15%) reservada internamente
  pelo scikit-learn para monitorar o critério de parada do early stopping.
  NOTA METODOLÓGICA: Esta validação interna NÃO se confunde com a validação cruzada
  externa (K-Fold) nem com o teste final independente.
- random_state=42: Semente pseudoaleatória para garantir reprodutibilidade estrita.
"""


# ==============================================================================
# 3. VALIDAÇÃO E CURADORIA DAS SEQUÊNCIAS
# ==============================================================================


def validar_sequencia(seq, min_len=10000, max_len=11500, max_n_ratio=0.01):
    """
    Valida a integridade da sequência nucleotídica.
    Retorna (is_valida, motivo_rejeicao).
    
    Critérios:
    - Comprimento compatível com genoma completo de Flavivírus (10.000 a 11.500 pb).
    - Proporção de bases ambíguas (N, R, Y, etc.) limitada a no máximo 1% (max_n_ratio).
    """
    if not seq or len(seq.strip()) == 0:
        return False, "Sequência vazia"

    seq = seq.upper().replace("U", "T")
    comp = len(seq)
    if not (min_len <= comp <= max_len):
        return False, f"Tamanho fora do intervalo ({comp} pb)"

    bases_validas = sum(1 for b in seq if b in BASES)
    proporcao_ambigua = (comp - bases_validas) / comp
    if proporcao_ambigua > max_n_ratio:
        return (
            False,
            f"Excesso de bases ambíguas ({proporcao_ambigua * 100:.2f}% > {max_n_ratio * 100:.1f}%)",
        )

    return True, None


def extrair_frequencias_trimeros(seq):
    """
    Extrai a frequência relativa normalizada dos 64 trímeros canônicos de DNA
    através de janela deslizante (passo 1). Trímeros com bases ambíguas são
    desconsiderados, garantindo que o vetor resultante some estritamente 1.0.
    """
    seq = seq.upper().replace("U", "T")
    total_janelas = len(seq) - 2
    if total_janelas <= 0:
        return [0.0] * 64

    # Contagem rápida em janela deslizante
    contagens = Counter(seq[i : i + 3] for i in range(total_janelas))

    # Soma apenas os trímeros estritamente canônicos {A, C, G, T}
    total_canonicos = sum(contagens[tri] for tri in TRIMEROS_64)
    if total_canonicos == 0:
        return [0.0] * 64

    return [contagens[tri] / total_canonicos for tri in TRIMEROS_64]


def carregar_fasta(caminho_arquivo, rotulo_sorotipo=None, max_n_ratio=0.01):
    """
    Lê um arquivo FASTA, aplica filtros de qualidade e registra os motivos de descarte.
    Retorna (registros_validos, estatisticas_arquivo).
    """
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    registos = []
    seq_atual = []
    header_atual = ""
    header_completo = ""
    estatisticas = {"total_lidas": 0, "aceitas": 0, "motivos_rejeicao": defaultdict(int)}

    def processar_sequencia(header_curto, header_full, seq_lista):
        if not seq_lista:
            return
        estatisticas["total_lidas"] += 1
        seq_str = "".join(seq_lista)
        valida, motivo = validar_sequencia(seq_str, max_n_ratio=max_n_ratio)
        if valida:
            estatisticas["aceitas"] += 1
            features = extrair_frequencias_trimeros(seq_str)
            # Hash SHA-256 para auditoria de duplicatas exatas
            sha256_hash = hashlib.sha256(
                seq_str.upper().replace("U", "T").encode()
            ).hexdigest()
            registos.append({
                "id": header_curto,
                "descricao": header_full,
                "features": features,
                "sorotipo": rotulo_sorotipo,
                "hash": sha256_hash,
                "comprimento": len(seq_str),
            })
        else:
            motivo_chave = (
                "Fora do Tamanho"
                if "Tamanho" in motivo
                else "Bases Ambíguas"
            )
            estatisticas["motivos_rejeicao"][motivo_chave] += 1

    with open(caminho_arquivo, "r", encoding="utf-8", errors="ignore") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            if linha.startswith(">"):
                if seq_atual:
                    processar_sequencia(header_atual, header_completo, seq_atual)
                    seq_atual = []
                header_completo = linha[1:].strip()
                header_atual = header_completo.split()[0]
            else:
                seq_atual.append(linha)
        if seq_atual:
            processar_sequencia(header_atual, header_completo, seq_atual)

    return registos, estatisticas


def construir_dataset(arquivos_sorotipos, max_n_ratio=0.01, desduplicar_exatos=True):
    """
    Carrega todos os arquivos FASTA, executa auditoria de duplicatas, remove conflitos
    inter-sorotipos e estrutura o DataFrame final de desenvolvimento e teste.
    """
    print("\n" + "=" * 78)
    print("ETAPA 1: AUDITORIA, CURADORIA E QUALIDADE DOS DADOS GENÔMICOS")
    print("=" * 78)

    todos_registros = []
    resumo_filtragem = []

    for caminho, sorotipo in arquivos_sorotipos.items():
        registros, stats = carregar_fasta(caminho, rotulo_sorotipo=sorotipo, max_n_ratio=max_n_ratio)
        todos_registros.extend(registros)
        resumo_filtragem.append({
            "Sorotipo": sorotipo,
            "Arquivo": caminho,
            "Total Lidas": stats["total_lidas"],
            "Aceitas": stats["aceitas"],
            "Rejeitadas (Tamanho)": stats["motivos_rejeicao"]["Fora do Tamanho"],
            "Rejeitadas (N > 1%)": stats["motivos_rejeicao"]["Bases Ambíguas"],
        })

    df_auditoria = pd.DataFrame(resumo_filtragem)
    print("\nEstatísticas de Leitura e Filtragem de Sequências:")
    print(df_auditoria.to_string(index=False))

    # Auditoria de duplicatas exatas via hash SHA-256
    hash_map = defaultdict(list)
    for reg in todos_registros:
        hash_map[reg["hash"]].append(reg)

    conflitos_rotulo = []
    duplicatas_mesmo_sorotipo = 0

    for sha_h, ocorrencias in hash_map.items():
        sorotipos_presentes = set(r["sorotipo"] for r in ocorrencias)
        if len(sorotipos_presentes) > 1:
            conflitos_rotulo.append(ocorrencias)
        elif len(ocorrencias) > 1:
            duplicatas_mesmo_sorotipo += (len(ocorrencias) - 1)

    print("\nAuditoria de Redundância Genética por Hash SHA-256:")
    print(f" - Total de sequências válidas pós-filtros de qualidade: {len(todos_registros)}")
    print(f" - Sequências nucleotídicas únicas: {len(hash_map)}")
    print(f" - Duplicatas exatas no mesmo sorotipo: {duplicatas_mesmo_sorotipo}")

    if conflitos_rotulo:
        print(f" [ALERTA METODOLÓGICO] Detectado(s) {len(conflitos_rotulo)} conflito(s) de anotação inter-sorotipo!")
        print("   Sequências 100% idênticas com rótulos diferentes identificadas em bancos públicos:")
        for conflito in conflitos_rotulo:
            ids_conflito = [f"{c['id']} ({c['sorotipo']})" for c in conflito]
            print(f"   * {', '.join(ids_conflito)}")
        print("   -> Ação metodológica: Removendo registros conflitantes para evitar ruído de rótulo.")

    # Filtragem e desduplicação
    hashes_conflitantes = set(conflito[0]["hash"] for conflito in conflitos_rotulo)
    registros_filtrados = []
    hashes_processados = set()

    for reg in todos_registros:
        h = reg["hash"]
        if h in hashes_conflitantes:
            continue
        if desduplicar_exatos:
            if h in hashes_processados:
                continue
            hashes_processados.add(h)
        registros_filtrados.append(reg)

    print(f"\nConjunto Curado Final: {len(registros_filtrados)} sequências independentes mantidas.")

    dados_matriz = []
    for reg in registros_filtrados:
        dados_matriz.append([reg["id"]] + reg["features"] + [reg["sorotipo"]])

    colunas = ["ID"] + TRIMEROS_64 + ["Sorotipo"]
    df = pd.DataFrame(dados_matriz, columns=colunas)

    print("\nDistribuição das Classes no Dataset Curado:")
    contagens = df["Sorotipo"].value_counts()
    for sorotipo, count in contagens.items():
        print(f" - {sorotipo}: {count} sequências ({count / len(df) * 100:.1f}%)")

    return df


# ==============================================================================
# 4. ANÁLISE DESCRITIVA DAS CARACTERÍSTICAS
# ==============================================================================


def analisar_distribuicao_trimeros(df):
    """
    Gera análise estatística descritiva das frequências relativas dos trímeros
    em cada sorotipo para caracterização das entradas da rede neural.
    """
    print("\n" + "=" * 78)
    print("ETAPA 2: ANÁLISE DESCRITIVA DAS CARACTERÍSTICAS (TRÍMEROS)")
    print("=" * 78)

    medias_por_classe = df.groupby("Sorotipo")[TRIMEROS_64].mean()

    print("\nTop 5 Trímeros Mais Frequentes por Sorotipo (Média Relativa):")
    for sorotipo in medias_por_classe.index:
        top5 = medias_por_classe.loc[sorotipo].sort_values(ascending=False).head(5)
        top5_str = ", ".join([f"{tri}: {val * 100:.2f}%" for tri, val in top5.items()])
        print(f" - {sorotipo}: {top5_str}")

    print("\nTop 5 Trímeros Menos Frequentes por Sorotipo (Sub-representação):")
    for sorotipo in medias_por_classe.index:
        bottom5 = medias_por_classe.loc[sorotipo].sort_values().head(5)
        bottom5_str = ", ".join([f"{tri}: {val * 100:.2f}%" for tri, val in bottom5.items()])
        print(f" - {sorotipo}: {bottom5_str}")


def avaliar_proximidade_genetica(X_dev, y_dev, X_test, y_test):
    """
    Investiga a proximidade entre as amostras de teste e de desenvolvimento
    no espaço das 64 características (frequências de trímeros) para avaliar a
    independência contínua entre as partições.
    """
    print("\n" + "=" * 78)
    print("INVESTIGAÇÃO DE SIMILARIDADE GENÉTICA ENTRE TREINO E TESTE")
    print("=" * 78)

    distancias_minimas = []
    for i in range(len(X_test)):
        mascara_mesmo_sorotipo = (y_dev == y_test[i])
        if np.any(mascara_mesmo_sorotipo):
            X_mesmo = X_dev[mascara_mesmo_sorotipo]
            dists = np.linalg.norm(X_mesmo - X_test[i], axis=1)
            distancias_minimas.append(np.min(dists))

    d_array = np.array(distancias_minimas)
    print("Distância Euclidiana Mínima no espaço de trímeros (amostra de teste -> par de mesmo sorotipo no treino):")
    print(f" - Mínima observada: {np.min(d_array):.6f}")
    print(f" - Mediana:          {np.median(d_array):.6f}")
    print(f" - Média:            {np.mean(d_array):.6f} (± {np.std(d_array):.6f})")
    print(
        "\nRESSALVA METODOLÓGICA DE INDEPENDÊNCIA: Hashes SHA-256 garantem a eliminação de sequências "
        "100% idênticas (distância zero). Contudo, bancos públicos como NCBI/GenBank contêm isolados de "
        "mesmos surtos que possuem alta identidade (>99.8%). A partição aleatória estratificada não "
        "elimina completamente o parentesco próximo entre linhagens, limitação inerente a dados públicos "
        "sem metadados epidemiológicos ou árvore filogenética completa."
    )


# ==============================================================================
# 5. EXPERIMENTOS E SELEÇÃO DE ARQUITETURAS DO PMC
# ==============================================================================


def calcular_qtd_parametros_pmc(hidden_layers, n_entradas=64, n_saidas=4):
    """Calcula a quantidade total de parâmetros treináveis (pesos + bias) do PMC."""
    camadas = [n_entradas] + list(hidden_layers) + [n_saidas]
    total_params = 0
    for i in range(len(camadas) - 1):
        pesos = camadas[i] * camadas[i + 1]
        bias = camadas[i + 1]
        total_params += (pesos + bias)
    return total_params


def comparar_arquiteturas_pmc(X_dev, y_dev, n_splits=5):
    """
    Executa experimentação sistemática de diferentes configurações do PMC
    utilizando estritamente os dados de desenvolvimento com StratifiedKFold e cross_validate.
    
    Critério de Seleção Rigoroso:
    1. Métrica Primária: Macro F1-Score médio de validação cruzada.
    2. Métrica Secundária (Desempate): Acurácia média.
    3. Princípio da Parcimônia (Navalha de Occam): Se a diferença de Macro F1 entre
       a configuração de maior pontuação e a baseline (ou outra configuração mais compacta)
       for estatisticamente desprezível (<= 0.05%), opta-se pela arquitetura com menor
       número de parâmetros para prevenir sobreparametrização.
       
    Retorna a configuração efetivamente selecionada e seu nome.
    """
    print("\n" + "=" * 78)
    print("ETAPA 3: EXPERIMENTAÇÃO E SELEÇÃO DINÂMICA DE ARQUITETURAS DO PMC (5-FOLD CV)")
    print("=" * 78)
    print("Avaliação de configurações candidatas utilizando exclusivamente dados de desenvolvimento:\n")

    configuracoes = {
        "Baseline (64, 32), alpha=0.001": {
            "hidden_layer_sizes": (64, 32),
            "alpha": 0.001,
            "max_iter": 500,
        },
        "1 Camada (64), alpha=0.001": {
            "hidden_layer_sizes": (64,),
            "alpha": 0.001,
            "max_iter": 500,
        },
        "2 Camadas Largas (128, 64), alpha=0.001": {
            "hidden_layer_sizes": (128, 64),
            "alpha": 0.001,
            "max_iter": 500,
        },
        "3 Camadas Profundas (64, 32, 16), alpha=0.001": {
            "hidden_layer_sizes": (64, 32, 16),
            "alpha": 0.001,
            "max_iter": 500,
        },
        "Baseline (64, 32), alpha=0.01 (Alta Reg.)": {
            "hidden_layer_sizes": (64, 32),
            "alpha": 0.01,
            "max_iter": 500,
        },
        "Baseline (64, 32), alpha=0.0001 (Baixa Reg.)": {
            "hidden_layer_sizes": (64, 32),
            "alpha": 0.0001,
            "max_iter": 500,
        },
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    resultados = []

    for nome_cfg, params in configuracoes.items():
        pmc = MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=params["alpha"],
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=params["max_iter"],
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("pmc", pmc)])

        # Avaliação conjunta e eficiente via cross_validate (uma única passagem das dobras)
        cv_scores = cross_validate(
            pipe, X_dev, y_dev, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1
        )

        media_acc = np.mean(cv_scores["test_accuracy"]) * 100
        std_acc = np.std(cv_scores["test_accuracy"]) * 100
        media_f1 = np.mean(cv_scores["test_f1_macro"]) * 100
        std_f1 = np.std(cv_scores["test_f1_macro"]) * 100
        n_params = calcular_qtd_parametros_pmc(params["hidden_layer_sizes"])

        resultados.append({
            "nome": nome_cfg,
            "params": params,
            "media_f1": media_f1,
            "std_f1": std_f1,
            "media_acc": media_acc,
            "std_acc": std_acc,
            "n_params": n_params,
            "Tabela_Nome": nome_cfg,
            "Tabela_F1": f"{media_f1:.2f} ± {std_f1:.2f}",
            "Tabela_Acc": f"{media_acc:.2f} ± {std_acc:.2f}",
            "Tabela_Params": f"{n_params:,}".replace(",", "."),
        })

    # Tabela formatada
    df_tabela = pd.DataFrame(resultados)[[
        "Tabela_Nome", "Tabela_F1", "Tabela_Acc", "Tabela_Params"
    ]].rename(columns={
        "Tabela_Nome": "Configuração do PMC",
        "Tabela_F1": "Macro F1-Score (%) [Primário]",
        "Tabela_Acc": "Acurácia Média (%) [Secundário]",
        "Tabela_Params": "Qtd. Parâmetros",
    })
    print(df_tabela.to_string(index=False))

    # Ordenação por critério estrito: 1º Macro F1 (desc), 2º Acurácia (desc)
    candidatos_ordenados = sorted(
        resultados, key=lambda c: (c["media_f1"], c["media_acc"]), reverse=True
    )
    topo = candidatos_ordenados[0]

    # Aplicação do Princípio da Parcimônia
    # Se houver empate técnico (diferença <= 0.05% de Macro F1), seleciona a configuração mais compacta
    tolerancia_empate = 0.05
    candidatos_empate = [
        c for c in candidatos_ordenados if (topo["media_f1"] - c["media_f1"]) <= tolerancia_empate
    ]
    # Escolhe a de menor número de parâmetros entre as equivalentes tecnicamente
    vencedor = sorted(candidatos_empate, key=lambda c: c["n_params"])[0]

    print("\n--- Critério de Decisão e Seleção do PMC ---")
    print(f" - Configuração com maior Macro F1 bruto: '{topo['nome']}' ({topo['media_f1']:.2f}%)")
    if vencedor["nome"] != topo["nome"]:
        print(
            f" - Aplicação da Parcimônia: A configuração '{vencedor['nome']}' atingiu desempenho estatisticamente "
            f"equivalente (Δ Macro F1 = {topo['media_f1'] - vencedor['media_f1']:.3f}% <= {tolerancia_empate}%) "
            f"com apenas {vencedor['n_params']:,} parâmetros (contra {topo['n_params']:,})."
        )
    else:
        print(f" - Configuração selecionada: '{vencedor['nome']}'")

    print(f" -> Modelo adotado para o treinamento final: '{vencedor['nome']}'.")
    return vencedor["params"], vencedor["nome"]


# ==============================================================================
# 6. TREINAMENTO, CONVERGÊNCIA E DIAGNÓSTICO
# ==============================================================================


def treinar_pmc_otimizado(X_train, y_train, config):
    """Treina o modelo final nos dados de desenvolvimento."""
    pmc = MLPClassifier(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=config["alpha"],
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=config["max_iter"],
        early_stopping=True,
        validation_fraction=0.15,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    pmc.fit(X_train, y_train)
    return pmc


def plotar_diagnosticos_convergencia(pmc, caminho_perda="curva_perda.png", caminho_val="curva_validacao_interna.png"):
    """
    Gera gráficos separados e rigorosos para a evolução da perda (loss) e para a
    evolução da acurácia de validação interna do early stopping.
    """
    # 1. Gráfico da Perda de Treinamento
    plt.figure(figsize=(7, 4))
    plt.plot(pmc.loss_curve_, color="#1f77b4", lw=2, label="Loss (Treino)")
    plt.title("Evolução da Função de Perda (Loss) do PMC", fontsize=11, fontweight="bold")
    plt.xlabel("Épocas de Treinamento", fontsize=10)
    plt.ylabel("Perda (Cross-Entropy)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_perda, dpi=300)
    plt.close()

    # 2. Gráfico da Acurácia de Validação Interna do Early Stopping
    if hasattr(pmc, "validation_scores_") and pmc.validation_scores_:
        plt.figure(figsize=(7, 4))
        plt.plot(
            pmc.validation_scores_,
            color="#2ca02c",
            lw=2,
            linestyle="-",
            marker="o",
            markersize=3,
            label="Acurácia de Validação Interna",
        )
        plt.title("Critério de Parada: Validação Interna do Early Stopping", fontsize=11, fontweight="bold")
        plt.xlabel("Épocas de Treinamento", fontsize=10)
        plt.ylabel("Acurácia Interna (15% split)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(caminho_val, dpi=300)
        plt.close()

    print(f"\nDiagnóstico de Convergência da Rede:")
    print(f" - Épocas executadas até a parada: {pmc.n_iter_}")
    print(f" - Perda final alcançada: {pmc.loss_:.6f}")
    if hasattr(pmc, "best_loss_") and pmc.best_loss_ is not None:
        print(f" - Melhor perda registrada: {pmc.best_loss_:.6f}")
    print(f" - Gráficos salvos: '{caminho_perda}' e '{caminho_val}'.")


# ==============================================================================
# 7. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ISOLADO
# ==============================================================================


def avaliar_teste_final(pipeline, X_test, y_test, ids_test, classes):
    """
    Avaliação definitiva no conjunto de teste completamente cego.
    Gera métricas completas, matrizes absoluta e normalizada, e tabela de erros.
    """
    print("\n" + "=" * 78)
    print("ETAPA 4: AVALIAÇÃO DEFINITIVA NO CONJUNTO DE TESTE ISOLADO (20% CEGO)")
    print("=" * 78)

    y_pred = pipeline.predict(X_test)
    probas = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    bal_acc = balanced_accuracy_score(y_test, y_pred) * 100

    print(f"\nAcurácia Global no Teste: {acc:.2f}%")
    print(f"Acurácia Balanceada no Teste: {bal_acc:.2f}%")

    print("\nRelatório Detalhado de Classificação:")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))

    # Matriz de Confusão Absoluta e Normalizada
    matriz_abs = confusion_matrix(y_test, y_pred, labels=classes)
    matriz_norm = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.heatmap(
        matriz_abs,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[0],
    )
    axes[0].set_title("Matriz de Confusão Absoluta", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predição do Modelo")
    axes[0].set_ylabel("Classe Real")

    sns.heatmap(
        matriz_norm * 100,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[1],
    )
    axes[1].set_title("Matriz de Confusão Normalizada (%) [Recall]", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predição do Modelo")
    axes[1].set_ylabel("Classe Real")

    plt.tight_layout()
    plt.savefig("matriz_confusao_final.png", dpi=300)
    plt.close()
    print("Matrizes de confusão (absoluta e normalizada) salvas em 'matriz_confusao_final.png'.")

    # Auditoria de Erros
    erros = []
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            prob_dict = {classes[c]: round(probas[i][c] * 100, 2) for c in range(len(classes))}
            erros.append({
                "ID": ids_test[i],
                "Real": y_test[i],
                "Predito": y_pred[i],
                "Confiança (%)": round(np.max(probas[i]) * 100, 2),
                "Probabilidades": prob_dict,
            })

    if erros:
        print(f"\n[AUDITORIA DE ERROS] Total de divergências no teste: {len(erros)} de {len(y_test)}")
        df_erros = pd.DataFrame(erros)
        print(df_erros[["ID", "Real", "Predito", "Confiança (%)"]].to_string(index=False))
        df_erros.to_csv("erros_classificacao_teste.csv", index=False)
        print("Detalhamento dos erros exportado para 'erros_classificacao_teste.csv'.")
    else:
        print("\n[PERFEIÇÃO NO TESTE] Nenhuma divergência observada no conjunto de teste isolado!")

    return probas, y_pred


# ==============================================================================
# 8. CALIBRAÇÃO DE PROBABILIDADES E PERMUTATION IMPORTANCE
# ==============================================================================


def avaliar_calibracao_probabilidades(y_test, probas, classes):
    """
    Avalia a calibração das probabilidades estimadas pelo Softmax do PMC
    utilizando a métrica Brier Score Multiclasse e histograma de confiança.
    """
    print("\n" + "=" * 78)
    print("ETAPA 5: ANÁLISE DE CALIBRAÇÃO DAS PROBABILIDADES DO PMC")
    print("=" * 78)

    lb = LabelBinarizer()
    lb.fit(classes)
    y_test_bin = lb.transform(y_test)

    # Brier Score Multiclasse: Média dos desvios quadráticos entre probabilidade e one-hot
    brier_score_multi = np.mean(np.sum((y_test_bin - probas) ** 2, axis=1))

    print(f"Brier Score Multiclasse: {brier_score_multi:.5f} (Valores próximos a 0 indicam excelente calibração)")
    print(
        "NOTA METODOLÓGICA: Redes neurais multicamadas com função Softmax frequentemente produzem "
        "probabilidades sobreconfiantes devido à saturação dos logtis. O Brier Score quantifica "
        "o desvio entre a incerteza predita e a concordância estocástica observada."
    )

    confiancas = np.max(probas, axis=1) * 100
    plt.figure(figsize=(7, 4))
    plt.hist(confiancas, bins=20, color="#17becf", edgecolor="black", alpha=0.8)
    plt.title("Distribuição da Confiança Máxima nas Predições do Teste", fontsize=11, fontweight="bold")
    plt.xlabel("Confiança Predita (%)", fontsize=10)
    plt.ylabel("Frequência de Amostras", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("distribuicao_confianca.png", dpi=300)
    plt.close()
    print("Histograma de confiança salvo em 'distribuicao_confianca.png'.")


def avaliar_importancia_caracteristicas(pipeline, X_test, y_test, top_n=15):
    """
    Calcula a importância das características através de Permutation Importance
    no conjunto de teste preservado, utilizando perda logarítmica (neg_log_loss)
    para quantificar a sensibilidade probabilística contínua da rede neural.
    """
    print("\n" + "=" * 78)
    print("ETAPA 6: IMPORTÂNCIA DAS CARACTERÍSTICAS (PERMUTATION IMPORTANCE)")
    print("=" * 78)

    # Avaliação contínua por perda logarítmica (neg_log_loss): mede o aumento de incerteza da rede
    perm_loss = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="neg_log_loss"
    )

    importancias_medias = perm_loss.importances_mean
    indices_ordenados = np.argsort(importancias_medias)[::-1][:top_n]

    trimeros_top = [TRIMEROS_64[i] for i in indices_ordenados]
    medias_top = importancias_medias[indices_ordenados]
    stds_top = perm_loss.importances_std[indices_ordenados]

    print(f"\nTop {top_n} Trímeros Mais Determinantes para as Decisões do PMC (Impacto na Perda):")
    for tri, m, s in zip(trimeros_top, medias_top, stds_top):
        print(f" - Trímero '{tri}': Aumento na Perda Logarítmica = +{m:.5f} (± {s:.5f})")

    # Gráfico de barras horizontais
    plt.figure(figsize=(8, 5))
    y_pos = np.arange(top_n)
    plt.barh(y_pos, medias_top[::-1], xerr=stds_top[::-1], align="center", color="#3b528b", ecolor="black", capsize=3)
    plt.yticks(y_pos, trimeros_top[::-1])
    plt.xlabel("Aumento na Perda Logarítmica (Cross-Entropy)", fontsize=10)
    plt.title(f"Top {top_n} Características mais Influentes (Permutation Importance)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("importancia_trimeros.png", dpi=300)
    plt.close()
    print("Gráfico de relevância das características salvo em 'importancia_trimeros.png'.")

    print(
        "\nRESSALVA CIENTÍFICA: A importância computacional reflete o peso e a dependência estatística "
        "atribuídos pela rede neural aos 64 trímeros. Ela NÃO implica causalidade biológica direta "
        "nem substitui análises funcionais de virologia molecular."
    )


# ==============================================================================
# 9. PERSISTÊNCIA INTEGRADA DO PIPELINE E INFERÊNCIA
# ==============================================================================


def salvar_pipeline_completo(pipeline, caminho="pipeline_pmc_denv.joblib"):
    """
    Salva o objeto Pipeline unificado (StandardScaler + MLPClassifier),
    garantindo compatibilidade e prevenindo desajuste de normalização.
    """
    joblib.dump(pipeline, caminho)
    # Mantém compatibilidade com referências anteriores
    joblib.dump(pipeline.named_steps["pmc"], "modelo_pmc_denv.joblib")
    joblib.dump(pipeline.named_steps["scaler"], "scaler_pmc_denv.joblib")
    print(f"\nPipeline unificado salvo com sucesso em '{caminho}'.")


def carregar_pipeline(caminho="pipeline_pmc_denv.joblib"):
    """Carrega o pipeline serializado completo."""
    if os.path.exists(caminho):
        return joblib.load(caminho)
    # Fallback para artefatos separados caso o pipeline ainda não exista
    modelo = joblib.load("modelo_pmc_denv.joblib")
    scaler = joblib.load("scaler_pmc_denv.joblib")
    return Pipeline([("scaler", scaler), ("pmc", modelo)])


def predizer_amostra(seq, pipeline=None):
    """Prediz o sorotipo de uma única sequência e retorna probabilidades."""
    if pipeline is None:
        pipeline = carregar_pipeline()
    valida, motivo = validar_sequencia(seq)
    if not valida:
        raise ValueError(f"Sequência rejeitada: {motivo}")

    features = np.array([extrair_frequencias_trimeros(seq)])
    pred = pipeline.predict(features)[0]
    probas = pipeline.predict_proba(features)[0]
    classes = pipeline.named_steps["pmc"].classes_
    prob_dict = {str(c): float(np.round(p * 100, 2)) for c, p in zip(classes, probas)}
    return str(pred), prob_dict


def predizer_arquivo_fasta(caminho_fasta, pipeline=None, salvar_csv=True):
    """
    Classifica todas as sequências de um arquivo FASTA avulso e apresenta
    tabela detalhada com ID, predição, confiança e probabilidades por classe.
    """
    if pipeline is None:
        pipeline = carregar_pipeline()

    registros, stats = carregar_fasta(caminho_fasta)
    if not registros:
        print(f"\n[AVISO] Nenhuma sequência válida identificada em '{caminho_fasta}'.")
        print(f" - Total de sequências lidas: {stats['total_lidas']}")
        for motivo, qtd in stats["motivos_rejeicao"].items():
            print(f" - Rejeitadas por '{motivo}': {qtd}")
        return []

    print(f"\nClassificando {len(registros)} sequências de '{caminho_fasta}':")
    resultados = []
    classes = list(pipeline.named_steps["pmc"].classes_)

    for r in registros:
        feat = np.array([r["features"]])
        pred = pipeline.predict(feat)[0]
        probas = pipeline.predict_proba(feat)[0]
        prob_dict = {str(c): float(np.round(p * 100, 2)) for c, p in zip(classes, probas)}
        resultados.append({
            "ID": r["id"],
            "Descrição": r["descricao"],
            "Comprimento (pb)": r["comprimento"],
            "Predição": str(pred),
            "Confiança (%)": prob_dict[str(pred)],
            **{f"P({c}) %": prob_dict[str(c)] for c in classes},
        })

    df_preds = pd.DataFrame(resultados)
    colunas_visuais = ["ID", "Predição", "Confiança (%)"] + [f"P({c}) %" for c in classes]
    print(df_preds[colunas_visuais].to_string(index=False))

    if salvar_csv:
        nome_base = os.path.splitext(os.path.basename(caminho_fasta))[0]
        caminho_saida = f"predicoes_{nome_base}.csv"
        df_preds.to_csv(caminho_saida, index=False)
        print(f"\nRelatório de predições exportado para '{caminho_saida}'.")

    return resultados


# ==============================================================================
# 10. FLUXO METODOLÓGICO PRINCIPAL
# ==============================================================================


def main():
    # Modo CLI para predição direta: python PMC.py arquivo.fasta
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        caminho_entrada = sys.argv[1]
        pipe = carregar_pipeline()
        predizer_arquivo_fasta(caminho_entrada, pipe)
        return

    # 1. Curadoria e Desduplicação dos Dados
    df = construir_dataset(ARQUIVOS_SOROTIPOS, max_n_ratio=0.01, desduplicar_exatos=True)

    # 2. Análise Descritiva das Características
    analisar_distribuicao_trimeros(df)

    # 3. Particionamento Estrito: 80% Desenvolvimento / 20% Teste Cego
    X = df[TRIMEROS_64].values
    y = df["Sorotipo"].values
    ids = df["ID"].values
    classes = sorted(list(set(y)))

    X_dev, X_test, y_dev, y_test, ids_dev, ids_test = train_test_split(
        X, y, ids, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nParticionamento dos Dados: Desenvolvimento = {len(X_dev)} (80%), Teste Cego = {len(X_test)} (20%)")

    # 4. Investigação de Proximidade Genética entre Partições
    avaliar_proximidade_genetica(X_dev, y_dev, X_test, y_test)

    # 5. Experimentos e Seleção Dinâmica de Arquitetura do PMC (5-Fold CV nos dados de desenvolvimento)
    melhor_config, nome_config = comparar_arquiteturas_pmc(X_dev, y_dev, n_splits=5)

    # 6. Treinamento da Arquitetura Escolhida nos Dados de Desenvolvimento
    print("\n" + "=" * 78)
    print(f"ETAPA 4: AJUSTE DO MODELO FINAL ('{nome_config}') E DIAGNÓSTICO DE CONVERGÊNCIA")
    print("=" * 78)
    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev)

    pmc = treinar_pmc_otimizado(X_dev_scaled, y_dev, melhor_config)
    plotar_diagnosticos_convergencia(pmc)

    # Montagem do Pipeline Final Integrado
    pipeline = Pipeline([("scaler", scaler), ("pmc", pmc)])

    # 7. Avaliação no Teste Cego Isolado
    X_test_scaled = scaler.transform(X_test)
    probas_test, y_pred_test = avaliar_teste_final(pipeline, X_test, y_test, ids_test, classes)

    # 8. Calibração de Probabilidades e Permutation Importance
    avaliar_calibracao_probabilidades(y_test, probas_test, classes)
    avaliar_importancia_caracteristicas(pipeline, X_test, y_test, top_n=15)

    # 9. Persistência do Pipeline
    salvar_pipeline_completo(pipeline)

    print("\n" + "=" * 78)
    print("PROJETO EXECUTADO COM SUCESSO SEGUINDO RIGOR METODOLÓGICO TOTAL!")
    print("=" * 78)


if __name__ == "__main__":
    main()
