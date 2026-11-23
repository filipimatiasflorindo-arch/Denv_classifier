"""
================================================================================
EXPERIMENTO CIENTÍFICO: REGRESSÃO LOGÍSTICA COM K-MERS DE K=4 A K=6
Análise do Impacto da Dimensionalidade (256, 1.024 e 4.096 features) na
Classificação Linear de Fragmentos Curtos, Amplicons e Reads com Ruído NGS,
em Comparação Direta com o PMC Enxuto V3.0 (LeanMLPDropout).
================================================================================
"""

import os
import sys
import time
import itertools
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Garante que os diretórios raiz e de experimentos estejam no sys.path
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXP_DIR)
for _p in (ROOT_DIR, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PMC import (
    ARQUIVOS_SOROTIPOS,
    DIR_DADOS_TESTES,
    DIR_GRAFICOS,
    DIR_RESULTADOS,
    LeanMLPDropout,
    aplicar_ruido_estocastico,
    carregar_fasta,
    deduplicar_genomas_estrito,
    fatiar_genoma,
)
from analisar_teste_aleatorio import parse_header_organism

BASES = ["A", "C", "G", "T"]


def construir_indice_kmers(k: int):
    """Gera o dicionário de índices canônicos para 4^k oligômeros."""
    kmers = ["".join(p) for p in itertools.product(BASES, repeat=k)]
    k_idx = {km: i for i, km in enumerate(kmers)}
    return k_idx, len(kmers)


def extrair_vetor_kmer(seq: str, k: int, k_idx: dict, num_kmers: int) -> np.ndarray:
    """Extrai o vetor de frequências relativas de k-mers normalizado (soma = 1.0)."""
    seq_clean = seq.upper().replace("U", "T")
    total = len(seq_clean) - k + 1
    if total <= 0:
        return np.zeros(num_kmers, dtype=np.float32)
    counts = Counter(seq_clean[i : i + k] for i in range(total))
    vec = np.zeros(num_kmers, dtype=np.float32)
    val = 0
    for km, c in counts.items():
        idx = k_idx.get(km)
        if idx is not None:
            vec[idx] = c
            val += c
    if val > 0:
        vec /= val
    return vec


def calcular_brier_multiclasse(y_true: np.ndarray, probas: np.ndarray, classes: list) -> float:
    """Calcula o Brier Score multiclasse para avaliar calibração probabilística."""
    lb = LabelBinarizer()
    lb.fit(classes)
    y_bin = lb.transform(y_true)
    return float(np.mean(np.sum((y_bin - probas) ** 2, axis=1)))


def main():
    print("=" * 90)
    print("EXPERIMENTO DE ABLAÇÃO: REGRESSÃO LOGÍSTICA COM K = 4 (256), K = 5 (1.024) E K = 6 (4.096)")
    print("Avaliação em Janelas de 2.000 pb, Sequências Reais Heterogêneas e Leituras NGS com Ruído")
    print("=" * 90)

    # 1. Carregar e deduplicar genomas completos de treino
    print("\n[1/4] Carregando genomas de referência e realizando split a nível de genoma (80/20)...")
    todos_genomas = []
    for arq, sorotipo in ARQUIVOS_SOROTIPOS.items():
        registros, _ = carregar_fasta(arq, sorotipo)
        todos_genomas.extend(registros)

    unicos, _, _ = deduplicar_genomas_estrito(todos_genomas)

    y_gen = [g["sorotipo"] for g in unicos]
    train_gen, test_gen = train_test_split(unicos, test_size=0.20, random_state=42, stratify=y_gen)

    frags_train = []
    for g in train_gen:
        frags_train.extend(fatiar_genoma(g["seq"], g["id"], g["sorotipo"], 2000, 1000))

    frags_test = []
    for g in test_gen:
        frags_test.extend(fatiar_genoma(g["seq"], g["id"], g["sorotipo"], 2000, 1000))

    y_tr = np.array([f["sorotipo"] for f in frags_train])
    y_te = np.array([f["sorotipo"] for f in frags_test])
    print(f"Genomas Únicos: {len(unicos):,} (Treino: {len(train_gen):,} | Teste Cego: {len(test_gen):,})")
    print(f"Fragmentos de 2.000 pb: Treino = {len(frags_train):,} | Teste Cego = {len(frags_test):,}")

    # 2. Carregar conjunto externo heterogêneo (teste_aleatorio.fasta)
    print("\n[2/4] Carregando banco externo heterogêneo ('dados/testes/teste_aleatorio.fasta')...")
    caminho_fasta = os.path.join(DIR_DADOS_TESTES, "teste_aleatorio.fasta")
    aleatorio_recs = []
    with open(caminho_fasta, "r", encoding="utf-8", errors="ignore") as f:
        cur_desc, cur_seq = "", []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_desc:
                    aleatorio_recs.append((cur_desc, "".join(cur_seq)))
                    cur_seq = []
                cur_desc = line[1:]
            else:
                cur_seq.append(line)
        if cur_desc:
            aleatorio_recs.append((cur_desc, "".join(cur_seq)))

    classes = ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]
    aleatorio_validos = []
    for desc, s in aleatorio_recs:
        lbl, is_pat = parse_header_organism(desc)
        if lbl in classes and not is_pat and len(s) >= 6:
            aleatorio_validos.append((lbl, len(s), s))

    y_aleat = np.array([lbl for lbl, _, _ in aleatorio_validos])
    lens_aleat = np.array([l for _, l, _ in aleatorio_validos])
    seqs_limpas = [s for _, _, s in aleatorio_validos]

    # Gerar versões com ruído NGS (4% e 10%) em fragmentos curtos (< 1.000 pb)
    seqs_ruido_4 = [
        aplicar_ruido_estocastico(s, taxa_erro=0.04, random_state=42 + i) if comp < 1000 else s
        for i, (_, comp, s) in enumerate(aleatorio_validos)
    ]
    seqs_ruido_10 = [
        aplicar_ruido_estocastico(s, taxa_erro=0.10, random_state=1042 + i) if comp < 1000 else s
        for i, (_, comp, s) in enumerate(aleatorio_validos)
    ]

    mask_curto = lens_aleat < 300
    mask_amplicon = (lens_aleat >= 300) & (lens_aleat < 1000)
    mask_gene = (lens_aleat >= 1000) & (lens_aleat < 5000)
    mask_genoma = lens_aleat >= 5000

    print(
        f"Sequências validadas: {len(aleatorio_validos):,} | "
        f"<300pb: {np.sum(mask_curto)} | 300-1000pb: {np.sum(mask_amplicon)} | "
        f"1000-5000pb: {np.sum(mask_gene)} | >=5000pb: {np.sum(mask_genoma)}"
    )

    # 3. Loop experimental para K = 4, 5 e 6
    print("\n[3/4] Executando extração e treinamento para K = 4, 5 e 6...")
    resultados = []

    for k in [4, 5, 6]:
        k_idx, num_kmers = construir_indice_kmers(k)
        print("\n" + "-" * 90)
        print(f">> AVALIANDO K = {k} ({num_kmers:,} características canônicas | 4^{k})")
        print("-" * 90)

        t0 = time.time()
        X_tr = np.array([extrair_vetor_kmer(f["seq"], k, k_idx, num_kmers) for f in frags_train])
        X_te = np.array([extrair_vetor_kmer(f["seq"], k, k_idx, num_kmers) for f in frags_test])
        X_al_limpo = np.array([extrair_vetor_kmer(s, k, k_idx, num_kmers) for s in seqs_limpas])
        X_al_r4 = np.array([extrair_vetor_kmer(s, k, k_idx, num_kmers) for s in seqs_ruido_4])
        X_al_r10 = np.array([extrair_vetor_kmer(s, k, k_idx, num_kmers) for s in seqs_ruido_10])
        t_extracao = time.time() - t0

        esp_2000 = float(np.mean(X_tr == 0) * 100)
        esp_amp = float(np.mean(X_al_limpo[mask_amplicon] == 0) * 100)
        esp_curto = float(np.mean(X_al_limpo[mask_curto] == 0) * 100)

        print(
            f"   Extração concluída em {t_extracao:.1f}s | Esparsidade (% zeros): "
            f"2000pb={esp_2000:.1f}% | 300-1000pb={esp_amp:.1f}% | <300pb={esp_curto:.1f}%"
        )

        # Testamos Regressão Logística (L2 padrão C=1.0 e L2 forte C=0.1) e PMC Enxuto (32,) + Dropout
        modelos_k = [
            (
                "Regressão Logística (L2, C=1.0)",
                num_kmers * 4 + 4,
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=300, C=1.0, random_state=42, solver="lbfgs")),
                ]),
            ),
            (
                "Regressão Logística (L2 Forte, C=0.1)",
                num_kmers * 4 + 4,
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=300, C=0.1, random_state=42, solver="lbfgs")),
                ]),
            ),
            (
                "PMC Enxuto (32,) + Dropout (p=0.3)",
                (num_kmers * 32 + 32) + (32 * 4 + 4),
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LeanMLPDropout(hidden_layers=(32,), dropout_rate=0.3, alpha=0.01, max_iter=25, random_state=42)),
                ]),
            ),
        ]

        for nome_modelo, n_params, pipe in modelos_k:
            t_ini = time.time()
            pipe.fit(X_tr, y_tr)
            t_treino = time.time() - t_ini

            # Avaliação no Teste Cego (2.000 pb)
            preds_te = pipe.predict(X_te)
            probas_te = pipe.predict_proba(X_te)
            acc_te = accuracy_score(y_te, preds_te) * 100
            f1_te = f1_score(y_te, preds_te, average="macro") * 100
            brier_te = calcular_brier_multiclasse(y_te, probas_te, classes)

            # Avaliação no Teste Aleatório LIMPO
            p_limpo = pipe.predict(X_al_limpo)
            acc_glob_limpo = accuracy_score(y_aleat, p_limpo) * 100
            f1_glob_limpo = f1_score(y_aleat, p_limpo, average="macro") * 100
            acc_curto_limpo = accuracy_score(y_aleat[mask_curto], p_limpo[mask_curto]) * 100
            acc_amp_limpo = accuracy_score(y_aleat[mask_amplicon], p_limpo[mask_amplicon]) * 100
            acc_gene_limpo = accuracy_score(y_aleat[mask_gene], p_limpo[mask_gene]) * 100
            acc_genoma_limpo = accuracy_score(y_aleat[mask_genoma], p_limpo[mask_genoma]) * 100

            # Avaliação com Ruído NGS 4%
            p_r4 = pipe.predict(X_al_r4)
            acc_glob_r4 = accuracy_score(y_aleat, p_r4) * 100
            acc_curto_r4 = accuracy_score(y_aleat[mask_curto], p_r4[mask_curto]) * 100
            acc_amp_r4 = accuracy_score(y_aleat[mask_amplicon], p_r4[mask_amplicon]) * 100

            # Avaliação com Ruído NGS 10%
            p_r10 = pipe.predict(X_al_r10)
            acc_glob_r10 = accuracy_score(y_aleat, p_r10) * 100
            acc_curto_r10 = accuracy_score(y_aleat[mask_curto], p_r10[mask_curto]) * 100
            acc_amp_r10 = accuracy_score(y_aleat[mask_amplicon], p_r10[mask_amplicon]) * 100

            reg = {
                "K": k,
                "Features (4^K)": num_kmers,
                "Modelo": nome_modelo,
                "Parametros": n_params,
                "Tempo Treino (s)": round(t_treino, 2),
                "Esparsidade 2000pb (%)": round(esp_2000, 1),
                "Esparsidade <300pb (%)": round(esp_curto, 1),
                "Teste Cego 2000pb (%)": round(acc_te, 2),
                "Brier Score (2000pb)": round(brier_te, 5),
                "Limpo Global (%)": round(acc_glob_limpo, 2),
                "Limpo Macro-F1 (%)": round(f1_glob_limpo, 2),
                "Limpo <300pb (%)": round(acc_curto_limpo, 2),
                "Limpo 300-1000pb (%)": round(acc_amp_limpo, 2),
                "Limpo 1000-5000pb (%)": round(acc_gene_limpo, 2),
                "Limpo >=5000pb (%)": round(acc_genoma_limpo, 2),
                "Ruido 4% Global (%)": round(acc_glob_r4, 2),
                "Ruido 4% <300pb (%)": round(acc_curto_r4, 2),
                "Ruido 4% 300-1000pb (%)": round(acc_amp_r4, 2),
                "Ruido 10% Global (%)": round(acc_glob_r10, 2),
                "Ruido 10% <300pb (%)": round(acc_curto_r10, 2),
                "Ruido 10% 300-1000pb (%)": round(acc_amp_r10, 2),
            }
            resultados.append(reg)

            print(
                f"   [{nome_modelo:<37}] ({t_treino:4.1f}s) | "
                f"2000pb: {acc_te:6.2f}% | Global: {acc_glob_limpo:6.2f}% | "
                f"<300pb: {acc_curto_limpo:5.2f}% (R4%: {acc_curto_r4:5.2f}%) | "
                f"300-1000pb: {acc_amp_limpo:5.2f}% (R4%: {acc_amp_r4:5.2f}% | R10%: {acc_amp_r10:5.2f}%)"
            )

    # 4. Exportação de CSV e Geração de Gráfico Científico
    print("\n[4/4] Salvando resultados tabulares e gerando painel gráfico comparativo...")
    df_res = pd.DataFrame(resultados)
    dir_bench = os.path.join(DIR_RESULTADOS, "benchmarks")
    os.makedirs(dir_bench, exist_ok=True)
    caminho_csv = os.path.join(dir_bench, "benchmark_regressao_logistica_k4_k6.csv")
    df_res.to_csv(caminho_csv, index=False)
    print(f"Tabela completa salva em '{caminho_csv}'.")

    # Gerar Gráfico Comparativo de Alta Resolução
    os.makedirs(DIR_GRAFICOS, exist_ok=True)
    caminho_png = os.path.join(DIR_GRAFICOS, "comparativo_regressao_logistica_k4_k6.png")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    ks_labels = ["K=4\n(256)", "K=5\n(1.024)", "K=6\n(4.096)"]
    x_pos = np.arange(len(ks_labels))

    estilos = {
        "Regressão Logística (L2, C=1.0)": ("#d62728", "o", "--"),
        "Regressão Logística (L2 Forte, C=0.1)": ("#ff7f0e", "^", "-."),
        "PMC Enxuto (32,) + Dropout (p=0.3)": ("#1f77b4", "s", "-"),
    }

    # Painel 1: Acurácia em Reads Ultra-Curtas (<300 pb) - Limpo
    for nome_m, (cor, marker, ls) in estilos.items():
        sub = df_res[df_res["Modelo"] == nome_m].sort_values("K")
        axes[0].plot(x_pos, sub["Limpo <300pb (%)"], color=cor, marker=marker, linestyle=ls, linewidth=2.3, markersize=8, label=nome_m)
    axes[0].set_title("A) Reads Ultra-Curtas (< 300 pb) - Sem Ruído", fontsize=11.5, fontweight="bold")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(ks_labels, fontweight="bold")
    axes[0].set_ylabel("Acurácia (%)", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8.5, loc="lower right")

    # Painel 2: Acurácia em Amplicons (300 - 1.000 pb) - Limpo vs Ruído 4%
    for nome_m, (cor, marker, ls) in estilos.items():
        sub = df_res[df_res["Modelo"] == nome_m].sort_values("K")
        axes[1].plot(x_pos, sub["Limpo 300-1000pb (%)"], color=cor, marker=marker, linestyle=ls, linewidth=2.3, markersize=8, label=f"{nome_m[:19]} (Limpo)")
        axes[1].plot(x_pos, sub["Ruido 4% 300-1000pb (%)"], color=cor, marker=marker, linestyle=":", linewidth=1.8, markersize=6, alpha=0.75, label=f"{nome_m[:19]} (Ruído 4%)")
    axes[1].set_title("B) Amplicons (300 - 1.000 pb): Limpo vs Ruído 4%", fontsize=11.5, fontweight="bold")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(ks_labels, fontweight="bold")
    axes[1].set_ylabel("Acurácia (%)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=7.5, ncol=1, loc="lower right")

    # Painel 3: Amplicons sob Ruído Extremo (10%) em função de K
    for nome_m, (cor, marker, ls) in estilos.items():
        sub = df_res[df_res["Modelo"] == nome_m].sort_values("K")
        axes[2].plot(x_pos, sub["Ruido 10% 300-1000pb (%)"], color=cor, marker=marker, linestyle=ls, linewidth=2.3, markersize=8, label=nome_m)
    axes[2].set_title("C) Amplicons (300 - 1.000 pb) sob Ruído NGS de 10%", fontsize=11.5, fontweight="bold")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(ks_labels, fontweight="bold")
    axes[2].set_ylabel("Acurácia (%)", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=8.5, loc="lower right")

    plt.suptitle("Impacto da Ordem do K-mer (K=4, 5, 6) na Regressão Logística vs PMC Enxuto V3.0", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico comparativo salvo em '{caminho_png}'.")

    print("\n" + "=" * 90)
    print("RESUMO CONSOLIDADO: REGRESSÃO LOGÍSTICA E PMC DE K=4 A K=6")
    print("=" * 90)
    cols_resumo = [
        "K",
        "Features (4^K)",
        "Modelo",
        "Teste Cego 2000pb (%)",
        "Limpo Global (%)",
        "Limpo <300pb (%)",
        "Limpo 300-1000pb (%)",
        "Ruido 4% <300pb (%)",
        "Ruido 4% 300-1000pb (%)",
        "Ruido 10% 300-1000pb (%)",
    ]
    print(df_res[cols_resumo].to_string(index=False))


if __name__ == "__main__":
    main()
