"""
================================================================================
GERADOR DE GRÁFICOS CIENTÍFICOS EM ALTA RESOLUÇÃO (600 DPI - PADRÃO SBC/IEEE)
Remove versões desatualizadas em graficos/ (mantendo graficos/legado_v1/) e
regera todos os 7 painéis oficiais do projeto diretamente a partir dos CSVs
unificados (N=3.109 genomas) e do pipeline V3.0 serializado.
================================================================================
"""

import hashlib
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

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
    K_MER_SIZE,
    carregar_fasta,
    carregar_pipeline,
    deduplicar_genomas_estrito,
    extrair_frequencias_kmers,
    fatiar_genoma,
)
from avaliar_validacao_externa import parse_header_organism
from experimento_validacao_final_sbc import calcular_brier, calcular_ece_e_bins, ler_fasta_generico

DPI_ALTA = 600
DIR_BENCH = os.path.join(DIR_RESULTADOS, "benchmarks")
CLASSES = ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 14.5,
    "legend.framealpha": 0.95,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
})


def _salvar_figura(caminho: str, indice: int):
    """Salva figura em 600 DPI com fallback automático caso o arquivo esteja bloqueado pelo Windows."""
    try:
        plt.savefig(caminho, dpi=DPI_ALTA, bbox_inches="tight")
        print(f"  [{indice}/7] Salvo em 600 DPI: {caminho}")
    except (PermissionError, OSError):
        base, ext = os.path.splitext(caminho)
        alt = f"{base}_600dpi{ext}"
        plt.savefig(alt, dpi=DPI_ALTA, bbox_inches="tight")
        print(f"  [{indice}/7] Arquivo original em uso; salvo em 600 DPI como: {alt}")
    finally:
        plt.close()


def remover_graficos_antigos():
    """Remove os PNGs antigos na raiz de graficos/, preservando graficos/legado_v1/."""
    removidos = []
    if os.path.exists(DIR_GRAFICOS):
        for nome in os.listdir(DIR_GRAFICOS):
            caminho = os.path.join(DIR_GRAFICOS, nome)
            if os.path.isfile(caminho) and nome.lower().endswith(".png"):
                try:
                    os.remove(caminho)
                    removidos.append(nome)
                except PermissionError:
                    print(f"  [AVISO] '{nome}' em uso; será sobrescrito diretamente ao salvar.")
    print(f"Gráficos antigos removidos ({len(removidos)}): {removidos}")


def gerar_1_matriz_confusao(pipe_pmc, X_te_no, y_te_no):
    """1. Matriz de Confusão Absoluta e Normalizada (PMC Enxuto, Teste Cego Não-Sobreposto 2.000 pb)."""
    y_pred = pipe_pmc.predict(X_te_no)
    m_abs = confusion_matrix(y_te_no, y_pred, labels=CLASSES)
    m_norm = confusion_matrix(y_te_no, y_pred, labels=CLASSES, normalize="true") * 100

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2))
    sns.heatmap(m_abs, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0], annot_kws={"size": 16, "weight": "bold"})
    axes[0].set_title("A) Matriz Absoluta (2.000 pb, N=3.110)", fontsize=16, fontweight="bold", pad=12)
    axes[0].set_xlabel("Sorotipo Predito (PMC Enxuto)", fontsize=15, fontweight="bold")
    axes[0].set_ylabel("Sorotipo Real", fontsize=15, fontweight="bold")
    axes[0].tick_params(labelsize=13.5)

    sns.heatmap(m_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1], annot_kws={"size": 16, "weight": "bold"})
    axes[1].set_title("B) Matriz Normalizada (%) [Recall]", fontsize=16, fontweight="bold", pad=12)
    axes[1].set_xlabel("Sorotipo Predito (PMC Enxuto)", fontsize=15, fontweight="bold")
    axes[1].set_ylabel("Sorotipo Real", fontsize=15, fontweight="bold")
    axes[1].tick_params(labelsize=13.5)

    plt.suptitle("PMC Enxuto (32,) — Teste Cego Não-Sobreposto (Acurácia = 99,90%)", fontsize=17.5, fontweight="bold", y=1.02)
    plt.tight_layout(w_pad=2.5)
    _salvar_figura(os.path.join(DIR_GRAFICOS, "matriz_confusao_fragmentos.png"), 1)


def gerar_2_avaliacao_teste_aleatorio(pipe_pmc, df_all, df_ineditas):
    """2. Painel Estratificado da Validação Externa (Todas N=1.846 vs Inéditas N=1.595, right=False)."""
    labels_faixa = ["< 300 pb\n(Reads)", "300–999 pb\n(Amplicons)", "1.000–4.999 pb\n(Genes)", "≥ 5.000 pb\n(Genomas)"]
    chaves = ["<300", "300-1000", "1000-5000", ">=5000"]

    acc_all, acc_in, conf_all = [], [], []
    for ch in chaves:
        s1 = df_all[df_all["Faixa"] == ch]
        s2 = df_ineditas[df_ineditas["Faixa"] == ch]
        acc_all.append(accuracy_score(s1["Real"], s1["Pred"]) * 100)
        acc_in.append(accuracy_score(s2["Real"], s2["Pred"]) * 100)
        conf_all.append(s1["Conf"].mean())

    x = np.arange(len(labels_faixa))
    w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.8))

    b1 = axes[0].bar(x - w / 2, acc_all, width=w, color="#1f77b4", label="Todas Validadas (N=1.846 | 97,40%)")
    b2 = axes[0].bar(x + w / 2, acc_in, width=w, color="#2ca02c", label="Estritamente Inéditas (N=1.595 | 97,55%)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_faixa, fontsize=13.5, fontweight="bold")
    axes[0].set_ylabel("Acurácia (%)", fontsize=15, fontweight="bold")
    axes[0].set_ylim(70, 107)
    axes[0].set_title("A) Acurácia por Faixa de Comprimento (PMC Enxuto)", fontsize=15.5, fontweight="bold", pad=12)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=14)

    ns_in = [int(np.sum(df_ineditas["Faixa"] == ch)) for ch in chaves]
    bars_c = axes[1].bar(x, conf_all, width=0.52, color=["#ff7f0e", "#9467bd", "#17becf", "#1f77b4"], alpha=0.9)
    for bar, n_i in zip(bars_c, ns_in):
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.35, f"{h:.2f}%\n(N={n_i})", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_faixa, fontsize=13.5, fontweight="bold")
    axes[1].set_ylabel("Confiança Média Softmax (%)", fontsize=15, fontweight="bold")
    axes[1].set_ylim(88, 105)
    axes[1].set_title("B) Confiança Média e Volume Inédito por Faixa", fontsize=15.5, fontweight="bold", pad=12)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Validação Externa Heterogênea — PMC Enxuto (32,)", fontsize=17.5, fontweight="bold", y=1.02)
    plt.tight_layout(w_pad=2.6)
    _salvar_figura(os.path.join(DIR_GRAFICOS, "avaliacao_teste_aleatorio.png"), 2)


def gerar_3_curva_degradacao_ruido_5sementes():
    """3. Curva de Degradação sob Ruído NGS (0% a 15%) com Média ± DP (5 sementes) em Amplicons e Reads Inéditas."""
    df = pd.read_csv(os.path.join(DIR_BENCH, "validacao_sbc_ruido_5sementes.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.8))

    estilos = {
        "PMC Enxuto V3.0 (32,) + Dropout": ("#1f77b4", "s", "-", "PMC Enxuto (32,) + Dropout"),
        "Regressão Logística (L2, C=1.0)": ("#d62728", "o", "--", "Reg. Logística Multinomial (L2)"),
        "PMC Grande (512, 128)": ("#2ca02c", "^", "-.", "PMC Grande (512, 128)"),
    }

    for ax, faixa, titulo in [
        (axes[0], "Amplicons Inéditos (300-1.000 pb)", "A) Amplicons Inéditos (300–999 pb, N=201)"),
        (axes[1], "Reads Inéditas (< 300 pb)", "B) Reads Inéditas (< 300 pb, N=81)"),
    ]:
        sub_f = df[df["Faixa"] == faixa]
        for mod, (cor, mk, ls, lbl) in estilos.items():
            s = sub_f[sub_f["Modelo"] == mod].sort_values("Ruido (%)")
            ax.errorbar(
                s["Ruido (%)"],
                s["Acc Media (%)"],
                yerr=s["Acc DP (%)"],
                color=cor,
                marker=mk,
                linestyle=ls,
                linewidth=3.2,
                markersize=10,
                capsize=5,
                elinewidth=2.2,
                label=lbl,
            )
        ax.set_title(titulo, fontsize=15.5, fontweight="bold", pad=12)
        ax.set_xlabel("Taxa de Ruído Estocástico NGS (%)", fontsize=15, fontweight="bold")
        ax.set_ylabel("Acurácia (% — Média ± DP, 5 sem.)", fontsize=15, fontweight="bold")
        ax.set_xticks([0, 4, 8, 10, 15])
        ax.tick_params(labelsize=13.5)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", fontsize=14.5)

    plt.suptitle("Resiliência a Ruído NGS (Substituições + Indels) nas Faixas Curtas Inéditas", fontsize=17.5, fontweight="bold", y=1.02)
    plt.tight_layout(w_pad=2.6)
    _salvar_figura(os.path.join(DIR_GRAFICOS, "curva_degradacao_ruido_extremo.png"), 3)


def gerar_4_ablacao_k4_k6():
    """4. Ablação K=4, 5 e 6 (Regressão Logística Multinomial vs PMC Enxuto)."""
    df = pd.read_csv(os.path.join(DIR_BENCH, "benchmark_regressao_logistica_k4_k6.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 8.0))
    ks_labels = ["K=4\n(256)", "K=5\n(1.024)", "K=6\n(4.096)"]
    x_pos = np.arange(len(ks_labels))

    estilos = {
        "Regressão Logística (L2, C=1.0)": ("#d62728", "o", "--", "Reg. Logística (C=1.0)"),
        "Regressão Logística (L2 Forte, C=0.1)": ("#ff7f0e", "^", "-.", "Reg. Logística (C=0.1)"),
        "PMC Enxuto (32,) + Dropout (p=0.3)": ("#1f77b4", "s", "-", "PMC Enxuto (32,) + Dropout"),
    }

    for nome_m, (cor, mk, ls, lbl) in estilos.items():
        sub = df[df["Modelo"] == nome_m].sort_values("K")
        axes[0].plot(x_pos, sub["Limpo <300pb (%)"], color=cor, marker=mk, linestyle=ls, linewidth=3.2, markersize=10.5, label=lbl)
    axes[0].set_title("A) Reads (< 300 pb) — Limpo", fontsize=15.5, fontweight="bold", pad=12)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(ks_labels, fontsize=13.5, fontweight="bold")
    axes[0].set_ylabel("Acurácia (%)", fontsize=15, fontweight="bold")
    axes[0].tick_params(labelsize=13)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=13.5, loc="lower right")

    for nome_m, (cor, mk, ls, lbl) in estilos.items():
        sub = df[df["Modelo"] == nome_m].sort_values("K")
        curto = "PMC Enxuto" if "PMC" in lbl else ("RL (C=1.0)" if "1.0" in lbl else "RL (C=0.1)")
        axes[1].plot(x_pos, sub["Limpo 300-1000pb (%)"], color=cor, marker=mk, linestyle=ls, linewidth=3.2, markersize=10.5, label=f"{curto} (0%)")
        axes[1].plot(x_pos, sub["Ruido 4% 300-1000pb (%)"], color=cor, marker=mk, linestyle=":", linewidth=2.5, markersize=8.5, alpha=0.8, label=f"{curto} (4%)")
    axes[1].set_title("B) Amplicons: Limpo vs Ruído 4%", fontsize=15.5, fontweight="bold", pad=12)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(ks_labels, fontsize=13.5, fontweight="bold")
    axes[1].set_ylabel("Acurácia (%)", fontsize=15, fontweight="bold")
    axes[1].tick_params(labelsize=13)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=12.8, loc="lower right")

    for nome_m, (cor, mk, ls, lbl) in estilos.items():
        sub = df[df["Modelo"] == nome_m].sort_values("K")
        axes[2].plot(x_pos, sub["Ruido 10% 300-1000pb (%)"], color=cor, marker=mk, linestyle=ls, linewidth=3.2, markersize=10.5, label=lbl)
    axes[2].set_title("C) Amplicons sob Ruído 10%", fontsize=15.5, fontweight="bold", pad=12)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(ks_labels, fontsize=13.5, fontweight="bold")
    axes[2].set_ylabel("Acurácia (%)", fontsize=15, fontweight="bold")
    axes[2].tick_params(labelsize=13)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=13.5, loc="lower right")

    plt.suptitle("Impacto da Ordem do K-mer (K=4, 5, 6): Regressão Logística vs PMC Enxuto (32,)", fontsize=17.5, fontweight="bold", y=1.02)
    plt.tight_layout(w_pad=2.2)
    _salvar_figura(os.path.join(DIR_GRAFICOS, "comparativo_regressao_logistica_k4_k6.png"), 4)


def gerar_5_comparativo_rbf():
    """5. Rede Neural RBF (Moody-Darken) & Kernel RBF-LR (Nyström) vs PMC Enxuto vs Regressão Logística."""
    df = pd.read_csv(os.path.join(DIR_BENCH, "benchmark_rbf_vs_pmc_rl.csv"))
    df_k6 = df[df["K"] == 6].copy()
    df_k6["Nome Curto"] = df_k6["Nome Curto"].replace({"PMC V3.0 (32,)": "PMC Enxuto (32,)"})

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.2))
    nomes = df_k6["Nome Curto"].tolist()
    y_pos = np.arange(len(nomes))
    h = 0.26

    axes[0].barh(y_pos + h, df_k6["Teste Cego 2000pb (%)"], height=h, label="Janelas 2.000 pb (Teste Cego)", color="#2ca02c", alpha=0.88)
    axes[0].barh(y_pos, df_k6["Limpo 300-1000pb (%)"], height=h, label="Amplicons (300–999 pb)", color="#1f77b4", alpha=0.9)
    axes[0].barh(y_pos - h, df_k6["Limpo <300pb (%)"], height=h, label="Reads Curtas (< 300 pb)", color="#ff7f0e", alpha=0.9)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(nomes, fontweight="bold", fontsize=14)
    axes[0].set_xlabel("Acurácia (%)", fontsize=15, fontweight="bold")
    axes[0].set_title("A) Desempenho em K=6 por Escala Genômica", fontweight="bold", fontsize=15.5, pad=12)
    axes[0].set_xlim(0, 105)
    axes[0].tick_params(axis="x", labelsize=13.5)
    axes[0].grid(True, axis="x", linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=14)

    axes[1].barh(y_pos + h, df_k6["Limpo 300-1000pb (%)"], height=h, label="0% Ruído (Limpo)", color="#1f77b4", alpha=0.9)
    axes[1].barh(y_pos, df_k6["Ruido 4% 300-1000pb (%)"], height=h, label="4% Ruído NGS", color="#9467bd", alpha=0.9)
    axes[1].barh(y_pos - h, df_k6["Ruido 10% 300-1000pb (%)"], height=h, label="10% Ruído Severo", color="#d62728", alpha=0.9)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(nomes, fontweight="bold", fontsize=14)
    axes[1].set_xlabel("Acurácia em Amplicons 300–999 pb (%)", fontsize=15, fontweight="bold")
    axes[1].set_title("B) Resiliência a Ruído NGS em Amplicons (K=6)", fontweight="bold", fontsize=15.5, pad=12)
    axes[1].set_xlim(0, 102)
    axes[1].tick_params(axis="x", labelsize=13.5)
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right", fontsize=14)

    plt.suptitle("Paradigmas Neurais em K=6: Rede RBF (Moody-Darken) & Kernel RBF-LR vs PMC Enxuto", fontsize=17, fontweight="bold", y=1.02)
    plt.tight_layout(w_pad=3.0)
    _salvar_figura(os.path.join(DIR_GRAFICOS, "comparativo_rbf_vs_pmc_rl.png"), 5)


def gerar_6_segment_conv1d_posicao():
    """6. Segment-wise Conv1D 1x1 (Posicional vs Global) e Varredura Posicional nas 10 Regiões do DENV."""
    df_mod = pd.read_csv(os.path.join(DIR_BENCH, "benchmark_cnn1d_comparativo.csv"))
    df_pos = pd.read_csv(os.path.join(DIR_BENCH, "benchmark_posicao_genomica_cnn1d.csv"))

    renomear = {
        "CNN 1D Posicional": "Conv1D 1×1 Posicional",
        "CNN 1D Global": "Conv1D 1×1 Global",
        "PMC V3.0 (32,)": "PMC Enxuto (32,)",
        "Reg. Logística": "Reg. Logística",
    }
    df_mod["Label"] = df_mod["Nome Curto"].map(lambda s: renomear.get(s, s))

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 8.2))
    x_genes = np.arange(len(df_pos))
    labels_genes = df_pos["Regiao Genômica"].tolist()

    axes[0].plot(x_genes, df_pos["PMC V3.0 (Ruído 4% %)"], marker="s", color="#1f77b4", linewidth=3.2, markersize=9.5, label="PMC Enxuto (32,) + Dropout")
    axes[0].plot(x_genes, df_pos["CNN 1D Posicional (Ruído 4% %)"], marker="^", color="#9467bd", linewidth=2.8, markersize=9.5, linestyle="-.", label="Conv1D 1×1 Posicional (5'→3')")
    axes[0].plot(x_genes, df_pos["CNN 1D Global (Ruído 4% %)"], marker="o", color="#2ca02c", linewidth=2.8, markersize=8.5, linestyle="--", label="Conv1D 1×1 Global")
    axes[0].plot(x_genes, df_pos["Reg. Logística (Ruído 4% %)"], marker="d", color="#d62728", linewidth=2.8, markersize=8.5, linestyle=":", label="Reg. Logística (L2)")

    axes[0].set_title("A) Posição Genômica (Amplicons 500 pb, Ruído 4%)", fontsize=15.5, fontweight="bold", pad=12)
    axes[0].set_xticks(x_genes)
    axes[0].set_xticklabels(labels_genes, rotation=35, ha="right", fontweight="bold", fontsize=12)
    axes[0].set_ylabel("Acurácia por Região Gênica (%)", fontsize=15, fontweight="bold")
    axes[0].set_ylim(78, 101.8)
    axes[0].tick_params(axis="y", labelsize=13.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower left", fontsize=13.8)

    nomes_m = df_mod["Label"].tolist()
    y_pos = np.arange(len(nomes_m))
    h = 0.25

    axes[1].barh(y_pos + h, df_mod["Limpo 300-1000pb (%)"], height=h, color="#1f77b4", alpha=0.9, label="Amplicons Limpos (0%)")
    axes[1].barh(y_pos, df_mod["Ruido 4% 300-1000pb (%)"], height=h, color="#2ca02c", alpha=0.9, label="Amplicons Ruído 4%")
    axes[1].barh(y_pos - h, df_mod["Ruido 10% 300-1000pb (%)"], height=h, color="#d62728", alpha=0.9, label="Amplicons Ruído 10%")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(nomes_m, fontweight="bold", fontsize=14)
    axes[1].set_xlabel("Acurácia em Amplicons 300–999 pb (%)", fontsize=15, fontweight="bold")
    axes[1].set_title("B) Conv1D 1×1 vs PMC Enxuto em Amplicons Reais", fontsize=15.5, fontweight="bold", pad=12)
    axes[1].set_xlim(70, 95.5)
    axes[1].tick_params(axis="x", labelsize=13.5)
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right", fontsize=14)

    plt.suptitle("Análise Espacial (Conv1D 1×1 por Segmentos) e Varredura Posicional no Genoma DENV", fontsize=17, fontweight="bold", y=1.02)
    plt.tight_layout(w_pad=3.0)
    _salvar_figura(os.path.join(DIR_GRAFICOS, "comparativo_cnn1d_posicao_genomica.png"), 6)


def gerar_7_diagrama_confiabilidade(modelos, X_te_no, y_te_no, X_amp_in, y_amp_in):
    """7. Diagrama de Confiabilidade e ECE em 600 DPI (legendas externas abaixo sem sobreposição)."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 8.2))
    cores = {
        "PMC Enxuto V3.0 (32,) + Dropout": ("#1f77b4", "s", "-", "PMC Enxuto (32,)"),
        "Regressão Logística (L2, C=1.0)": ("#d62728", "o", "--", "Reg. Logística"),
        "PMC Grande (512, 128)": ("#2ca02c", "^", "-.", "PMC Grande"),
    }

    for ax, titulo, X_eval, y_eval in [
        (axes[0], f"A) Teste Cego Não-Sobreposto (N={len(y_te_no):,})", X_te_no, y_te_no),
        (axes[1], f"B) Amplicons Inéditos (N={len(y_amp_in)})", X_amp_in, y_amp_in),
    ]:
        ax.plot([0, 1], [0, 1], "k:", linewidth=2.4, label="Calibração Perfeita (y = x)")
        for nome, pipe in modelos.items():
            probas = pipe.predict_proba(X_eval)
            brier = calcular_brier(y_eval, probas, CLASSES)
            ece, _, b_conf, b_acc, _ = calcular_ece_e_bins(y_eval, probas, CLASSES, n_bins=10)
            cor, mk, ls, curto = cores[nome]
            valid = ~np.isnan(b_conf)
            ax.plot(
                b_conf[valid],
                b_acc[valid],
                color=cor,
                marker=mk,
                linestyle=ls,
                linewidth=3.2,
                markersize=10,
                label=f"{curto} (ECE={ece:.4f} | Brier={brier:.4f})",
            )
        ax.set_title(titulo, fontsize=15.5, fontweight="bold", pad=12)
        ax.set_xlabel("Confiança Média Predita (Softmax)", fontsize=15, fontweight="bold")
        ax.set_ylabel("Acurácia Empírica Observada", fontsize=15, fontweight="bold")
        ax.set_xlim(0.25, 1.03)
        ax.set_ylim(0.25, 1.03)
        ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.tick_params(labelsize=13.5)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=1,
            fontsize=14,
            frameon=True,
            fancybox=True,
        )

    plt.suptitle("Análise de Calibração Probabilística — Diagrama de Confiabilidade e ECE", fontsize=17.5, fontweight="bold", y=1.01)
    plt.tight_layout(w_pad=2.8)
    _salvar_figura(os.path.join(DIR_GRAFICOS, "diagrama_confiabilidade_ece.png"), 7)


def main():
    print("=" * 85)
    print("ATUALIZAÇÃO COMPLETA DOS 7 GRÁFICOS CIENTÍFICOS EM ALTA RESOLUÇÃO (600 DPI)")
    print("=" * 85)

    remover_graficos_antigos()
    os.makedirs(DIR_GRAFICOS, exist_ok=True)

    # Carregar dados unificados (3.109 genomas) e pipeline V3.0
    todos = []
    for arq, sor in ARQUIVOS_SOROTIPOS.items():
        regs, _ = carregar_fasta(arq, sor)
        todos.extend(regs)
    unicos, _, _ = deduplicar_genomas_estrito(todos)
    train_gen, test_gen = train_test_split(unicos, test_size=0.20, random_state=42, stratify=[g["sorotipo"] for g in unicos])
    train_hashes = {g["hash"] for g in train_gen}
    train_seqs = [g["seq"] for g in train_gen]

    frags_tr = [f for g in train_gen for f in fatiar_genoma(g["seq"], g["id"], g["sorotipo"], 2000, 1000)]
    X_tr = np.array([extrair_frequencias_kmers(f["seq"], K_MER_SIZE) for f in frags_tr], dtype=np.float32)
    y_tr = np.array([f["sorotipo"] for f in frags_tr])

    frags_te_no = [f for g in test_gen for f in fatiar_genoma(g["seq"], g["id"], g["sorotipo"], 2000, 2000)]
    X_te_no = np.array([extrair_frequencias_kmers(f["seq"], K_MER_SIZE) for f in frags_te_no], dtype=np.float32)
    y_te_no = np.array([f["sorotipo"] for f in frags_te_no])

    pipe_pmc = carregar_pipeline()
    pipe_lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=300, C=1.0, random_state=42, solver="lbfgs"))]).fit(X_tr, y_tr)
    pipe_grande = Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(512, 128), activation="relu", alpha=0.001, max_iter=40, early_stopping=True, random_state=42))]).fit(X_tr, y_tr)

    modelos = {
        "PMC Enxuto V3.0 (32,) + Dropout": pipe_pmc,
        "Regressão Logística (L2, C=1.0)": pipe_lr,
        "PMC Grande (512, 128)": pipe_grande,
    }

    # Preparar teste_aleatorio.fasta (Todas N=1.846 e Inéditas N=1.595)
    recs = ler_fasta_generico(os.path.join(DIR_DADOS_TESTES, "teste_aleatorio.fasta"))
    rows_all = []
    for rid, desc, seq in recs:
        lbl, is_pat = parse_header_organism(desc)
        if lbl not in CLASSES or is_pat or len(seq) < K_MER_SIZE:
            continue
        h = hashlib.sha256(seq.encode()).hexdigest()
        in_tr = (h in train_hashes) or (any(seq in ts for ts in train_seqs) if len(seq) >= 100 else False)
        feat = extrair_frequencias_kmers(seq, K_MER_SIZE).reshape(1, -1)
        pred = pipe_pmc.predict(feat)[0]
        conf = float(np.max(pipe_pmc.predict_proba(feat)[0]) * 100)
        rows_all.append({"ID": rid, "Real": lbl, "Pred": pred, "Conf": conf, "Tam": len(seq), "Inedita": not in_tr, "Seq": seq})

    df_all = pd.DataFrame(rows_all)
    df_all["Faixa"] = pd.cut(df_all["Tam"], bins=[0, 300, 1000, 5000, np.inf], labels=["<300", "300-1000", "1000-5000", ">=5000"], right=False)
    df_ineditas = df_all[df_all["Inedita"]].copy()

    mask_amp_in = df_ineditas["Faixa"] == "300-1000"
    X_amp_in = np.array([extrair_frequencias_kmers(s, K_MER_SIZE) for s in df_ineditas.loc[mask_amp_in, "Seq"]], dtype=np.float32)
    y_amp_in = df_ineditas.loc[mask_amp_in, "Real"].values

    # Gerar todos os 7 gráficos em 600 DPI
    gerar_1_matriz_confusao(pipe_pmc, X_te_no, y_te_no)
    gerar_2_avaliacao_teste_aleatorio(pipe_pmc, df_all, df_ineditas)
    gerar_3_curva_degradacao_ruido_5sementes()
    gerar_4_ablacao_k4_k6()
    gerar_5_comparativo_rbf()
    gerar_6_segment_conv1d_posicao()
    gerar_7_diagrama_confiabilidade(modelos, X_te_no, y_te_no, X_amp_in, y_amp_in)

    print("\nTodos os 7 gráficos foram regerados com sucesso em 600 DPI!")


if __name__ == "__main__":
    main()
