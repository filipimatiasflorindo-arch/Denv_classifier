"""
================================================================================
1. Teste Cego Interno em Janelas Estritamente Não-Sobrepostas (step=2000 pb)
   e Avaliação Agregada por Genoma Independente (N=622).
2. Avaliação em Conjunto Externo 100% Inédito (sequences_20261102_9338039.fasta
   + BVBRC_genome_sequence.fasta) com auditoria SHA-256/substring contra treino
   e validação externa, em sequências completas e amplicons não-sobrepostos de
   500 pb sob 0%, 4% e 10% de ruído NGS.
3. Análise de Calibração Probabilística: Expected Calibration Error (ECE),
   Brier Score Multiclasse e Diagrama de Confiabilidade (Reliability Diagram)
   comparando PMC Enxuto V3.0, Regressão Logística e PMC Grande (512, 128).
4. Avaliação Multi-Semente (5 sementes, Média ± DP + Teste Exato de McNemar)
   focada exclusivamente nas faixas submetidas a ruído (Amplicons 300-1.000 pb
   e Reads < 300 pb) da Validação Externa estritamente inédita (N=1.595),
   eliminando o efeito de mascaramento da acurácia global sob ruído.
================================================================================
"""

import hashlib
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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
    aplicar_ruido_estocastico,
    carregar_fasta,
    carregar_pipeline,
    deduplicar_genomas_estrito,
    extrair_frequencias_kmers,
    fatiar_genoma,
)
from avaliar_validacao_externa import parse_header_organism


def ler_fasta_generico(caminho: str):
    """Lê registros (id, descricao, seq_normalizada) de um arquivo FASTA."""
    recs = []
    with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
        cid, cdesc, cseq = "", "", []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cid:
                    recs.append((cid, cdesc, "".join(cseq).upper().replace("U", "T")))
                    cseq = []
                cdesc = line[1:].strip()
                cid = cdesc.split()[0] if cdesc else f"Seq_{len(recs) + 1}"
            else:
                cseq.append(line)
        if cid:
            recs.append((cid, cdesc, "".join(cseq).upper().replace("U", "T")))
    return recs


def calcular_ece_e_bins(y_true: np.ndarray, probas: np.ndarray, classes: list, n_bins: int = 10):
    """
    Calcula o Expected Calibration Error (ECE), Maximum Calibration Error (MCE)
    e estatísticas por faixa de confiança para o Diagrama de Confiabilidade.
    """
    pred_idx = np.argmax(probas, axis=1)
    pred_labels = np.array([classes[i] for i in pred_idx])
    confidencias = np.max(probas, axis=1)
    corretos = (pred_labels == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n_total = len(y_true)
    bin_confs, bin_accs, bin_counts = [], [], []

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confidencias >= lo) & (confidencias <= hi)
        else:
            mask = (confidencias >= lo) & (confidencias < hi)
        cnt = int(np.sum(mask))
        bin_counts.append(cnt)
        if cnt > 0:
            avg_conf = float(np.mean(confidencias[mask]))
            avg_acc = float(np.mean(corretos[mask]))
            gap = abs(avg_acc - avg_conf)
            ece += (cnt / n_total) * gap
            mce = max(mce, gap)
            bin_confs.append(avg_conf)
            bin_accs.append(avg_acc)
        else:
            bin_confs.append(np.nan)
            bin_accs.append(np.nan)

    return float(ece), float(mce), np.array(bin_confs), np.array(bin_accs), np.array(bin_counts)


def calcular_brier(y_true: np.ndarray, probas: np.ndarray, classes: list) -> float:
    lb = LabelBinarizer()
    lb.fit(classes)
    y_bin = lb.transform(y_true)
    return float(np.mean(np.sum((y_bin - probas) ** 2, axis=1)))


def main():
    print("=" * 95)
    print("VALIDAÇÃO FINAL BLINDADA PARA ARTIGO CIENTÍFICO (PADRÃO SBC)")
    print("=" * 95)

    # 1. Carregar e particionar genomas de referência (3.109 únicos -> 80/20)
    todos_genomas = []
    for arq, sorotipo in ARQUIVOS_SOROTIPOS.items():
        regs, _ = carregar_fasta(arq, sorotipo)
        todos_genomas.extend(regs)

    unicos, _, _ = deduplicar_genomas_estrito(todos_genomas)
    train_gen, test_gen = train_test_split(
        unicos, test_size=0.20, random_state=42, stratify=[g["sorotipo"] for g in unicos]
    )
    train_hashes = {g["hash"] for g in train_gen}
    train_seqs = [g["seq"] for g in train_gen]

    # Treinar modelos comparativos no mesmo conjunto de treino (22.383 fragmentos)
    frags_train = [f for g in train_gen for f in fatiar_genoma(g["seq"], g["id"], g["sorotipo"], 2000, 1000)]
    X_tr = np.array([extrair_frequencias_kmers(f["seq"], K_MER_SIZE) for f in frags_train], dtype=np.float32)
    y_tr = np.array([f["sorotipo"] for f in frags_train])

    classes = ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]
    pipe_pmc = carregar_pipeline()

    print("\n[1/4] Treinando modelos comparativos (Regressão Logística e PMC Grande 512-128)...")
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=300, C=1.0, random_state=42, solver="lbfgs")),
    ])
    pipe_lr.fit(X_tr, y_tr)

    pipe_pmc_grande = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(512, 128), activation="relu", alpha=0.001, max_iter=40, early_stopping=True, random_state=42)),
    ])
    pipe_pmc_grande.fit(X_tr, y_tr)

    modelos = {
        "PMC Enxuto V3.0 (32,) + Dropout": pipe_pmc,
        "Regressão Logística (L2, C=1.0)": pipe_lr,
        "PMC Grande (512, 128)": pipe_pmc_grande,
    }

    # ==========================================================================
    # ETAPA 1: TESTE CEGO INTERNO EM JANELAS NÃO-SOBREPOSTAS (step=2000) + POR GENOMA
    # ==========================================================================
    print("\n" + "-" * 95)
    print("[2/4] TESTE CEGO INTERNO: JANELAS NÃO-SOBREPOSTAS (step=2000 pb) E AGREGAÇÃO POR GENOMA (N=622)")
    print("-" * 95)

    frags_test_nonoverlap = [
        f for g in test_gen for f in fatiar_genoma(g["seq"], g["id"], g["sorotipo"], 2000, 2000)
    ]
    X_te_no = np.array([extrair_frequencias_kmers(f["seq"], K_MER_SIZE) for f in frags_test_nonoverlap], dtype=np.float32)
    y_te_no = np.array([f["sorotipo"] for f in frags_test_nonoverlap])
    gen_ids_no = np.array([f["id_genoma_origem"] for f in frags_test_nonoverlap])

    print(f"Total de fragmentos não-sobrepostos (2.000 pb, step=2.000 pb): {len(frags_test_nonoverlap):,} (de {len(test_gen)} genomas)")

    registros_calibracao = []
    dados_diagrama_te = {}

    for nome, pipe in modelos.items():
        preds = pipe.predict(X_te_no)
        probas = pipe.predict_proba(X_te_no)
        acc_frag = accuracy_score(y_te_no, preds) * 100
        bal_frag = balanced_accuracy_score(y_te_no, preds) * 100
        f1_frag = f1_score(y_te_no, preds, average="macro") * 100
        brier = calcular_brier(y_te_no, probas, classes)
        ece, mce, b_conf, b_acc, b_cnt = calcular_ece_e_bins(y_te_no, probas, classes, n_bins=10)
        dados_diagrama_te[nome] = (b_conf, b_acc, b_cnt, ece, brier)

        # Agregação por genoma (voto probabilístico médio das janelas não-sobrepostas de cada genoma)
        df_tmp = pd.DataFrame(probas, columns=classes)
        df_tmp["genoma"] = gen_ids_no
        df_tmp["real"] = y_te_no
        agg = df_tmp.groupby("genoma").agg({**{c: "mean" for c in classes}, "real": "first"})
        pred_gen = agg[classes].idxmax(axis=1).values
        acc_gen = accuracy_score(agg["real"].values, pred_gen) * 100

        registros_calibracao.append({
            "Conjunto": "Teste Cego Não-Sobreposto (2.000 pb)",
            "N": len(y_te_no),
            "Modelo": nome,
            "Acuracia (%)": round(acc_frag, 2),
            "Acc Balanceada (%)": round(bal_frag, 2),
            "Macro-F1 (%)": round(f1_frag, 2),
            "Acc por Genoma N=622 (%)": round(acc_gen, 2),
            "Brier Score": round(brier, 5),
            "ECE": round(ece, 5),
            "MCE": round(mce, 5),
        })
        print(
            f"   [{nome:<32}] Frags Não-Sobrep.: {acc_frag:6.2f}% (Bal: {bal_frag:6.2f}%, F1: {f1_frag:6.2f}%) | "
            f"Por Genoma (N=622): {acc_gen:6.2f}% | Brier: {brier:.5f} | ECE: {ece:.5f}"
        )

    # ==========================================================================
    # ETAPA 2: VALIDAÇÃO EXTERNA ESTRITAMENTE INÉDITA (N=1.595) + 5 SEMENTES (SEM GLOBAL MASCARADA)
    # ==========================================================================
    print("\n" + "-" * 95)
    print("[3/4] VALIDAÇÃO EXTERNA INÉDITA (teste_aleatorio.fasta SEM SOBREPOSIÇÃO: N=1.595) — 5 SEMENTES + ECE")
    print("-" * 95)

    recs_val = ler_fasta_generico(os.path.join(DIR_DADOS_TESTES, "teste_aleatorio.fasta"))
    val_hashes = set()
    val_ineditas = []
    for rid, desc, seq in recs_val:
        lbl, is_pat = parse_header_organism(desc)
        if lbl not in classes or is_pat or len(seq) < K_MER_SIZE:
            continue
        h = hashlib.sha256(seq.encode()).hexdigest()
        val_hashes.add(h)
        in_tr_exact = h in train_hashes
        in_tr_sub = any(seq in ts for ts in train_seqs) if len(seq) >= 100 else False
        if not in_tr_exact and not in_tr_sub:
            val_ineditas.append((rid, lbl, len(seq), seq))

    df_val = pd.DataFrame(val_ineditas, columns=["ID", "Real", "Tam", "Seq"])
    mask_curto = (df_val["Tam"] < 300).values
    mask_amp = ((df_val["Tam"] >= 300) & (df_val["Tam"] < 1000)).values

    print(f"Sequências estritamente inéditas na Validação Externa: {len(df_val):,} (Reads <300pb: {np.sum(mask_curto)} | Amplicons 300-1000pb: {np.sum(mask_amp)})")

    # Calibração (ECE) em Amplicons Inéditos Limpos (N=201)
    X_amp_in = np.array([extrair_frequencias_kmers(s, K_MER_SIZE) for s in df_val.loc[mask_amp, "Seq"]], dtype=np.float32)
    y_amp_in = df_val.loc[mask_amp, "Real"].values
    dados_diagrama_amp = {}

    for nome, pipe in modelos.items():
        preds = pipe.predict(X_amp_in)
        probas = pipe.predict_proba(X_amp_in)
        acc = accuracy_score(y_amp_in, preds) * 100
        bal = balanced_accuracy_score(y_amp_in, preds) * 100
        f1 = f1_score(y_amp_in, preds, average="macro") * 100
        brier = calcular_brier(y_amp_in, probas, classes)
        ece, mce, b_conf, b_acc, b_cnt = calcular_ece_e_bins(y_amp_in, probas, classes, n_bins=10)
        dados_diagrama_amp[nome] = (b_conf, b_acc, b_cnt, ece, brier)
        registros_calibracao.append({
            "Conjunto": "Validação Externa Inédita - Amplicons (300-1000 pb)",
            "N": len(y_amp_in),
            "Modelo": nome,
            "Acuracia (%)": round(acc, 2),
            "Acc Balanceada (%)": round(bal, 2),
            "Macro-F1 (%)": round(f1, 2),
            "Acc por Genoma N=622 (%)": np.nan,
            "Brier Score": round(brier, 5),
            "ECE": round(ece, 5),
            "MCE": round(mce, 5),
        })
        print(f"   [Amplicons Inéditos N={len(y_amp_in)} | {nome:<32}] Acc: {acc:5.2f}% | Bal: {bal:5.2f}% | F1: {f1:5.2f}% | Brier: {brier:.5f} | ECE: {ece:.5f}")

    # Multi-semente (5 sementes) focado EXCLUSIVAMENTE nas faixas que recebem ruído (<300 pb e 300-1000 pb)
    sementes = [42, 1042, 2042, 3042, 4042]
    registros_multiseed = []

    for faixa_nome, mask_f in [("Amplicons Inéditos (300-1.000 pb)", mask_amp), ("Reads Inéditas (< 300 pb)", mask_curto)]:
        seqs_f = df_val.loc[mask_f, "Seq"].tolist()
        y_f = df_val.loc[mask_f, "Real"].values
        print(f"\n   >> Ruído NGS Multi-Semente (5 sementes) em {faixa_nome} (N={len(y_f)}):")

        for taxa in [0.00, 0.04, 0.08, 0.10, 0.15]:
            accs_por_mod = {m: [] for m in modelos}
            f1s_por_mod = {m: [] for m in modelos}
            preds_seed42 = {}

            for s_base in (sementes if taxa > 0 else [42]):
                if taxa > 0:
                    seqs_n = [aplicar_ruido_estocastico(s, taxa_erro=taxa, random_state=s_base + i) for i, s in enumerate(seqs_f)]
                else:
                    seqs_n = seqs_f
                Xn = np.array([extrair_frequencias_kmers(s, K_MER_SIZE) for s in seqs_n], dtype=np.float32)
                for m_nome, pipe in modelos.items():
                    p = pipe.predict(Xn)
                    accs_por_mod[m_nome].append(accuracy_score(y_f, p) * 100)
                    f1s_por_mod[m_nome].append(f1_score(y_f, p, average="macro") * 100)
                    if s_base == 42:
                        preds_seed42[m_nome] = p

            # McNemar PMC V3.0 vs Regressão Logística na semente 42
            c_pmc = preds_seed42["PMC Enxuto V3.0 (32,) + Dropout"] == y_f
            c_lr = preds_seed42["Regressão Logística (L2, C=1.0)"] == y_f
            b_val = int(np.sum(c_pmc & ~c_lr))
            c_val = int(np.sum(~c_pmc & c_lr))
            p_mcnemar = binomtest(b_val, b_val + c_val, 0.5).pvalue if (b_val + c_val) > 0 else 1.0

            for m_nome in modelos:
                mean_acc = float(np.mean(accs_por_mod[m_nome]))
                std_acc = float(np.std(accs_por_mod[m_nome]))
                mean_f1 = float(np.mean(f1s_por_mod[m_nome]))
                registros_multiseed.append({
                    "Faixa": faixa_nome,
                    "N": len(y_f),
                    "Ruido (%)": int(taxa * 100),
                    "Modelo": m_nome,
                    "Acc Media (%)": round(mean_acc, 2),
                    "Acc DP (%)": round(std_acc, 2),
                    "Macro-F1 Media (%)": round(mean_f1, 2),
                    "McNemar p (vs RL, seed 42)": round(p_mcnemar, 5) if "PMC Enxuto" in m_nome else np.nan,
                })

            print(
                f"      Ruído {int(taxa*100):2d}% | PMC V3.0: {np.mean(accs_por_mod['PMC Enxuto V3.0 (32,) + Dropout']):5.2f}% ± {np.std(accs_por_mod['PMC Enxuto V3.0 (32,) + Dropout']):.2f}% "
                f"| RL: {np.mean(accs_por_mod['Regressão Logística (L2, C=1.0)']):5.2f}% ± {np.std(accs_por_mod['Regressão Logística (L2, C=1.0)']):.2f}% "
                f"| PMC Grande: {np.mean(accs_por_mod['PMC Grande (512, 128)']):5.2f}% ± {np.std(accs_por_mod['PMC Grande (512, 128)']):.2f}% "
                f"| McNemar PMC vs RL (seed 42): p={p_mcnemar:.5f} (b={b_val}, c={c_val})"
            )

    # ==========================================================================
    # ETAPA 3: CONJUNTO DE TESTE CEGO EXTERNO 100% INTOCADO (2025/2026 + BV-BRC)
    # ==========================================================================
    print("\n" + "-" * 95)
    print("[4/4] TESTE CEGO EXTERNO 100% INTOCADO (sequences_20261102_9338039.fasta + BVBRC_genome_sequence.fasta)")
    print("-" * 95)

    recs_ext = (
        ler_fasta_generico(os.path.join(DIR_DADOS_TESTES, "sequences_20261102_9338039.fasta"))
        + ler_fasta_generico(os.path.join(DIR_DADOS_TESTES, "BVBRC_genome_sequence.fasta"))
    )

    ext_intocados = []
    sobreposicoes_ext = 0
    for rid, desc, seq in recs_ext:
        lbl, is_pat = parse_header_organism(desc)
        if lbl not in classes or is_pat or len(seq) < K_MER_SIZE:
            continue
        h = hashlib.sha256(seq.encode()).hexdigest()
        in_prev = (h in train_hashes) or (h in val_hashes) or (any(seq in ts for ts in train_seqs) if len(seq) >= 100 else False)
        if in_prev:
            sobreposicoes_ext += 1
            continue
        ext_intocados.append((rid, lbl, len(seq), seq))

    print(f"Sequências externas 100% intocadas (zero sobreposição com treino/validação): {len(ext_intocados)} (descartadas por sobreposição: {sobreposicoes_ext})")

    # 3A. Sequências completas/originais do teste cego externo
    X_ext_orig = np.array([extrair_frequencias_kmers(s, K_MER_SIZE) for _, _, _, s in ext_intocados], dtype=np.float32)
    y_ext_orig = np.array([lbl for _, lbl, _, _ in ext_intocados])

    # 3B. Amplicons não-sobrepostos de 500 pb fatiados dos isolados externos intocados
    amplicons_ext = []
    for rid, lbl, comp, seq in ext_intocados:
        if comp >= 500:
            for start in range(0, comp - 500 + 1, 500):
                amplicons_ext.append((rid, lbl, seq[start : start + 500]))
        else:
            amplicons_ext.append((rid, lbl, seq))

    seqs_amp_ext = [s for _, _, s in amplicons_ext]
    y_amp_ext = np.array([lbl for _, lbl, _ in amplicons_ext])
    print(f"Amplicons clínicos não-sobrepostos de 500 pb gerados dos isolados intocados: N={len(amplicons_ext)}")

    registros_ext_final = []
    for nome, pipe in modelos.items():
        acc_orig = accuracy_score(y_ext_orig, pipe.predict(X_ext_orig)) * 100
        # Avaliar amplicons intocados em 0%, 4% e 10% (5 sementes para ruído)
        res_taxas = {}
        for taxa in [0.00, 0.04, 0.10]:
            accs_t = []
            for s_base in (sementes if taxa > 0 else [42]):
                sn = [aplicar_ruido_estocastico(s, taxa_erro=taxa, random_state=s_base + i) for i, s in enumerate(seqs_amp_ext)] if taxa > 0 else seqs_amp_ext
                Xn = np.array([extrair_frequencias_kmers(s, K_MER_SIZE) for s in sn], dtype=np.float32)
                accs_t.append(accuracy_score(y_amp_ext, pipe.predict(Xn)) * 100)
            res_taxas[taxa] = (float(np.mean(accs_t)), float(np.std(accs_t)))

        registros_ext_final.append({
            "Modelo": nome,
            "Seqs Originais Intocadas N": len(y_ext_orig),
            "Acc Seqs Originais (%)": round(acc_orig, 2),
            "Amplicons 500pb Intocados N": len(y_amp_ext),
            "Amplicons Limpos (%)": round(res_taxas[0.00][0], 2),
            "Amplicons Ruido 4% Media (%)": round(res_taxas[0.04][0], 2),
            "Amplicons Ruido 4% DP (%)": round(res_taxas[0.04][1], 2),
            "Amplicons Ruido 10% Media (%)": round(res_taxas[0.10][0], 2),
            "Amplicons Ruido 10% DP (%)": round(res_taxas[0.10][1], 2),
        })
        print(
            f"   [{nome:<32}] Originais (N={len(y_ext_orig)}): {acc_orig:6.2f}% | "
            f"Amplicons 500pb (N={len(y_amp_ext)}): Limpo={res_taxas[0.00][0]:5.2f}% | "
            f"R4%={res_taxas[0.04][0]:5.2f}% ± {res_taxas[0.04][1]:.2f}% | "
            f"R10%={res_taxas[0.10][0]:5.2f}% ± {res_taxas[0.10][1]:.2f}%"
        )

    # Salvar CSVs finais
    dir_bench = os.path.join(DIR_RESULTADOS, "benchmarks")
    os.makedirs(dir_bench, exist_ok=True)
    pd.DataFrame(registros_calibracao).to_csv(os.path.join(dir_bench, "validacao_sbc_calibracao_ece.csv"), index=False)
    pd.DataFrame(registros_multiseed).to_csv(os.path.join(dir_bench, "validacao_sbc_ruido_5sementes.csv"), index=False)
    pd.DataFrame(registros_ext_final).to_csv(os.path.join(dir_bench, "validacao_sbc_teste_externo_intocado.csv"), index=False)

    # Gerar Diagrama de Confiabilidade (Reliability Diagram) 300 DPI
    os.makedirs(DIR_GRAFICOS, exist_ok=True)
    caminho_png = os.path.join(DIR_GRAFICOS, "diagrama_confiabilidade_ece.png")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    cores = {
        "PMC Enxuto V3.0 (32,) + Dropout": ("#1f77b4", "s", "-"),
        "Regressão Logística (L2, C=1.0)": ("#d62728", "o", "--"),
        "PMC Grande (512, 128)": ("#2ca02c", "^", "-."),
    }

    for ax, titulo, dados in [
        (axes[0], "A) Teste Cego Não-Sobreposto (2.000 pb, N=3.110)", dados_diagrama_te),
        (axes[1], f"B) Validação Externa Inédita — Amplicons (300–1.000 pb, N={len(y_amp_in)})", dados_diagrama_amp),
    ]:
        ax.plot([0, 1], [0, 1], "k:", linewidth=1.6, label="Calibração Perfeita (y = x)")
        for nome, (b_conf, b_acc, _, ece, brier) in dados.items():
            cor, mk, ls = cores[nome]
            valid = ~np.isnan(b_conf)
            ax.plot(
                b_conf[valid],
                b_acc[valid],
                color=cor,
                marker=mk,
                linestyle=ls,
                linewidth=2.2,
                markersize=7,
                label=f"{nome.split(' (')[0]} (ECE={ece:.4f} | Brier={brier:.4f})",
            )
        ax.set_title(titulo, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Confiança Média Predita (Probabilidade Softmax)", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Acurácia Empírica Observada", fontsize=10.5, fontweight="bold")
        ax.set_xlim(0.25, 1.02)
        ax.set_ylim(0.25, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8.5)

    plt.suptitle("Análise de Calibração Probabilística (Diagrama de Confiabilidade e ECE)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nDiagrama de Confiabilidade salvo em '{caminho_png}'.")
    print("CSVs salvos em 'resultados/benchmarks/validacao_sbc_*.csv'.")


if __name__ == "__main__":
    main()
