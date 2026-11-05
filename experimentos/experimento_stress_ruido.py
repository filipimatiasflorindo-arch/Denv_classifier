import os
import sys
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXP_DIR)
for _p in (ROOT_DIR, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import PMC
from PMC import (
    DIR_DADOS_TESTES,
    DIR_GRAFICOS,
    DIR_RESULTADOS,
    K_MER_SIZE,
    NUM_KMERS,
    LeanMLPDropout,
    aplicar_ruido_estocastico,
    carregar_pipeline,
    carregar_fasta,
    fatiar_genoma,
    ARQUIVOS_SOROTIPOS,
)
from testar_arquiteturas_k6 import build_kmer_tools, extract_vec
from analisar_teste_aleatorio import parse_header_organism


def main():
    print("=" * 85)
    print("BENCHMARK DE ESTRESSE: RESILIÊNCIA A RUÍDO EXTREMO (8%, 10% E 15%)")
    print("Simulação de Erros Estocásticos de Basecalling / NGS Bruto / RNA Degradado")
    print("=" * 85)

    # 1. Carregar genomas para treinar a Regressão Logística e PMC Grande
    print("\n[1/4] Carregando genomas e fatiando para treinamento dos modelos comparativos...")
    todos_genomas = []
    for arq, sorotipo in ARQUIVOS_SOROTIPOS.items():
        registros, _ = carregar_fasta(arq, sorotipo)
        todos_genomas.extend(registros)

    vistos = set()
    unicos = [g for g in todos_genomas if not (g['hash'] in vistos or vistos.add(g['hash']))]

    from sklearn.model_selection import train_test_split
    y_gen = [g['sorotipo'] for g in unicos]
    train_gen, test_gen = train_test_split(unicos, test_size=0.20, random_state=42, stratify=y_gen)

    frags_train = []
    for g in train_gen:
        frags_train.extend(fatiar_genoma(g['seq'], g['id'], g['sorotipo'], 2000, 1000))

    y_tr = np.array([f['sorotipo'] for f in frags_train])
    k_idx, num_kmers = build_kmer_tools(6)

    print(f"Extraindo características K=6 para {len(frags_train):,} fragmentos de treino...")
    t0 = time.time()
    X_tr = np.array([extract_vec(f['seq'], 6, k_idx, num_kmers) for f in frags_train])
    print(f"Extração concluída em {time.time()-t0:.1f}s.")

    # 2. Carregar / Treinar Modelos
    print("\n[2/4] Preparando os modelos para o benchmark:")

    # Modelo 1: PMC V3.0 Oficial (Lean 32 + Dropout 0.3 + L2 0.01)
    print(" - Carregando PMC V3.0 Enxuto (32,) + Dropout (p=0.3) + L2 de 'modelos/pipeline_pmc_denv.joblib'...")
    pipe_pmc_v3 = carregar_pipeline()

    # Modelo 2: Regressão Logística
    print(" - Treinando Regressão Logística (L2, C=1.0)...")
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=200, C=1.0, random_state=42, solver='lbfgs'))
    ])
    t0 = time.time()
    pipe_lr.fit(X_tr, y_tr)
    print(f"   Regressão Logística treinada em {time.time()-t0:.1f}s.")

    # Modelo 3: PMC Grande (512, 128)
    print(" - Treinando PMC Grande (512, 128) [alpha=0.001]...")
    pipe_pmc_grande = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(512, 128), activation='relu', alpha=0.001,
                             max_iter=40, early_stopping=True, random_state=42))
    ])
    t0 = time.time()
    pipe_pmc_grande.fit(X_tr, y_tr)
    print(f"   PMC Grande treinado em {time.time()-t0:.1f}s.")

    modelos = {
        "PMC V3.0 (32,) + Dropout (p=0.3)": pipe_pmc_v3,
        "Regressão Logística (L2)": pipe_lr,
        "PMC Grande (512, 128)": pipe_pmc_grande,
    }

    # 3. Carregar sequências de teste_aleatorio.fasta
    print("\n[3/4] Carregando e validando sequências de 'dados/testes/teste_aleatorio.fasta'...")
    caminho_fasta = os.path.join(DIR_DADOS_TESTES, 'teste_aleatorio.fasta')
    aleatorio_recs = []
    with open(caminho_fasta, 'r', encoding='utf-8', errors='ignore') as f:
        cur_desc, cur_seq = '', []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if cur_desc:
                    aleatorio_recs.append((cur_desc, ''.join(cur_seq)))
                    cur_seq = []
                cur_desc = line[1:]
            else:
                cur_seq.append(line)
        if cur_desc:
            aleatorio_recs.append((cur_desc, ''.join(cur_seq)))

    classes = ['DENV-1', 'DENV-2', 'DENV-3', 'DENV-4']
    aleatorio_validos = []
    for desc, s in aleatorio_recs:
        lbl, is_pat = parse_header_organism(desc)
        if lbl in classes and not is_pat and len(s) >= 6:
            aleatorio_validos.append((lbl, len(s), s))

    y_aleat = np.array([lbl for lbl, _, _ in aleatorio_validos])
    lens_aleat = np.array([l for _, l, _ in aleatorio_validos])
    print(f"Total de sequências de teste validadas com ground-truth: {len(aleatorio_validos):,}")

    mask_curto = lens_aleat < 300
    mask_amplicon = (lens_aleat >= 300) & (lens_aleat < 1000)
    mask_gene = (lens_aleat >= 1000) & (lens_aleat < 5000)
    mask_genoma = lens_aleat >= 5000

    print(f"Distribuição: <300pb (N={np.sum(mask_curto)}) | 300-1000pb (N={np.sum(mask_amplicon)}) | 1000-5000pb (N={np.sum(mask_gene)}) | >=5000pb (N={np.sum(mask_genoma)})")

    # 4. Avaliar Níveis de Ruído: 0%, 4%, 8%, 10% e 15%
    taxas_ruido = [0.00, 0.04, 0.08, 0.10, 0.15]
    print("\n[4/4] Executando benchmark sob taxas de ruído estocástico: 0%, 4%, 8%, 10% e 15%...")

    registros_stress = []

    for taxa in taxas_ruido:
        pct_label = f"{int(taxa * 100)}%"
        print(f"\n" + "-" * 75)
        print(f"-> AVALIANDO TAXA DE RUÍDO ESTOCÁSTICO: {pct_label}")
        print("-" * 75)

        # Gerar sequências com ruído (em fragmentos < 1000 pb para simulação NGS)
        t_ini_noise = time.time()
        X_taxa = []
        for i, (lbl, comp, s) in enumerate(aleatorio_validos):
            if taxa > 0.0 and comp < 1000:
                s_mod = aplicar_ruido_estocastico(s, taxa_erro=taxa, random_state=42 + i)
            else:
                s_mod = s
            X_taxa.append(extract_vec(s_mod, 6, k_idx, num_kmers))
        X_taxa = np.array(X_taxa)
        print(f"Extração de características para ruído {pct_label} concluída em {time.time()-t_ini_noise:.1f}s.")

        for nome_mod, pipe in modelos.items():
            preds = pipe.predict(X_taxa)
            acc_glob = accuracy_score(y_aleat, preds) * 100
            bal_acc = balanced_accuracy_score(y_aleat, preds) * 100
            f1_macro = f1_score(y_aleat, preds, average='macro') * 100

            acc_curto = accuracy_score(y_aleat[mask_curto], preds[mask_curto]) * 100
            acc_amp = accuracy_score(y_aleat[mask_amplicon], preds[mask_amplicon]) * 100
            acc_gene = accuracy_score(y_aleat[mask_gene], preds[mask_gene]) * 100
            acc_genoma = accuracy_score(y_aleat[mask_genoma], preds[mask_genoma]) * 100

            reg = {
                "Ruído": pct_label,
                "Taxa": taxa,
                "Modelo": nome_mod,
                "Acc Global (%)": round(acc_glob, 2),
                "Macro F1 (%)": round(f1_macro, 2),
                "Acc < 300 pb (%)": round(acc_curto, 2),
                "Acc 300-1000 pb (%)": round(acc_amp, 2),
                "Acc 1000-5000 pb (%)": round(acc_gene, 2),
                "Acc >= 5000 pb (%)": round(acc_genoma, 2),
            }
            registros_stress.append(reg)
            print(f"   [{nome_mod:<32}] Global: {acc_glob:5.2f}% | <300pb: {acc_curto:5.2f}% | 300-1000pb: {acc_amp:5.2f}%")

    dir_bench = os.path.join(DIR_RESULTADOS, "benchmarks")
    os.makedirs(dir_bench, exist_ok=True)
    caminho_csv = os.path.join(dir_bench, "benchmark_stress_ruido_8_10_15.csv")
    df_stress = pd.DataFrame(registros_stress)
    df_stress.to_csv(caminho_csv, index=False)
    print(f"\nResultados completos exportados para '{caminho_csv}'.")

    # Gerar gráfico comparativo da curva de degradação
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Plot 1: Amplicons (300-1000 pb) sob ruído
    for nome_mod in modelos.keys():
        sub = df_stress[df_stress["Modelo"] == nome_mod]
        axes[0].plot(sub["Taxa"] * 100, sub["Acc 300-1000 pb (%)"], marker='o', linewidth=2.2, label=nome_mod)
    axes[0].set_title("Resiliência a Ruído em Amplicons (300 - 1.000 pb)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Taxa de Erro NGS / Ruído Estocástico (%)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Acurácia (%)", fontsize=11, fontweight='bold')
    axes[0].set_xticks([0, 4, 8, 10, 15])
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=9)

    # Plot 2: Reads Curtas (<300 pb) sob ruído
    for nome_mod in modelos.keys():
        sub = df_stress[df_stress["Modelo"] == nome_mod]
        axes[1].plot(sub["Taxa"] * 100, sub["Acc < 300 pb (%)"], marker='s', linewidth=2.2, label=nome_mod)
    axes[1].set_title("Resiliência a Ruído em Reads Ultra-Curtas (< 300 pb)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Taxa de Erro NGS / Ruído Estocástico (%)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Acurácia (%)", fontsize=11, fontweight='bold')
    axes[1].set_xticks([0, 4, 8, 10, 15])
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=9)

    os.makedirs(DIR_GRAFICOS, exist_ok=True)
    caminho_png = os.path.join(DIR_GRAFICOS, "curva_degradacao_ruido_extremo.png")
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=300)
    print(f"Gráfico comparativo salvo em '{caminho_png}'.")

    print("\n" + "=" * 85)
    print("TABELA RESUMO CONSOLIDADA: DEGRADAÇÃO SOB RUÍDO CRESCENTE")
    print("=" * 85)
    piv_amp = df_stress.pivot(index="Ruído", columns="Modelo", values="Acc 300-1000 pb (%)")
    print("\n--- Acurácia em Amplicons (300 - 1.000 pb) ---")
    print(piv_amp)

    piv_curto = df_stress.pivot(index="Ruído", columns="Modelo", values="Acc < 300 pb (%)")
    print("\n--- Acurácia em Reads Ultra-Curtas (< 300 pb) ---")
    print(piv_curto)

    piv_glob = df_stress.pivot(index="Ruído", columns="Modelo", values="Acc Global (%)")
    print("\n--- Acurácia Global Geral ---")
    print(piv_glob)


if __name__ == '__main__':
    main()

