import os
import sys
import itertools
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXP_DIR)
for _p in (ROOT_DIR, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PMC import carregar_fasta, deduplicar_genomas_estrito, fatiar_genoma, ARQUIVOS_SOROTIPOS, DIR_DADOS_TESTES, DIR_RESULTADOS
from analisar_teste_aleatorio import parse_header_organism

BASES = ['A', 'C', 'G', 'T']

def build_kmer_tools(k=6):
    kmers = [''.join(p) for p in itertools.product(BASES, repeat=k)]
    k_idx = {km: i for i, km in enumerate(kmers)}
    return k_idx, len(kmers)

def extract_vec(seq, k, k_idx, num_kmers):
    seq_clean = seq.upper().replace('U', 'T')
    total = len(seq_clean) - k + 1
    if total <= 0:
        return np.zeros(num_kmers, dtype=np.float32)
    counts = Counter(seq_clean[i:i+k] for i in range(total))
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

def main():
    print("=" * 80)
    print("BENCHMARK DE ARQUITETURAS DE REDES NEURAIS PARA K=6 (4.096 ENTRADAS)")
    print("=" * 80)

    # 1. Carregar genomas
    todos_genomas = []
    for arq, sorotipo in ARQUIVOS_SOROTIPOS.items():
        registros, _ = carregar_fasta(arq, sorotipo)
        todos_genomas.extend(registros)

    # Deduplicar estritamente (removendo duplicatas e conflitos inter-sorotipo)
    unicos, _, _ = deduplicar_genomas_estrito(todos_genomas)

    # Split a nível de genoma (80/20)
    from sklearn.model_selection import train_test_split
    y_gen = [g['sorotipo'] for g in unicos]
    train_gen, test_gen = train_test_split(unicos, test_size=0.20, random_state=42, stratify=y_gen)

    # Fatiar genomas em fragmentos de 2000 pb (passo 1000 pb)
    frags_train = []
    for g in train_gen:
        frags_train.extend(fatiar_genoma(g['seq'], g['id'], g['sorotipo'], 2000, 1000))

    frags_test = []
    for g in test_gen:
        frags_test.extend(fatiar_genoma(g['seq'], g['id'], g['sorotipo'], 2000, 1000))

    print(f"Genomas Treino: {len(train_gen)} | Genomas Teste: {len(test_gen)}")
    print(f"Fragmentos Treino: {len(frags_train)} | Fragmentos Teste: {len(frags_test)}")

    y_tr = np.array([f['sorotipo'] for f in frags_train])
    y_te = np.array([f['sorotipo'] for f in frags_test])

    # Extrair características K=6 uma única vez
    k_idx, num_kmers = build_kmer_tools(k=6)
    t0 = time.time()
    X_tr = np.array([extract_vec(f['seq'], 6, k_idx, num_kmers) for f in frags_train])
    X_te = np.array([extract_vec(f['seq'], 6, k_idx, num_kmers) for f in frags_test])
    print(f"Extração K=6 concluída em {time.time()-t0:.1f}s")

    # Carregar teste_aleatorio.fasta
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
    X_aleat = np.array([extract_vec(s, 6, k_idx, num_kmers) for _, _, s in aleatorio_validos])

    mask_curto = lens_aleat < 300
    mask_amplicon = (lens_aleat >= 300) & (lens_aleat < 1000)

    # Arquiteturas para testar
    arquiteturas = [
        (256, 128),
        (512, 128),
        (512, 256),
        (512, 256, 128),
        (1024, 256),
        (256, 64),
    ]

    resultados = []

    print("\n" + "=" * 80)
    print("INICIANDO TREINAMENTO E AVALIAÇÃO COMPARATIVA DAS ARQUITETURAS")
    print("=" * 80)

    for arch in arquiteturas:
        arch_str = "-".join(map(str, arch))
        print(f"\n>> Testando Arquitetura: [{arch_str}] ...")

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pmc', MLPClassifier(
                hidden_layer_sizes=arch,
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=50,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=42,
            ))
        ])

        t_inicio = time.time()
        pipe.fit(X_tr, y_tr)
        t_treino = time.time() - t_inicio
        epocas = pipe.named_steps['pmc'].n_iter_

        # 1. Teste Cego 2000 pb
        preds_te = pipe.predict(X_te)
        acc_te = accuracy_score(y_te, preds_te) * 100
        f1_te = f1_score(y_te, preds_te, average='macro') * 100

        # 2. teste_aleatorio.fasta
        preds_aleat = pipe.predict(X_aleat)
        acc_aleat = accuracy_score(y_aleat, preds_aleat) * 100
        f1_aleat = f1_score(y_aleat, preds_aleat, average='macro') * 100

        # 3. Fragmentos ultra-curtos (<300 pb)
        acc_curto = accuracy_score(y_aleat[mask_curto], preds_aleat[mask_curto]) * 100

        # 4. Amplicons (300-1000 pb)
        acc_amplicon = accuracy_score(y_aleat[mask_amplicon], preds_aleat[mask_amplicon]) * 100

        resultado = {
            "Arquitetura": arch_str,
            "Camadas": arch,
            "Epocas": epocas,
            "Tempo (s)": round(t_treino, 1),
            "Acc Teste Cego (2000 pb)": round(acc_te, 2),
            "F1 Teste Cego": round(f1_te, 2),
            "Acc Aleatorio (Global)": round(acc_aleat, 2),
            "F1 Aleatorio": round(f1_aleat, 2),
            "Acc < 300 pb": round(acc_curto, 2),
            "Acc 300-1000 pb": round(acc_amplicon, 2),
        }
        resultados.append(resultado)
        print(f"   Concluído em {t_treino:.1f}s ({epocas} épocas)")
        print(f"   -> Teste Cego (2000 pb): {acc_te:.2f}% | Aleatório Global: {acc_aleat:.2f}%")
        print(f"   -> < 300 pb: {acc_curto:.2f}% | 300-1000 pb: {acc_amplicon:.2f}%")

    dir_bench = os.path.join(DIR_RESULTADOS, "benchmarks")
    os.makedirs(dir_bench, exist_ok=True)
    caminho_csv = os.path.join(dir_bench, "benchmark_arquiteturas_k6.csv")
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(caminho_csv, index=False)
    print("\n" + "=" * 80)
    print("TABELA RESUMO FINAL:")
    print("=" * 80)
    print(df_res.to_string(index=False))

if __name__ == '__main__':
    main()

