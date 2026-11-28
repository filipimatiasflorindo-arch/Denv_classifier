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
from sklearn.metrics import accuracy_score

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXP_DIR)
for _p in (ROOT_DIR, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PMC import carregar_fasta, deduplicar_genomas_estrito, fatiar_genoma, ARQUIVOS_SOROTIPOS, DIR_DADOS_TESTES
from avaliar_validacao_externa import parse_header_organism

BASES = ['A', 'C', 'G', 'T']

def build_kmer_tools(k):
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

def run_experiment():
    print("=" * 75)
    print("EXPERIMENTO COMPARATIVO: K=5 (1.024) vs K=6 (4.096)")
    print("=" * 75)

    # 1. Carregar genomas
    np.random.seed(42)
    todos_genomas = []
    for arq, sorotipo in ARQUIVOS_SOROTIPOS.items():
        registros, _ = carregar_fasta(arq, sorotipo)
        todos_genomas.extend(registros)
    
    # Deduplicar estritamente
    unicos, _, _ = deduplicar_genomas_estrito(todos_genomas)

    # Split a nível de genoma (80/20)
    from sklearn.model_selection import train_test_split
    y_gen = [g['sorotipo'] for g in unicos]
    train_gen, test_gen = train_test_split(unicos, test_size=0.20, random_state=42, stratify=y_gen)
    print(f"Genomas Treino: {len(train_gen)} | Genomas Teste: {len(test_gen)}")

    # Fatiar em 2000 pb (passo 1000 pb)
    frags_train = []
    for g in train_gen:
        frags_train.extend(fatiar_genoma(g['seq'], g['id'], g['sorotipo'], 2000, 1000))

    frags_test = []
    for g in test_gen:
        frags_test.extend(fatiar_genoma(g['seq'], g['id'], g['sorotipo'], 2000, 1000))

    print(f"Fragmentos Treino: {len(frags_train)} | Fragmentos Teste: {len(frags_test)}")
    y_tr = np.array([f['sorotipo'] for f in frags_train])
    y_te = np.array([f['sorotipo'] for f in frags_test])

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

    # Filtrar ground truth
    classes = ['DENV-1', 'DENV-2', 'DENV-3', 'DENV-4']
    aleatorio_validos = []
    for desc, s in aleatorio_recs:
        lbl, is_pat = parse_header_organism(desc)
        if lbl in classes and not is_pat and len(s) >= 6:
            aleatorio_validos.append((lbl, len(s), s))

    print(f"Sequências teste_aleatorio validadas: {len(aleatorio_validos)}")

    for k in [5, 6]:
        print("\n" + "-" * 75)
        print(f"AVALIANDO K = {k} ({4**k} características)")
        print("-" * 75)
        k_idx, num_kmers = build_kmer_tools(k)

        # Extrair features treino
        t0 = time.time()
        X_tr = np.array([extract_vec(f['seq'], k, k_idx, num_kmers) for f in frags_train])
        X_te = np.array([extract_vec(f['seq'], k, k_idx, num_kmers) for f in frags_test])
        t_ext = time.time() - t0
        print(f"Extração K={k}: {t_ext:.1f}s | Shape X_tr: {X_tr.shape}")

        # Esparsidade média
        esparsidade_tr = np.mean(X_tr == 0) * 100
        print(f"Esparsidade média no treino (2.000 pb): {esparsidade_tr:.1f}% zeros")

        # Pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pmc', MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=50,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=42,
            ))
        ])

        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        t_train = time.time() - t0
        n_epochs = pipe.named_steps['pmc'].n_iter_
        print(f"Treinamento concluído em {t_train:.1f}s ({n_epochs} épocas)")

        # Teste Cego 2.000 pb
        preds_te = pipe.predict(X_te)
        acc_te = accuracy_score(y_te, preds_te) * 100
        print(f"-> Acurácia Teste Cego (Janelas 2.000 pb): {acc_te:.2f}%")

        # Avaliação em teste_aleatorio.fasta
        X_aleat = np.array([extract_vec(s, k, k_idx, num_kmers) for _, _, s in aleatorio_validos])
        y_aleat = np.array([lbl for lbl, _, _ in aleatorio_validos])
        lens_aleat = np.array([l for _, l, _ in aleatorio_validos])
        preds_aleat = pipe.predict(X_aleat)

        acc_total = accuracy_score(y_aleat, preds_aleat) * 100
        print(f"-> Acurácia Global teste_aleatorio.fasta: {acc_total:.2f}%")

        # Estratificação (intervalos semiabertos [a, b) padronizados)
        bins = [0, 300, 1000, 5000, np.inf]
        strata = ['< 300 pb', '300-1000 pb', '1000-5000 pb', '>= 5000 pb']
        cats = pd.cut(lens_aleat, bins=bins, labels=strata, right=False)
        for st in strata:
            mask = (cats == st)
            sub_acc = accuracy_score(y_aleat[mask], preds_aleat[mask]) * 100
            sub_esp = np.mean(X_aleat[mask] == 0) * 100
            print(f"   [{st:<12}] (N={np.sum(mask):>4}) | Acurácia: {sub_acc:>6.2f}% | Esparsidade: {sub_esp:.1f}% zeros")

if __name__ == '__main__':
    run_experiment()
