import os
import sys
import itertools
import time
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional, List, Tuple, Dict, Any

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXP_DIR)
for _p in (ROOT_DIR, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PMC import carregar_fasta, fatiar_genoma, ARQUIVOS_SOROTIPOS, DIR_DADOS_TESTES, DIR_RESULTADOS
from testar_arquiteturas_k6 import build_kmer_tools, extract_vec
from analisar_teste_aleatorio import parse_header_organism

# ==============================================================================
# 1. FUNÇÃO DE RUÍDO ESTOCÁSTICO (SIMULAÇÃO DE READS BRUTAS NGS / INDELS / MUTAÇÕES)
# ==============================================================================

def aplicar_ruido_estocastico(
    seq: str,
    taxa_erro: float = 0.04,  # Média de 4% (faixa de 3% a 5%)
    prop_substituicao: float = 0.70,
    prop_insercao: float = 0.15,
    prop_delecao: float = 0.15,
    random_state: Optional[int] = None,
) -> str:
    """
    Simula ruído estocástico de sequenciamento NGS (leituras brutas antes do basecalling):
    - 70% de substituições de base (mutações pontuais)
    - 15% de inserções aleatórias (indels)
    - 15% de deleções (indels)
    """
    if not seq:
        return ""

    rng = np.random.default_rng(random_state)
    bases = ["A", "C", "G", "T"]
    nova_seq = []

    for base in seq.upper().replace("U", "T"):
        if rng.random() > taxa_erro:
            nova_seq.append(base)
        else:
            tipo_erro = rng.random()
            if tipo_erro < prop_substituicao:
                # Substituição aleatória por base diferente
                opcoes = [b for b in bases if b != base]
                nova_seq.append(rng.choice(opcoes))
            elif tipo_erro < prop_substituicao + prop_insercao:
                # Inserção
                nova_seq.append(base)
                nova_seq.append(rng.choice(bases))
            else:
                # Deleção: ignora a base atual
                pass

    return "".join(nova_seq)


# ==============================================================================
# 2. IMPLEMENTAÇÃO DO PMC ENXUTO COM DROPOUT E L2 EM NUMPY PURO (SEM PYTORCH)
# ==============================================================================

class LeanMLPDropout(BaseEstimator, ClassifierMixin):
    """
    Perceptron Multicamadas (PMC) implementado em NumPy puro:
    - Suporta Dropout estocástico com máscara invertida nas camadas ocultas.
    - Regularização L2 com penalidade alpha.
    - Otimizador Adam com mini-batches para convergência rápida e estável.
    - Compatível 100% com a API scikit-learn (Pipeline, fit, predict, predict_proba).
    """
    def __init__(
        self,
        hidden_layers: Tuple[int, ...] = (64, 16),
        dropout_rate: float = 0.3,
        alpha: float = 0.01,
        learning_rate: float = 0.001,
        max_iter: int = 25,
        batch_size: int = 128,
        random_state: int = 42,
    ):
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        num_classes = len(self.classes_)
        num_samples, input_dim = X.shape

        layer_dims = [input_dim] + list(self.hidden_layers) + [num_classes]
        self.weights_ = []
        self.biases_ = []

        # Inicialização He / Kaiming para ReLU
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            limit = np.sqrt(2.0 / fan_in)
            w = rng.normal(0, limit, size=(fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights_.append(w)
            self.biases_.append(b)

        # Momentos do Adam
        m_w = [np.zeros_like(w) for w in self.weights_]
        v_w = [np.zeros_like(w) for w in self.weights_]
        m_b = [np.zeros_like(b) for b in self.biases_]
        v_b = [np.zeros_like(b) for b in self.biases_]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        num_batches = int(np.ceil(num_samples / self.batch_size))
        keep_prob = 1.0 - self.dropout_rate

        for epoch in range(self.max_iter):
            indices = rng.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y_idx[indices]

            for b in range(num_batches):
                t += 1
                start = b * self.batch_size
                end = min(start + self.batch_size, num_samples)
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]
                m = len(yb)

                # Forward pass com Inverted Dropout
                activations = [xb]
                zs = []
                masks = []

                for l in range(len(self.weights_) - 1):
                    z = activations[-1] @ self.weights_[l] + self.biases_[l]
                    zs.append(z)
                    a = np.maximum(0, z)  # ReLU
                    if self.dropout_rate > 0:
                        mask = (rng.random(a.shape) < keep_prob).astype(np.float32) / keep_prob
                        a *= mask
                        masks.append(mask)
                    else:
                        masks.append(None)
                    activations.append(a)

                # Camada de saída (Softmax)
                z_out = activations[-1] @ self.weights_[-1] + self.biases_[-1]
                zs.append(z_out)
                exp_z = np.exp(z_out - np.max(z_out, axis=1, keepdims=True))
                probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                activations.append(probs)

                # Backward pass
                dlogits = probs.copy()
                dlogits[np.arange(m), yb] -= 1.0
                dlogits /= m

                grad_w = [None] * len(self.weights_)
                grad_b = [None] * len(self.biases_)

                grad_w[-1] = activations[-2].T @ dlogits + (self.alpha * self.weights_[-1])
                grad_b[-1] = np.sum(dlogits, axis=0)

                delta = dlogits
                for l in range(len(self.weights_) - 2, -1, -1):
                    delta = delta @ self.weights_[l + 1].T
                    if masks[l] is not None:
                        delta *= masks[l]
                    delta = delta * (zs[l] > 0)  # Derivada da ReLU

                    grad_w[l] = activations[l].T @ delta + (self.alpha * self.weights_[l])
                    grad_b[l] = np.sum(delta, axis=0)

                # Atualização Adam
                for l in range(len(self.weights_)):
                    m_w[l] = beta1 * m_w[l] + (1 - beta1) * grad_w[l]
                    v_w[l] = beta2 * v_w[l] + (1 - beta2) * (grad_w[l] ** 2)
                    m_hat_w = m_w[l] / (1 - beta1 ** t)
                    v_hat_w = v_w[l] / (1 - beta2 ** t)
                    self.weights_[l] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + eps)

                    m_b[l] = beta1 * m_b[l] + (1 - beta1) * grad_b[l]
                    v_b[l] = beta2 * v_b[l] + (1 - beta2) * (grad_b[l] ** 2)
                    m_hat_b = m_b[l] / (1 - beta1 ** t)
                    v_hat_b = v_b[l] / (1 - beta2 ** t)
                    self.biases_[l] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + eps)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        a = X
        for l in range(len(self.weights_) - 1):
            z = a @ self.weights_[l] + self.biases_[l]
            a = np.maximum(0, z)
        z_out = a @ self.weights_[-1] + self.biases_[-1]
        exp_z = np.exp(z_out - np.max(z_out, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


# ==============================================================================
# 3. BENCHMARK COMPLETO: PMC ENXUTO vs REGRESSÃO LOGÍSTICA (LIMPO E COM RUÍDO)
# ==============================================================================

def main():
    print("=" * 85)
    print("EXPERIMENTO: ARQUITETURAS ENXUTAS, DROPOUT (p=0.3), L2 (alpha=0.01) E RUÍDO NGS")
    print("=" * 85)

    # 1. Carregar genomas
    todos_genomas = []
    for arq, sorotipo in ARQUIVOS_SOROTIPOS.items():
        registros, _ = carregar_fasta(arq, sorotipo)
        todos_genomas.extend(registros)

    vistos = set()
    unicos = [g for g in todos_genomas if not (g['hash'] in vistos or vistos.add(g['hash']))]

    # Split estrito a nível de genoma (80/20)
    from sklearn.model_selection import train_test_split
    y_gen = [g['sorotipo'] for g in unicos]
    train_gen, test_gen = train_test_split(unicos, test_size=0.20, random_state=42, stratify=y_gen)

    # Fatiamento
    frags_train = []
    for g in train_gen:
        frags_train.extend(fatiar_genoma(g['seq'], g['id'], g['sorotipo'], 2000, 1000))

    frags_test = []
    for g in test_gen:
        frags_test.extend(fatiar_genoma(g['seq'], g['id'], g['sorotipo'], 2000, 1000))

    y_tr = np.array([f['sorotipo'] for f in frags_train])
    y_te = np.array([f['sorotipo'] for f in frags_test])

    k_idx, num_kmers = build_kmer_tools(6)
    print(f"\nExtraindo características K=6 (4.096) para {len(frags_train):,} fragmentos de treino...")
    t0 = time.time()
    X_tr = np.array([extract_vec(f['seq'], 6, k_idx, num_kmers) for f in frags_train])
    X_te = np.array([extract_vec(f['seq'], 6, k_idx, num_kmers) for f in frags_test])
    print(f"Extração concluída em {time.time()-t0:.1f}s.")

    # 2. Carregar teste_aleatorio.fasta
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

    # Extrair características dos dados LIMPOS
    print(f"Extraindo características K=6 para dados limpos de teste_aleatorio ({len(aleatorio_validos):,} seqs)...")
    X_aleat_limpo = np.array([extract_vec(s, 6, k_idx, num_kmers) for _, _, s in aleatorio_validos])

    # 3. Gerar dados RUIDOSOS (Simulando 3% a 5% de erro NGS bruto em fragmentos curtos)
    print("\nAplicando ruído estocástico (4% mutações/indels) em fragmentos curtos (< 1.000 pb)...")
    seqs_ruidosas = []
    for i, (lbl, comp, s) in enumerate(aleatorio_validos):
        if comp < 1000:
            # Aplica ruído estocástico na sequência
            s_ruido = aplicar_ruido_estocastico(s, taxa_erro=0.04, random_state=42 + i)
            seqs_ruidosas.append(s_ruido)
        else:
            seqs_ruidosas.append(s)

    X_aleat_ruidoso = np.array([extract_vec(s, 6, k_idx, num_kmers) for s in seqs_ruidosas])

    mask_curto = lens_aleat < 300
    mask_amplicon = (lens_aleat >= 300) & (lens_aleat < 1000)

    # 4. Definição dos Modelos para Avaliação
    modelos = {
        "Regressão Logística (L2)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=200, C=1.0, random_state=42, solver='lbfgs'))
        ]),
        "PMC (512, 128) [Grande, alpha=0.001]": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(512, 128), activation='relu', alpha=0.001,
                                 max_iter=50, early_stopping=True, random_state=42))
        ]),
        "PMC (64, 16) [Enxuto, alpha=0.01, Sem Dropout]": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(64, 16), activation='relu', alpha=0.01,
                                 max_iter=50, early_stopping=True, random_state=42))
        ]),
        "PMC (32,) [1 Camada, alpha=0.01, Sem Dropout]": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(32,), activation='relu', alpha=0.01,
                                 max_iter=50, early_stopping=True, random_state=42))
        ]),
        "PMC (64, 16) + Dropout (p=0.3) + L2 (0.01)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LeanMLPDropout(hidden_layers=(64, 16), dropout_rate=0.3, alpha=0.01, max_iter=25, random_state=42))
        ]),
        "PMC (32,) + Dropout (p=0.3) + L2 (0.01)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LeanMLPDropout(hidden_layers=(32,), dropout_rate=0.3, alpha=0.01, max_iter=25, random_state=42))
        ]),
    }

    tabela_resultados = []

    print("\n" + "=" * 85)
    print("TREINANDO MODELOS E AVALIANDO EM CENÁRIO LIMPO E RUIDOSO")
    print("=" * 85)

    for nome, pipe in modelos.items():
        print(f"\n>> Treinando: {nome} ...")
        t_ini = time.time()
        pipe.fit(X_tr, y_tr)
        t_treino = time.time() - t_ini

        # Teste Cego 2000 pb
        preds_te = pipe.predict(X_te)
        acc_te = accuracy_score(y_te, preds_te) * 100

        # Cenário 1: Limpo
        p_limpo = pipe.predict(X_aleat_limpo)
        acc_glob_limpo = accuracy_score(y_aleat, p_limpo) * 100
        acc_curto_limpo = accuracy_score(y_aleat[mask_curto], p_limpo[mask_curto]) * 100
        acc_amp_limpo = accuracy_score(y_aleat[mask_amplicon], p_limpo[mask_amplicon]) * 100

        # Cenário 2: Ruidoso (3-5% mutações e indels)
        p_ruido = pipe.predict(X_aleat_ruidoso)
        acc_glob_ruido = accuracy_score(y_aleat, p_ruido) * 100
        acc_curto_ruido = accuracy_score(y_aleat[mask_curto], p_ruido[mask_curto]) * 100
        acc_amp_ruido = accuracy_score(y_aleat[mask_amplicon], p_ruido[mask_amplicon]) * 100

        queda_curto = acc_curto_limpo - acc_curto_ruido

        tabela_resultados.append({
            "Modelo": nome,
            "Tempo (s)": round(t_treino, 1),
            "Acc Teste Cego (2000pb)": round(acc_te, 2),
            "Limpo: Global": round(acc_glob_limpo, 2),
            "Limpo: <300pb": round(acc_curto_limpo, 2),
            "Limpo: 300-1000pb": round(acc_amp_limpo, 2),
            "Ruído 4%: <300pb": round(acc_curto_ruido, 2),
            "Ruído 4%: 300-1000pb": round(acc_amp_ruido, 2),
            "Queda (<300pb)": round(queda_curto, 2),
        })

        print(f"   Treino: {t_treino:.1f}s | Teste Cego: {acc_te:.2f}%")
        print(f"   [LIMPO]  Global: {acc_glob_limpo:.2f}% | <300pb: {acc_curto_limpo:.2f}% | 300-1000pb: {acc_amp_limpo:.2f}%")
        print(f"   [RUÍDO]  Global: {acc_glob_ruido:.2f}% | <300pb: {acc_curto_ruido:.2f}% | 300-1000pb: {acc_amp_ruido:.2f}% | Queda: {queda_curto:.2f}%")

    dir_bench = os.path.join(DIR_RESULTADOS, "benchmarks")
    os.makedirs(dir_bench, exist_ok=True)
    caminho_csv = os.path.join(dir_bench, "benchmark_enxuto_dropout_ruido.csv")
    df_comp = pd.DataFrame(tabela_resultados)
    df_comp.to_csv(caminho_csv, index=False)
    print("\n" + "=" * 85)
    print("TABELA COMPARATIVA FINAL CONSOLIDADA:")
    print("=" * 85)
    print(df_comp.to_string(index=False))

if __name__ == '__main__':
    main()

