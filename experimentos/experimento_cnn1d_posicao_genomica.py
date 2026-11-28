"""
================================================================================
EXPERIMENTO CIENTÍFICO: CNN 1D E INFLUÊNCIA DA POSIÇÃO GENÔMICA NO RESULTADO
1. Implementação em NumPy puro de duas arquiteturas de Redes Neurais
   Convolucionais Unidimensionais (CNN 1D):
   - CNN 1D Global (Conv1D + Global Average & Max Pooling + Inverted Dropout):
     Captura motivos locais em sub-janelas mantendo invariância à translação.
   - CNN 1D Posicional (Conv1D + Spatial Positional Pooling + Inverted Dropout):
     Preserva a ordem espacial 5' -> 3' das sub-janelas para testar se a
     dependência posicional rígida ajuda ou prejudica em fragmentos avulsos.
2. Varredura Posicional ao Longo do Genoma do DENV (0 a 10.700 pb):
   Avalia como a posição de origem do fragmento (genes estruturais C, prM, E vs
   não-estruturais NS1, NS2, NS3, NS4 e a polimerase conservada NS5) influencia
   a acurácia da CNN 1D, do PMC V3.0 e da Regressão Logística.
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

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
from avaliar_validacao_externa import parse_header_organism

BASES = ["A", "C", "G", "T"]


def construir_indice_kmers(k: int):
    kmers = ["".join(p) for p in itertools.product(BASES, repeat=k)]
    k_idx = {km: i for i, km in enumerate(kmers)}
    return k_idx, len(kmers)


def extrair_vetor_kmer(seq: str, k: int, k_idx: dict, num_kmers: int) -> np.ndarray:
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


def extrair_mapa_espacial_1d(
    seq: str,
    k: int,
    k_idx: dict,
    num_kmers: int,
    n_segmentos: int = 4,
) -> np.ndarray:
    """
    Divide a sequência ao longo do eixo 5' -> 3' em `n_segmentos` sub-janelas
    espaciais contíguas (com leve sobreposição de borda de k-1 bases) e extrai
    o perfil local de K-mers de cada posição espacial t in {0, ..., n_segmentos-1}.
    Retorna um tensor (n_segmentos, num_kmers) achatado em (n_segmentos * num_kmers,)
    para compatibilidade com StandardScaler / Pipeline, que a CNN 1D remodela
    internamente para (N, n_segmentos, num_kmers).
    """
    seq_clean = seq.upper().replace("U", "T")
    comp = len(seq_clean)
    if comp < k * 2:
        v = extrair_vetor_kmer(seq_clean, k, k_idx, num_kmers)
        return np.tile(v, (n_segmentos, 1)).astype(np.float32)

    passo = comp / n_segmentos
    mapa = np.zeros((n_segmentos, num_kmers), dtype=np.float32)
    for s in range(n_segmentos):
        ini = int(round(s * passo))
        fim = min(comp, int(round((s + 1) * passo)) + (k - 1))
        sub_seq = seq_clean[ini:fim]
        if len(sub_seq) < k:
            sub_seq = seq_clean
        mapa[s] = extrair_vetor_kmer(sub_seq, k, k_idx, num_kmers)

    return mapa


# ==============================================================================
# IMPLEMENTAÇÃO DE CNN 1D EM NUMPY PURO (CONV1D + GLOBAL OU POSITIONAL POOLING)
# ==============================================================================


class Conv1DGenomicClassifier(BaseEstimator, ClassifierMixin):
    """
    Rede Neural Convolucional 1D (CNN 1D) em NumPy puro para sequências genômicas:
    - Entrada: Mapa espacial 1D X de formato (N, T, D), onde T é o número de
      segmentos espaciais ao longo da fita 5' -> 3' e D = 4^K é o número de canais.
    - Camada Convolucional 1D (kernel_size=1):
      Aplica F filtros convolucionais pontuais (1x1) com pesos compartilhados
      W_conv (D, F) sobre cada um dos T segmentos espaciais de forma independente,
      seguidos de ativação ReLU.
    - Dois Modos de Pooling para testar a influência da posição espacial:
      1. modo_pooling='global' (Invariante à Posição / Translação):
         Combina Global Average Pooling (média ao longo de t=1..T) e Global Max
         Pooling (pico máximo de ativação local em qualquer posição t=1..T),
         gerando um vetor de 2*F características invariantes a onde o fragmento
         foi recortado.
      2. modo_pooling='posicional' (Sensível à Posição 5' -> 3'):
         Achata (Flatten) as ativações espaciais [H[:, 0, :], ..., H[:, T-1, :]]
         mantendo pesos exclusivos na camada de saída para o início (5'), meio e
         fim (3') da sequência.
    - Inverted Dropout (p=0.3) + Regularização L2 (alpha=0.01) + Otimizador Adam.
    """

    def __init__(
        self,
        n_filtros: int = 32,
        modo_pooling: str = "global",  # 'global' ou 'posicional'
        dropout_rate: float = 0.3,
        alpha: float = 0.01,
        learning_rate: float = 0.001,
        max_iter: int = 25,
        batch_size: int = 128,
        random_state: int = 42,
    ):
        self.n_filtros = n_filtros
        self.modo_pooling = modo_pooling
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state

    def _preparar_entrada(self, X_3d: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        N, T, D = X_3d.shape
        X_flat = X_3d.reshape(N * T, D)
        if fit_scaler:
            self.scaler_canais_ = StandardScaler()
            X_norm = self.scaler_canais_.fit_transform(X_flat)
        else:
            X_norm = self.scaler_canais_.transform(X_flat)
        return X_norm.reshape(N, T, D).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        num_classes = len(self.classes_)

        X_norm = self._preparar_entrada(X, fit_scaler=True)
        num_samples, T, D = X_norm.shape
        F = self.n_filtros

        # Pesos do banco de filtros Conv1D (compartilhados entre todas as posições t=1..T)
        limit_conv = np.sqrt(2.0 / D)
        self.W_conv_ = rng.normal(0, limit_conv, size=(D, F)).astype(np.float32)
        self.b_conv_ = np.zeros(F, dtype=np.float32)

        # Dimensão pós-pooling
        dim_pool = (2 * F) if self.modo_pooling == "global" else (T * F)
        limit_out = np.sqrt(2.0 / dim_pool)
        self.W_out_ = rng.normal(0, limit_out, size=(dim_pool, num_classes)).astype(np.float32)
        self.b_out_ = np.zeros(num_classes, dtype=np.float32)

        # Momentos do Adam
        m_wc, v_wc = np.zeros_like(self.W_conv_), np.zeros_like(self.W_conv_)
        m_bc, v_bc = np.zeros_like(self.b_conv_), np.zeros_like(self.b_conv_)
        m_wo, v_wo = np.zeros_like(self.W_out_), np.zeros_like(self.W_out_)
        m_bo, v_bo = np.zeros_like(self.b_out_), np.zeros_like(self.b_out_)

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        num_batches = int(np.ceil(num_samples / self.batch_size))
        keep_prob = 1.0 - self.dropout_rate

        for _ in range(self.max_iter):
            perm = rng.permutation(num_samples)
            X_shuf = X_norm[perm]
            y_shuf = y_idx[perm]

            for b in range(num_batches):
                step += 1
                ini = b * self.batch_size
                fim = min(ini + self.batch_size, num_samples)
                xb = X_shuf[ini:fim]  # (m, T, D)
                yb = y_shuf[ini:fim]
                m = len(yb)

                # 1. Forward Pass Conv1D ao longo das T posições espaciais
                # Z_conv: (m, T, F)
                Z_conv = np.einsum("mtd,df->mtf", xb, self.W_conv_) + self.b_conv_
                H_conv = np.maximum(0.0, Z_conv)  # ReLU

                # 2. Camada de Pooling (Global Invariante vs Posicional 5'->3')
                if self.modo_pooling == "global":
                    avg_pool = np.mean(H_conv, axis=1)  # (m, F)
                    max_idx = np.argmax(H_conv, axis=1)  # (m, F)
                    max_pool = np.max(H_conv, axis=1)   # (m, F)
                    feat_pool = np.hstack([avg_pool, max_pool])  # (m, 2*F)
                else:
                    feat_pool = H_conv.reshape(m, T * F)  # (m, T*F)

                # 3. Inverted Dropout
                if self.dropout_rate > 0:
                    mask = (rng.random(feat_pool.shape) < keep_prob).astype(np.float32) / keep_prob
                    feat_drop = feat_pool * mask
                else:
                    mask = None
                    feat_drop = feat_pool

                # 4. Camada de Saída Softmax
                logits = feat_drop @ self.W_out_ + self.b_out_
                exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

                # 5. Backward Pass Analítico
                dlogits = probs.copy()
                dlogits[np.arange(m), yb] -= 1.0
                dlogits /= m

                grad_wo = feat_drop.T @ dlogits + self.alpha * self.W_out_
                grad_bo = np.sum(dlogits, axis=0)

                dfeat = dlogits @ self.W_out_.T
                if mask is not None:
                    dfeat *= mask

                if self.modo_pooling == "global":
                    d_avg = dfeat[:, :F]
                    d_max = dfeat[:, F:]
                    # Gradiente do Global Average Pooling distribuído igualmente por T
                    dH = np.repeat((d_avg / T)[:, np.newaxis, :], T, axis=1)
                    # Gradiente do Global Max Pooling roteado para a janela t vencedora
                    m_grid = np.arange(m)[:, np.newaxis]
                    f_grid = np.arange(F)[np.newaxis, :]
                    dH[m_grid, max_idx, f_grid] += d_max
                else:
                    dH = dfeat.reshape(m, T, F)

                dZ_conv = dH * (Z_conv > 0)
                grad_wc = np.einsum("mtd,mtf->df", xb, dZ_conv) + self.alpha * self.W_conv_
                grad_bc = np.sum(dZ_conv, axis=(0, 1))

                # 6. Atualização Adam
                for param, grad, mp, vp in [
                    (self.W_conv_, grad_wc, m_wc, v_wc),
                    (self.b_conv_, grad_bc, m_bc, v_bc),
                    (self.W_out_, grad_wo, m_wo, v_wo),
                    (self.b_out_, grad_bo, m_bo, v_bo),
                ]:
                    mp[:] = beta1 * mp + (1.0 - beta1) * grad
                    vp[:] = beta2 * vp + (1.0 - beta2) * (grad ** 2)
                    m_hat = mp / (1.0 - beta1 ** step)
                    v_hat = vp / (1.0 - beta2 ** step)
                    param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_norm = self._preparar_entrada(X, fit_scaler=False)
        m, T, _ = X_norm.shape
        Z_conv = np.einsum("mtd,df->mtf", X_norm, self.W_conv_) + self.b_conv_
        H_conv = np.maximum(0.0, Z_conv)

        if self.modo_pooling == "global":
            avg_pool = np.mean(H_conv, axis=1)
            max_pool = np.max(H_conv, axis=1)
            feat_pool = np.hstack([avg_pool, max_pool])
        else:
            feat_pool = H_conv.reshape(m, T * self.n_filtros)

        logits = feat_pool @ self.W_out_ + self.b_out_
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def main():
    print("=" * 95)
    print("EXPERIMENTO CIENTÍFICO: CNN 1D (GLOBAL vs POSICIONAL) E INFLUÊNCIA DA POSIÇÃO GENÔMICA")
    print("=" * 95)

    # 1. Carregar e deduplicar genomas completos
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

    # 2. Carregar teste_aleatorio.fasta
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

    seqs_ruido_4 = [
        aplicar_ruido_estocastico(s, taxa_erro=0.04, random_state=42 + i) if comp < 1000 else s
        for i, (_, comp, s) in enumerate(aleatorio_validos)
    ]
    seqs_ruido_10 = [
        aplicar_ruido_estocastico(s, taxa_erro=0.10, random_state=42 + i) if comp < 1000 else s
        for i, (_, comp, s) in enumerate(aleatorio_validos)
    ]

    mask_curto = lens_aleat < 300
    mask_amplicon = (lens_aleat >= 300) & (lens_aleat < 1000)

    # Extração K=6 Global e Espacial 1D (T=4 sub-janelas ao longo de 5' -> 3')
    k = 6
    n_seg = 4
    k_idx, num_kmers = construir_indice_kmers(k)

    print(f"\n[3/4] Extraindo mapas espaciais 1D ({n_seg} segmentos x {num_kmers:,} canais K=6)...")
    t0 = time.time()
    X_tr_1d = np.array([extrair_mapa_espacial_1d(f["seq"], k, k_idx, num_kmers, n_seg) for f in frags_train])
    X_te_1d = np.array([extrair_mapa_espacial_1d(f["seq"], k, k_idx, num_kmers, n_seg) for f in frags_test])
    X_al_limpo_1d = np.array([extrair_mapa_espacial_1d(s, k, k_idx, num_kmers, n_seg) for s in seqs_limpas])
    X_al_r4_1d = np.array([extrair_mapa_espacial_1d(s, k, k_idx, num_kmers, n_seg) for s in seqs_ruido_4])
    X_al_r10_1d = np.array([extrair_mapa_espacial_1d(s, k, k_idx, num_kmers, n_seg) for s in seqs_ruido_10])

    # Vetores globais (para PMC V3.0 e Regressão Logística)
    X_tr_glob = np.array([extrair_vetor_kmer(f["seq"], k, k_idx, num_kmers) for f in frags_train])
    X_te_glob = np.array([extrair_vetor_kmer(f["seq"], k, k_idx, num_kmers) for f in frags_test])
    X_al_limpo_glob = np.array([extrair_vetor_kmer(s, k, k_idx, num_kmers) for s in seqs_limpas])
    X_al_r4_glob = np.array([extrair_vetor_kmer(s, k, k_idx, num_kmers) for s in seqs_ruido_4])
    X_al_r10_glob = np.array([extrair_vetor_kmer(s, k, k_idx, num_kmers) for s in seqs_ruido_10])
    print(f"Extração 1D e Global concluída em {time.time() - t0:.1f}s.")

    # Treinar os 4 modelos principais
    modelos = [
        (
            "CNN 1D Posicional (Conv1D + Spatial Flatten 5'->3')",
            "CNN 1D Posicional",
            Conv1DGenomicClassifier(n_filtros=32, modo_pooling="posicional", dropout_rate=0.3, alpha=0.01, max_iter=25, random_state=42),
            True,
        ),
        (
            "CNN 1D Global (Conv1D + Global Avg/Max Pool)",
            "CNN 1D Global",
            Conv1DGenomicClassifier(n_filtros=32, modo_pooling="global", dropout_rate=0.3, alpha=0.01, max_iter=25, random_state=42),
            True,
        ),
        (
            "PMC Enxuto V3.0 (32,) + Dropout (p=0.3)",
            "PMC V3.0 (32,)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LeanMLPDropout(hidden_layers=(32,), dropout_rate=0.3, alpha=0.01, max_iter=25, random_state=42)),
            ]),
            False,
        ),
        (
            "Regressão Logística (L2, C=1.0)",
            "Reg. Logística",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=300, C=1.0, random_state=42, solver="lbfgs")),
            ]),
            False,
        ),
    ]

    resultados_modelos = []
    modelos_treinados = {}

    print("\n" + "-" * 95)
    print(">> PARTE A: COMPARATIVO DE ARQUITETURAS (CNN 1D POSICIONAL vs CNN 1D GLOBAL vs PMC V3.0 vs RL)")
    print("-" * 95)

    for nome_longo, nome_curto, mod, usa_1d in modelos:
        t_ini = time.time()
        if usa_1d:
            mod.fit(X_tr_1d, y_tr)
            p_te = mod.predict(X_te_1d)
            p_limpo = mod.predict(X_al_limpo_1d)
            p_r4 = mod.predict(X_al_r4_1d)
            p_r10 = mod.predict(X_al_r10_1d)
        else:
            mod.fit(X_tr_glob, y_tr)
            p_te = mod.predict(X_te_glob)
            p_limpo = mod.predict(X_al_limpo_glob)
            p_r4 = mod.predict(X_al_r4_glob)
            p_r10 = mod.predict(X_al_r10_glob)
        t_treino = time.time() - t_ini
        modelos_treinados[nome_curto] = (mod, usa_1d)

        acc_te = accuracy_score(y_te, p_te) * 100
        acc_glob = accuracy_score(y_aleat, p_limpo) * 100
        f1_glob = f1_score(y_aleat, p_limpo, average="macro") * 100
        acc_curto = accuracy_score(y_aleat[mask_curto], p_limpo[mask_curto]) * 100
        acc_amp = accuracy_score(y_aleat[mask_amplicon], p_limpo[mask_amplicon]) * 100
        acc_amp_r4 = accuracy_score(y_aleat[mask_amplicon], p_r4[mask_amplicon]) * 100
        acc_amp_r10 = accuracy_score(y_aleat[mask_amplicon], p_r10[mask_amplicon]) * 100

        resultados_modelos.append({
            "Modelo": nome_longo,
            "Nome Curto": nome_curto,
            "Tempo Treino (s)": round(t_treino, 2),
            "Teste Cego 2000pb (%)": round(acc_te, 2),
            "Limpo Global (%)": round(acc_glob, 2),
            "Limpo Macro-F1 (%)": round(f1_glob, 2),
            "Limpo <300pb (%)": round(acc_curto, 2),
            "Limpo 300-1000pb (%)": round(acc_amp, 2),
            "Ruido 4% 300-1000pb (%)": round(acc_amp_r4, 2),
            "Ruido 10% 300-1000pb (%)": round(acc_amp_r10, 2),
        })

        print(
            f"   [{nome_longo:<50}] ({t_treino:4.1f}s) | "
            f"2000pb: {acc_te:6.2f}% | Global: {acc_glob:6.2f}% | "
            f"<300pb: {acc_curto:5.2f}% | 300-1000pb: {acc_amp:5.2f}% "
            f"(R4%: {acc_amp_r4:5.2f}% | R10%: {acc_amp_r10:5.2f}%)"
        )

    # ==========================================================================
    # PARTE B: VARREDURA POSICIONAL AO LONGO DO GENOMA DO DENV (0 A 10.500 pb)
    # ==========================================================================
    print("\n" + "-" * 95)
    print(">> PARTE B: INFLUÊNCIA DA POSIÇÃO DO GENOMA (AMPLICONS DE 500 pb EM CADA GENE DO VÍRUS)")
    print("Avaliando os 622 genomas de teste cego fatiados em cada coordenada funcional (5' -> 3')")
    print("-" * 95)

    regioes_funcionais = [
        ("5'UTR / C / prM", 0, 935, "Estrutural (Capsídeo/Membrana)"),
        ("E (Domínio I/II)", 936, 1680, "Estrutural (Envelope N-term)"),
        ("E (Domínio III)", 1681, 2420, "Estrutural (Envelope Receptor)"),
        ("NS1", 2421, 3476, "Não-Estrutural (Glicoproteína)"),
        ("NS2A / NS2B", 3477, 4520, "Não-Estrutural (Cofator Protease)"),
        ("NS3 (Protease)", 4521, 5400, "Não-Estrutural (Serina Protease)"),
        ("NS3 (Helicase)", 5401, 6374, "Não-Estrutural (RNA Helicase)"),
        ("NS4A / 2K / NS4B", 6375, 7637, "Não-Estrutural (Complexo Replicação)"),
        ("NS5 (MTase)", 7638, 8900, "Não-Estrutural (Metiltransferase)"),
        ("NS5 (RdRp) / 3'UTR", 8901, 10400, "Não-Estrutural (RNA Polimerase - Mais Conservada)"),
    ]

    registros_posicao = []
    tam_amplicon = 500  # Simula um amplicon diagnóstico típico de RT-PCR (500 pb)

    for nome_reg, ini_reg, fim_reg, papel_bio in regioes_funcionais:
        centro = (ini_reg + fim_reg) // 2
        ini_corte = max(0, centro - tam_amplicon // 2)
        fim_corte = ini_corte + tam_amplicon

        seqs_reg = []
        y_reg = []
        for g in test_gen:
            if len(g["seq"]) >= fim_corte:
                seqs_reg.append(g["seq"][ini_corte:fim_corte])
                y_reg.append(g["sorotipo"])

        y_reg = np.array(y_reg)
        X_reg_glob = np.array([extrair_vetor_kmer(s, k, k_idx, num_kmers) for s in seqs_reg])
        X_reg_1d = np.array([extrair_mapa_espacial_1d(s, k, k_idx, num_kmers, n_seg) for s in seqs_reg])

        # Também testar com 4% de ruído NGS em cada região do genoma
        seqs_reg_ruido = [aplicar_ruido_estocastico(s, taxa_erro=0.04, random_state=42 + i) for i, s in enumerate(seqs_reg)]
        X_reg_r4_glob = np.array([extrair_vetor_kmer(s, k, k_idx, num_kmers) for s in seqs_reg_ruido])
        X_reg_r4_1d = np.array([extrair_mapa_espacial_1d(s, k, k_idx, num_kmers, n_seg) for s in seqs_reg_ruido])

        accs_limpo = {}
        accs_r4 = {}
        for nome_curto, (mod, usa_1d) in modelos_treinados.items():
            p_l = mod.predict(X_reg_1d if usa_1d else X_reg_glob)
            p_r = mod.predict(X_reg_r4_1d if usa_1d else X_reg_r4_glob)
            accs_limpo[nome_curto] = round(accuracy_score(y_reg, p_l) * 100, 2)
            accs_r4[nome_curto] = round(accuracy_score(y_reg, p_r) * 100, 2)

        reg_pos = {
            "Regiao Genômica": nome_reg,
            "Coordenadas Amplicon (pb)": f"{ini_corte}-{fim_corte}",
            "Centro (pb)": centro,
            "Função Biológica": papel_bio,
            "PMC V3.0 (Limpo %)": accs_limpo["PMC V3.0 (32,)"],
            "CNN 1D Global (Limpo %)": accs_limpo["CNN 1D Global"],
            "CNN 1D Posicional (Limpo %)": accs_limpo["CNN 1D Posicional"],
            "Reg. Logística (Limpo %)": accs_limpo["Reg. Logística"],
            "PMC V3.0 (Ruído 4% %)": accs_r4["PMC V3.0 (32,)"],
            "CNN 1D Global (Ruído 4% %)": accs_r4["CNN 1D Global"],
            "CNN 1D Posicional (Ruído 4% %)": accs_r4["CNN 1D Posicional"],
            "Reg. Logística (Ruído 4% %)": accs_r4["Reg. Logística"],
        }
        registros_posicao.append(reg_pos)

        print(
            f"   [{nome_reg:<18} | {ini_corte:4d}-{fim_corte:5d} pb] "
            f"Limpo -> PMC: {accs_limpo['PMC V3.0 (32,)']:6.2f}% | CNN-Glob: {accs_limpo['CNN 1D Global']:6.2f}% | "
            f"CNN-Pos: {accs_limpo['CNN 1D Posicional']:6.2f}% | RL: {accs_limpo['Reg. Logística']:6.2f}% || "
            f"Ruído 4% -> PMC: {accs_r4['PMC V3.0 (32,)']:6.2f}% | CNN-Glob: {accs_r4['CNN 1D Global']:6.2f}% | RL: {accs_r4['Reg. Logística']:6.2f}%"
        )

    # 4. Salvar Tabelas CSV e Gerar Painel Gráfico
    print("\n[4/4] Salvando relatórios CSV e gerando gráfico científico da varredura posicional...")
    dir_bench = os.path.join(DIR_RESULTADOS, "benchmarks")
    os.makedirs(dir_bench, exist_ok=True)

    df_mod = pd.DataFrame(resultados_modelos)
    df_pos = pd.DataFrame(registros_posicao)

    caminho_csv_mod = os.path.join(dir_bench, "benchmark_cnn1d_comparativo.csv")
    caminho_csv_pos = os.path.join(dir_bench, "benchmark_posicao_genomica_cnn1d.csv")
    df_mod.to_csv(caminho_csv_mod, index=False)
    df_pos.to_csv(caminho_csv_pos, index=False)
    print(f"Tabelas salvas em:\n - '{caminho_csv_mod}'\n - '{caminho_csv_pos}'")

    # Gerar Gráfico de Alta Resolução (2 Painéis)
    os.makedirs(DIR_GRAFICOS, exist_ok=True)
    caminho_png = os.path.join(DIR_GRAFICOS, "comparativo_cnn1d_posicao_genomica.png")

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6))

    # Painel A: Perfil de Acurácia ao Longo da Posição do Genoma (5' -> 3') em Amplicons de 500 pb (Ruído 4%)
    x_genes = np.arange(len(df_pos))
    labels_genes = df_pos["Regiao Genômica"].tolist()

    axes[0].plot(x_genes, df_pos["PMC V3.0 (Ruído 4% %)"], marker="s", color="#1f77b4", linewidth=2.4, markersize=7, label="PMC Enxuto V3.0 (32,) + Dropout")
    axes[0].plot(x_genes, df_pos["CNN 1D Global (Ruído 4% %)"], marker="o", color="#2ca02c", linewidth=2.2, markersize=7, linestyle="--", label="CNN 1D Global (Avg+Max Pool)")
    axes[0].plot(x_genes, df_pos["CNN 1D Posicional (Ruído 4% %)"], marker="^", color="#9467bd", linewidth=2.0, markersize=6, linestyle="-.", label="CNN 1D Posicional (Spatial 5'->3')")
    axes[0].plot(x_genes, df_pos["Reg. Logística (Ruído 4% %)"], marker="d", color="#d62728", linewidth=2.0, markersize=6, linestyle=":", label="Regressão Logística (L2)")

    axes[0].set_title("A) Influência da Posição Genômica (Amplicons de 500 pb com 4% Ruído NGS)", fontsize=11.5, fontweight="bold")
    axes[0].set_xticks(x_genes)
    axes[0].set_xticklabels(labels_genes, rotation=35, ha="right", fontweight="bold", fontsize=9)
    axes[0].set_ylabel("Acurácia por Região Gênica (%)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(82, 101)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower left", fontsize=8.8)

    # Painel B: Comparativo Geral em Amplicons Reais Heterogêneos (teste_aleatorio.fasta: 300-1.000 pb)
    nomes_m = df_mod["Nome Curto"].tolist()
    y_pos = np.arange(len(nomes_m))
    h = 0.25

    axes[1].barh(y_pos + h, df_mod["Limpo 300-1000pb (%)"], height=h, color="#1f77b4", alpha=0.9, label="Amplicons Limpos (0% Ruído)")
    axes[1].barh(y_pos, df_mod["Ruido 4% 300-1000pb (%)"], height=h, color="#2ca02c", alpha=0.9, label="Amplicons Ruído NGS 4%")
    axes[1].barh(y_pos - h, df_mod["Ruido 10% 300-1000pb (%)"], height=h, color="#d62728", alpha=0.9, label="Amplicons Ruído Severo 10%")

    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(nomes_m, fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Acurácia em Amplicons 300-1.000 pb (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("B) CNN 1D (Posicional vs Global) vs PMC V3.0 em Amplicons Reais", fontsize=11.5, fontweight="bold")
    axes[1].set_xlim(70, 96)
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right", fontsize=9)

    plt.suptitle("Análise Espacial e Convolucional 1D: Efeito da Posição no Genoma DENV (5' -> 3') e Invariância à Translação", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em '{caminho_png}'.")


if __name__ == "__main__":
    main()
