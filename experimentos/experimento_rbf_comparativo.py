"""
================================================================================
EXPERIMENTO CIENTÍFICO: REDE NEURAL DE FUNÇÕES DE BASE RADIAL (RBF) vs PMC vs RL
Implementação de Rede Neural RBF Clássica (Centróides K-Means Estratificados +
Ativação Gaussiana Radial + Camada de Saída Regularizada) e Kernel RBF,
avaliando o efeito da Maldição da Dimensionalidade (K=4 vs K=6), Fragmentos
Curtos e Resiliência a Ruído Estocástico NGS (0%, 4% e 10%).
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, pairwise_distances

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


# ==============================================================================
# IMPLEMENTAÇÃO DA REDE NEURAL RBF (RADIAL BASIS FUNCTION NETWORK) EM NUMPY PURO
# ==============================================================================


class RedeNeuralRBF(BaseEstimator, ClassifierMixin):
    """
    Rede Neural de Funções de Base Radial (RBF Network) clássica (Haykin):
    1. Camada Oculta Radial:
       - Aprendizado híbrido: os M centros radiais (protótipos) são determinados
         por agrupamento K-Means estratificado por classe (garantindo cobertura
         equilibrada dos 4 sorotipos no espaço de K-mers).
       - Largura radial (sigma / gamma) calibrada automaticamente pela regra de
         espalhamento médio entre centróides (ou normalização radial L1/Softmax),
         evitando o colapso numérico de exponenciais em alta dimensão.
       - Função de ativação Gaussiana:
         phi_j(x) = exp(-gamma * ||x - mu_j||^2)
    2. Camada de Saída:
       - Pesos lineares treinados via otimização convexa (Entropia Cruzada com L2
         ou Mínimos Quadrados Regularizados / Ridge).
    """

    def __init__(
        self,
        n_centros: int = 32,
        normalizar_radial: bool = True,
        escala_gamma: float = 1.0,
        C: float = 10.0,
        random_state: int = 42,
    ):
        self.n_centros = n_centros
        self.normalizar_radial = normalizar_radial
        self.escala_gamma = escala_gamma
        self.C = C
        self.random_state = random_state

    def _calcular_ativacoes_radiais(self, X: np.ndarray) -> np.ndarray:
        """Calcula a matriz de ativações Gaussianas Phi(X) de dimensão (N, M)."""
        # Distância Euclidiana quadrática rápida: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c^T
        x_norm2 = np.sum(X ** 2, axis=1, keepdims=True)
        c_norm2 = np.sum(self.centros_ ** 2, axis=1, keepdims=True).T
        dist_sq = np.maximum(0.0, x_norm2 + c_norm2 - 2.0 * (X @ self.centros_.T))

        if self.normalizar_radial:
            # RBF Normalizada (Moody & Darken): estabiliza ativações em alta dimensão (K=6)
            # onde todas as distâncias Euclidianas crescem com D
            z = -self.gamma_ * dist_sq
            z_shift = z - np.max(z, axis=1, keepdims=True)
            phi = np.exp(z_shift)
            phi /= np.sum(phi, axis=1, keepdims=True) + 1e-12
        else:
            phi = np.exp(-self.gamma_ * dist_sq)

        return phi.astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        centros_por_classe = max(1, self.n_centros // n_classes)

        lista_centros = []
        for idx_c, c in enumerate(self.classes_):
            X_c = X[y == c]
            alocados = sum(len(c) for c in lista_centros)
            n_k = centros_por_classe if idx_c < n_classes - 1 else (self.n_centros - alocados)
            n_k = max(1, min(n_k, len(X_c)))
            if n_k == 1:
                centro_medio = np.mean(X_c, axis=0, keepdims=True)
                lista_centros.append(centro_medio)
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=n_k,
                    random_state=self.random_state + idx_c,
                    batch_size=512,
                    n_init=3,
                )
                kmeans.fit(X_c)
                lista_centros.append(kmeans.cluster_centers_)

        self.centros_ = np.vstack(lista_centros).astype(np.float32)

        # Calibração de gamma = escala_gamma / (2 * sigma^2), onde sigma^2 é a mediana das distâncias quadráticas entre centros
        dists_centros = pairwise_distances(self.centros_, metric="sqeuclidean")
        mediana_dist_sq = float(np.median(dists_centros[dists_centros > 0])) if len(self.centros_) > 1 else float(X.shape[1])
        self.gamma_ = self.escala_gamma / max(mediana_dist_sq, 1e-6)

        # Extrair matriz Phi(X) da camada oculta radial
        Phi_train = self._calcular_ativacoes_radiais(X)
        self.scaler_phi_ = StandardScaler()
        Phi_train_norm = self.scaler_phi_.fit_transform(Phi_train)

        # Camada de saída linear otimizada com L2
        self.camada_saida_ = LogisticRegression(
            C=self.C,
            max_iter=300,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.camada_saida_.fit(Phi_train_norm, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Phi = self._calcular_ativacoes_radiais(X)
        Phi_norm = self.scaler_phi_.transform(Phi)
        return self.camada_saida_.predict_proba(Phi_norm)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Phi = self._calcular_ativacoes_radiais(X)
        Phi_norm = self.scaler_phi_.transform(Phi)
        return self.camada_saida_.predict(Phi_norm)


def main():
    print("=" * 95)
    print("EXPERIMENTO COMPARATIVO: REDE NEURAL RBF & KERNEL RBF vs PMC ENXUTO V3.0 vs REGRESSÃO LOGÍSTICA")
    print("Análise de Localidade Radial (Distância Euclidiana) vs Projeção Direcional (ReLU + Dropout)")
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
        aplicar_ruido_estocastico(s, taxa_erro=0.10, random_state=1042 + i) if comp < 1000 else s
        for i, (_, comp, s) in enumerate(aleatorio_validos)
    ]

    mask_curto = lens_aleat < 300
    mask_amplicon = (lens_aleat >= 300) & (lens_aleat < 1000)
    mask_gene = (lens_aleat >= 1000) & (lens_aleat < 5000)
    mask_genoma = lens_aleat >= 5000

    # 3. Avaliar para K = 4 (Baixa Dimensão: 256) e K = 6 (Alta Dimensão: 4.096)
    print("\n[3/4] Comparando Arquiteturas RBF, Kernel RBF, Regressão Logística e PMC Enxuto em K=4 e K=6...")
    resultados = []

    for k in [4, 6]:
        k_idx, num_kmers = construir_indice_kmers(k)
        print("\n" + "-" * 95)
        print(f">> CENÁRIO K = {k} ({num_kmers:,} dimensões | Espaço {'Denso/Baixa Dimensão' if k == 4 else 'Alta Dimensão/Esparso'})")
        print("-" * 95)

        t0 = time.time()
        X_tr = np.array([extrair_vetor_kmer(f["seq"], k, k_idx, num_kmers) for f in frags_train])
        X_te = np.array([extrair_vetor_kmer(f["seq"], k, k_idx, num_kmers) for f in frags_test])
        X_al_limpo = np.array([extrair_vetor_kmer(s, k, k_idx, num_kmers) for s in seqs_limpas])
        X_al_r4 = np.array([extrair_vetor_kmer(s, k, k_idx, num_kmers) for s in seqs_ruido_4])
        X_al_r10 = np.array([extrair_vetor_kmer(s, k, k_idx, num_kmers) for s in seqs_ruido_10])
        print(f"   Extração K={k} concluída em {time.time() - t0:.1f}s.")

        modelos_k = [
            (
                "Rede Neural RBF (32 centros, K-Means)",
                "RBF Clássica (32)",
                (num_kmers * 32) + (32 * 4 + 4),
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", RedeNeuralRBF(n_centros=32, normalizar_radial=True, escala_gamma=1.0, C=10.0, random_state=42)),
                ]),
            ),
            (
                "Rede Neural RBF (128 centros, K-Means)",
                "RBF Média (128)",
                (num_kmers * 128) + (128 * 4 + 4),
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", RedeNeuralRBF(n_centros=128, normalizar_radial=True, escala_gamma=1.0, C=10.0, random_state=42)),
                ]),
            ),
            (
                "Rede Neural RBF (512 centros, K-Means)",
                "RBF Larga (512)",
                (num_kmers * 512) + (512 * 4 + 4),
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", RedeNeuralRBF(n_centros=512, normalizar_radial=True, escala_gamma=1.0, C=10.0, random_state=42)),
                ]),
            ),
            (
                "SVM Kernel RBF (Nystroem 512 componentes)",
                "SVM-RBF (512)",
                (num_kmers * 512) + (512 * 4 + 4),
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("nystroem", Nystroem(kernel="rbf", gamma=1.0 / num_kmers, n_components=512, random_state=42)),
                    ("clf", LogisticRegression(C=10.0, max_iter=300, random_state=42, solver="lbfgs")),
                ]),
            ),
            (
                "Regressão Logística (L2, C=1.0)",
                "Reg. Logística",
                num_kmers * 4 + 4,
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=300, C=1.0, random_state=42, solver="lbfgs")),
                ]),
            ),
            (
                "PMC Enxuto V3.0 (32,) + Dropout (p=0.3)",
                "PMC V3.0 (32,)",
                (num_kmers * 32 + 32) + (32 * 4 + 4),
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LeanMLPDropout(hidden_layers=(32,), dropout_rate=0.3, alpha=0.01, max_iter=25, random_state=42)),
                ]),
            ),
        ]

        for nome_longo, nome_curto, n_params, pipe in modelos_k:
            t_ini = time.time()
            pipe.fit(X_tr, y_tr)
            t_treino = time.time() - t_ini

            # Teste Cego 2000 pb
            preds_te = pipe.predict(X_te)
            acc_te = accuracy_score(y_te, preds_te) * 100

            # Teste Aleatório Limpo
            t_inf_ini = time.time()
            p_limpo = pipe.predict(X_al_limpo)
            t_inferencia_ms = (time.time() - t_inf_ini) * 1000.0

            acc_glob_limpo = accuracy_score(y_aleat, p_limpo) * 100
            f1_glob_limpo = f1_score(y_aleat, p_limpo, average="macro") * 100
            acc_curto_limpo = accuracy_score(y_aleat[mask_curto], p_limpo[mask_curto]) * 100
            acc_amp_limpo = accuracy_score(y_aleat[mask_amplicon], p_limpo[mask_amplicon]) * 100
            acc_gene_limpo = accuracy_score(y_aleat[mask_gene], p_limpo[mask_gene]) * 100
            acc_genoma_limpo = accuracy_score(y_aleat[mask_genoma], p_limpo[mask_genoma]) * 100

            # Ruído 4%
            p_r4 = pipe.predict(X_al_r4)
            acc_glob_r4 = accuracy_score(y_aleat, p_r4) * 100
            acc_curto_r4 = accuracy_score(y_aleat[mask_curto], p_r4[mask_curto]) * 100
            acc_amp_r4 = accuracy_score(y_aleat[mask_amplicon], p_r4[mask_amplicon]) * 100

            # Ruído 10%
            p_r10 = pipe.predict(X_al_r10)
            acc_glob_r10 = accuracy_score(y_aleat, p_r10) * 100
            acc_curto_r10 = accuracy_score(y_aleat[mask_curto], p_r10[mask_curto]) * 100
            acc_amp_r10 = accuracy_score(y_aleat[mask_amplicon], p_r10[mask_amplicon]) * 100

            reg = {
                "K": k,
                "Features": num_kmers,
                "Modelo": nome_longo,
                "Nome Curto": nome_curto,
                "Parametros": n_params,
                "Tempo Treino (s)": round(t_treino, 2),
                "Inferencia 1846 seqs (ms)": round(t_inferencia_ms, 1),
                "Teste Cego 2000pb (%)": round(acc_te, 2),
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
                f"   [{nome_longo:<41}] ({t_treino:4.1f}s) | "
                f"2000pb: {acc_te:6.2f}% | Global: {acc_glob_limpo:6.2f}% | "
                f"<300pb: {acc_curto_limpo:5.2f}% | 300-1000pb: {acc_amp_limpo:5.2f}% "
                f"(R4%: {acc_amp_r4:5.2f}% | R10%: {acc_amp_r10:5.2f}%)"
            )

    # 4. Salvar CSV e Gerar Painel Gráfico Científico
    print("\n[4/4] Exportando resultados e gerando painel visual comparativo...")
    df_res = pd.DataFrame(resultados)
    dir_bench = os.path.join(DIR_RESULTADOS, "benchmarks")
    os.makedirs(dir_bench, exist_ok=True)
    caminho_csv = os.path.join(dir_bench, "benchmark_rbf_vs_pmc_rl.csv")
    df_res.to_csv(caminho_csv, index=False)
    print(f"Tabela salva em '{caminho_csv}'.")

    # Gráfico Comparativo K=6
    os.makedirs(DIR_GRAFICOS, exist_ok=True)
    caminho_png = os.path.join(DIR_GRAFICOS, "comparativo_rbf_vs_pmc_rl.png")

    df_k6 = df_res[df_res["K"] == 6].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6))

    nomes = df_k6["Nome Curto"].tolist()
    y_pos = np.arange(len(nomes))
    altura = 0.26

    # Painel A: Desempenho por Comprimento em K=6 (Limpo)
    axes[0].barh(y_pos + altura, df_k6["Teste Cego 2000pb (%)"], height=altura, label="Janelas 2.000 pb (Teste Cego)", color="#2ca02c", alpha=0.85)
    axes[0].barh(y_pos, df_k6["Limpo 300-1000pb (%)"], height=altura, label="Amplicons (300 - 1.000 pb)", color="#1f77b4", alpha=0.9)
    axes[0].barh(y_pos - altura, df_k6["Limpo <300pb (%)"], height=altura, label="Reads Ultra-Curtas (< 300 pb)", color="#ff7f0e", alpha=0.9)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(nomes, fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Acurácia (%)", fontweight="bold", fontsize=11)
    axes[0].set_title("A) Desempenho em K=6 (4.096 hexâmeros) por Escala Genômica", fontweight="bold", fontsize=11.5)
    axes[0].set_xlim(35, 103)
    axes[0].grid(True, axis="x", linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=9)

    # Painel B: Amplicons (300-1000 pb) sob Ruído Estocástico Crescente (0%, 4% e 10%) em K=6
    axes[1].barh(y_pos + altura, df_k6["Limpo 300-1000pb (%)"], height=altura, label="0% Ruído (Limpo)", color="#1f77b4", alpha=0.9)
    axes[1].barh(y_pos, df_k6["Ruido 4% 300-1000pb (%)"], height=altura, label="4% Ruído NGS (Mutações + Indels)", color="#9467bd", alpha=0.9)
    axes[1].barh(y_pos - altura, df_k6["Ruido 10% 300-1000pb (%)"], height=altura, label="10% Ruído Severo", color="#d62728", alpha=0.9)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(nomes, fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Acurácia em Amplicons 300-1.000 pb (%)", fontweight="bold", fontsize=11)
    axes[1].set_title("B) Resiliência a Ruído NGS em Amplicons (K=6)", fontweight="bold", fontsize=11.5)
    axes[1].set_xlim(35, 100)
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right", fontsize=9)

    plt.suptitle("Duelo de Paradigmas Neurais: Rede RBF & Kernel RBF (Localidade Euclidiana) vs PMC V3.0 (Projeção Direcional)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico comparativo salvo em '{caminho_png}'.")

    print("\n" + "=" * 95)
    print("TABELA CONSOLIDADA: REDE NEURAL RBF vs KERNEL RBF vs REGRESSÃO LOGÍSTICA vs PMC V3.0")
    print("=" * 95)
    cols = [
        "K",
        "Nome Curto",
        "Teste Cego 2000pb (%)",
        "Limpo Global (%)",
        "Limpo <300pb (%)",
        "Limpo 300-1000pb (%)",
        "Ruido 4% 300-1000pb (%)",
        "Ruido 10% 300-1000pb (%)",
    ]
    print(df_res[cols].to_string(index=False))


if __name__ == "__main__":
    main()
