"""
================================================================================
SUÍTE DE TESTES UNITÁRIOS AUTOMATIZADOS: CLASSIFICADOR PMC DE FRAGMENTOS DENV (V3.0)
================================================================================
"""

import os
import sys
import unittest
import numpy as np

# Garante que o diretório raiz do projeto esteja no PYTHONPATH independente de onde o teste for executado
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PMC import (
    BASES,
    K_MER_SIZE,
    KMERS_CANONICOS,
    NUM_KMERS,
    validar_sequencia,
    carregar_fasta,
    extrair_frequencias_kmers,
    fatiar_genoma,
    mapear_regioes_genomicas,
    aplicar_ruido_estocastico,
    deduplicar_genomas_estrito,
    LeanMLPDropout,
    carregar_pipeline,
    predizer_arquivo_fasta,
)


class TestPMCFragmentos(unittest.TestCase):
    """Testes automatizados para validação de fragmentos e extração de K-mers."""

    def test_kmers_canonicos_dimensao_e_ordem(self):
        """Garante que existam 4^K K-mers (4.096 para K=6) ordenados canonicamente."""
        self.assertEqual(len(KMERS_CANONICOS), 4**K_MER_SIZE)
        self.assertEqual(NUM_KMERS, 4096)
        self.assertEqual(KMERS_CANONICOS[0], "A" * K_MER_SIZE)
        self.assertEqual(KMERS_CANONICOS[-1], "T" * K_MER_SIZE)
        self.assertEqual(len(set(KMERS_CANONICOS)), NUM_KMERS)

    def test_validar_sequencia_regras(self):
        """Verifica se os filtros de comprimento e bases ambíguas funcionam rigorosamente."""
        seq_curta = "A" * 9999
        seq_ok = "A" * 10500
        seq_longa = "A" * 11501

        self.assertFalse(validar_sequencia(seq_curta)[0])
        self.assertTrue(validar_sequencia(seq_ok)[0])
        self.assertFalse(validar_sequencia(seq_longa)[0])
        self.assertFalse(validar_sequencia("")[0])

        # Teste de bases ambíguas (N)
        seq_pouco_n = ("ACGT" * 2487) + ("N" * 52)  # 0.52% Ns
        self.assertTrue(validar_sequencia(seq_pouco_n, max_n_ratio=0.01)[0])

        seq_muito_n = ("ACGT" * 2450) + ("N" * 200)  # 2.0% Ns
        self.assertFalse(validar_sequencia(seq_muito_n, max_n_ratio=0.01)[0])

    def test_fatiar_genoma_sliding_window(self):
        """Garante que a fragmentação por janela deslizante preserve metadados e sobreposição."""
        # Genoma de 10.000 pb com janela de 2.000 pb e passo de 1.000 pb -> 9 fragmentos
        seq_teste = "ACGT" * 2500
        frags = fatiar_genoma(
            seq=seq_teste,
            id_genoma="TEST_01",
            sorotipo="DENV-2",
            window_size=2000,
            step_size=1000,
        )
        self.assertEqual(len(frags), 9)

        # Checa integridade do primeiro fragmento
        f0 = frags[0]
        self.assertEqual(f0["id_genoma_origem"], "TEST_01")
        self.assertEqual(f0["posicao_inicio"], 0)
        self.assertEqual(f0["posicao_fim"], 2000)
        self.assertEqual(f0["sorotipo"], "DENV-2")
        self.assertEqual(len(f0["seq"]), 2000)
        self.assertIn("regioes", f0)

        # Checa passo de sobreposição do segundo fragmento
        f1 = frags[1]
        self.assertEqual(f1["posicao_inicio"], 1000)
        self.assertEqual(f1["posicao_fim"], 3000)

    def test_mapear_regioes_genomicas(self):
        """Verifica se as coordenadas interceptam corretamente as regiões da poliproteína."""
        # Região Envelope (936 - 2420 pb)
        reg_env = mapear_regioes_genomicas(1000, 2000)
        self.assertIn("E (Envelope)", reg_env)

        # Região NS5 Polimerase (7638 - 10340 pb)
        reg_ns5 = mapear_regioes_genomicas(8000, 9500)
        self.assertIn("NS5", reg_ns5)

    def test_extrair_frequencias_kmers_soma_unitaria(self):
        """Garante que a extração de K-mers gere vetor de 4^K dimensões (K=4, 5, 6) somando 1.0."""
        fragmento = "ACGTGCAATGCA" * 200  # 2.400 pb
        for k_teste, dim_esperada in [(4, 256), (5, 1024), (6, 4096)]:
            vetor = extrair_frequencias_kmers(fragmento, k=k_teste)
            self.assertEqual(len(vetor), dim_esperada)
            self.assertAlmostEqual(float(np.sum(vetor)), 1.0, places=5)
            self.assertFalse(np.any(np.isnan(vetor)))
            self.assertFalse(np.any(np.isinf(vetor)))
        self.assertEqual(len(BASES), 4)

    def test_deduplicar_genomas_estrito(self):
        """Verifica se duplicatas intra-sorotipo e conflitos inter-sorotipo são filtrados."""
        amostras = [
            {"id": "G1", "sorotipo": "DENV-1", "hash": "hash_a"},
            {"id": "G1_dup", "sorotipo": "DENV-1", "hash": "hash_a"},
            {"id": "G2_conf1", "sorotipo": "DENV-2", "hash": "hash_conflito"},
            {"id": "G2_conf2", "sorotipo": "DENV-3", "hash": "hash_conflito"},
            {"id": "G3", "sorotipo": "DENV-4", "hash": "hash_b"},
        ]
        unicos, dup_mesmo, dup_conflito = deduplicar_genomas_estrito(amostras)
        self.assertEqual(len(unicos), 2)
        self.assertEqual(dup_mesmo, 1)
        self.assertEqual(dup_conflito, 1)

    def test_carregar_fasta_contagem_motivos_rejeicao(self):
        """Verifica se carregar_fasta separa corretamente rejeições por Comprimento vs Bases Ambíguas."""
        import tempfile
        conteudo = (
            ">seq_curta\n" + ("ACGT" * 100) + "\n"
            ">seq_muito_n\n" + ("ACGT" * 2450) + ("N" * 300) + "\n"
            ">seq_valida\n" + ("ACGT" * 2600) + "\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False, encoding="utf-8") as tmp:
            tmp.write(conteudo)
            caminho_tmp = tmp.name
        try:
            regs, stats = carregar_fasta(caminho_tmp, rotulo_sorotipo="DENV-1")
            self.assertEqual(stats["total_lidas"], 3)
            self.assertEqual(stats["aceitas"], 1)
            self.assertEqual(stats["motivos_rejeicao"]["Fora do Tamanho"], 1)
            self.assertEqual(stats["motivos_rejeicao"]["Bases Ambíguas"], 1)
            self.assertEqual(len(regs), 1)
        finally:
            os.remove(caminho_tmp)

    def test_invariancia_rna_dna(self):
        """Garante que Uracila ('U') e Timina ('T') gerem perfis estritamente idênticos."""
        frag_dna = "ACGTACTGACGT" * 200
        frag_rna = frag_dna.replace("T", "U")

        v_dna = extrair_frequencias_kmers(frag_dna, k=K_MER_SIZE)
        v_rna = extrair_frequencias_kmers(frag_rna, k=K_MER_SIZE)
        np.testing.assert_allclose(v_dna, v_rna, rtol=1e-5)

    def test_aplicar_ruido_estocastico(self):
        """Verifica o comportamento da simulação de ruído NGS (indels e mutações)."""
        seq_limpa = "ACGT" * 250  # 1.000 pb
        seq_ruidosa = aplicar_ruido_estocastico(seq_limpa, taxa_erro=0.04, random_state=42)

        self.assertNotEqual(seq_limpa, seq_ruidosa)
        # Comprimento deve permanecer próximo de 1.000 pb (tolerância de indels de +-10%)
        self.assertTrue(900 <= len(seq_ruidosa) <= 1100)

        # Taxa de erro zero não deve alterar a sequência
        seq_zero = aplicar_ruido_estocastico(seq_limpa, taxa_erro=0.0, random_state=42)
        self.assertEqual(seq_limpa, seq_zero)


class TestPMCIntegracao(unittest.TestCase):
    """Testes de inferência ponta a ponta com o modelo LeanMLPDropout serializado."""

    def test_lean_mlp_dropout_fit_predict(self):
        """Verifica o treinamento e inferência do estimador LeanMLPDropout em NumPy puro."""
        rng = np.random.default_rng(42)
        X_toy = rng.normal(0, 1, size=(50, 64)).astype(np.float32)
        y_toy = rng.choice(["DENV-1", "DENV-2", "DENV-3", "DENV-4"], size=50)

        model = LeanMLPDropout(hidden_layers=(16,), dropout_rate=0.3, alpha=0.01, max_iter=3, random_state=42)
        model.fit(X_toy, y_toy)

        probs = model.predict_proba(X_toy)
        preds = model.predict(X_toy)

        self.assertEqual(probs.shape, (50, 4))
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(50), rtol=1e-5)
        self.assertEqual(len(preds), 50)
        for p in preds:
            self.assertIn(p, ["DENV-1", "DENV-2", "DENV-3", "DENV-4"])

    def test_predicao_pipeline_fragmento(self):
        """Verifica se o pipeline serializado aceita um fragmento de 2.000 pb e retorna predição válida."""
        try:
            pipe = carregar_pipeline()
        except FileNotFoundError:
            self.skipTest("Pipeline ainda não treinado/salvo.")

        frag_sintetico = "ATGC" * 500  # 2.000 pb
        feat = extrair_frequencias_kmers(frag_sintetico, k=K_MER_SIZE).reshape(1, -1)
        pred = pipe.predict(feat)[0]
        probas = pipe.predict_proba(feat)[0]

        self.assertIn(pred, ["DENV-1", "DENV-2", "DENV-3", "DENV-4"])
        self.assertAlmostEqual(float(np.sum(probas)), 1.0, places=4)

    def test_predizer_arquivo_fasta_cli(self):
        """Verifica se predizer_arquivo_fasta resolve arquivos em dados/testes/ e classifica."""
        try:
            pipe = carregar_pipeline()
        except FileNotFoundError:
            self.skipTest("Pipeline ainda não treinado/salvo.")

        df_res = predizer_arquivo_fasta("amostra_referencia_denv1.fasta", pipeline=pipe, salvar_csv=False)
        self.assertFalse(df_res.empty)
        self.assertIn("Sorotipo Predito", df_res.columns)
        self.assertEqual(df_res.iloc[0]["Sorotipo Predito"], "DENV-1")


if __name__ == "__main__":
    unittest.main()
