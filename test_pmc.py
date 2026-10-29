import unittest
import numpy as np
from PMC import (
    BASES,
    TRIMEROS_64,
    validar_sequencia,
    extrair_frequencias_trimeros,
    calcular_qtd_parametros_pmc,
    carregar_pipeline,
    predizer_amostra,
)


class TestPMCFeatures(unittest.TestCase):
    """Testes automatizados para validação de sequências e extração de trímeros."""

    def test_trimeros_64_dimensao_e_ordem(self):
        """Garante que existem exatamente 64 trímeros na ordem canônica esperada."""
        self.assertEqual(len(TRIMEROS_64), 64)
        self.assertEqual(TRIMEROS_64[0], "AAA")
        self.assertEqual(TRIMEROS_64[-1], "TTT")
        self.assertEqual(len(set(TRIMEROS_64)), 64)

    def test_validar_sequencia_tamanho(self):
        """Verifica se sequências fora do intervalo [min_len, max_len] são rejeitadas."""
        seq_curta = "A" * 9999
        seq_ok = "A" * 10500
        seq_longa = "A" * 11501

        valida_curta, motivo_curta = validar_sequencia(seq_curta)
        self.assertFalse(valida_curta)
        self.assertIn("Tamanho fora do intervalo", motivo_curta)

        valida_ok, motivo_ok = validar_sequencia(seq_ok)
        self.assertTrue(valida_ok)
        self.assertIsNone(motivo_ok)

        valida_longa, motivo_longa = validar_sequencia(seq_longa)
        self.assertFalse(valida_longa)
        self.assertIn("Tamanho fora do intervalo", motivo_longa)

    def test_validar_sequencia_vazia(self):
        """Verifica se sequências vazias ou apenas com espaços são devidamente rejeitadas."""
        valida_vazia, motivo_vazia = validar_sequencia("")
        self.assertFalse(valida_vazia)
        self.assertEqual(motivo_vazia, "Sequência vazia")

        valida_espacos, motivo_espacos = validar_sequencia("   ")
        self.assertFalse(valida_espacos)
        self.assertEqual(motivo_espacos, "Sequência vazia")

    def test_validar_sequencia_tolerancia_ambiguas(self):
        """Verifica se a tolerância a bases ambíguas (N) respeita o limiar configurado."""
        seq_aceita = ("ACGT" * 2487) + ("N" * 52)  # total 10000 pb, 52 Ns = 0.52%
        valida_aceita, motivo_aceita = validar_sequencia(seq_aceita, max_n_ratio=0.01)
        self.assertTrue(valida_aceita)
        self.assertIsNone(motivo_aceita)

        seq_rejeitada = ("ACGT" * 2450) + ("N" * 200)  # total 10000 pb, 200 Ns = 2.0%
        valida_rejeitada, motivo_rejeitada = validar_sequencia(seq_rejeitada, max_n_ratio=0.01)
        self.assertFalse(valida_rejeitada)
        self.assertIn("Excesso de bases ambíguas", motivo_rejeitada)

    def test_extrair_frequencias_soma_unitaria(self):
        """Verifica se a soma das frequências relativas dos 64 trímeros resulta estritamente em 1.0."""
        seq_limpa = ("ACGTGCAATGCA" * 900)
        vetor_limpo = extrair_frequencias_trimeros(seq_limpa)
        self.assertEqual(len(vetor_limpo), 64)
        self.assertAlmostEqual(sum(vetor_limpo), 1.0, places=6)
        self.assertFalse(any(np.isnan(vetor_limpo)))
        self.assertFalse(any(np.isinf(vetor_limpo)))

        seq_com_n = ("ACGT" * 2500) + "NNN" + ("ACGT" * 100)
        vetor_com_n = extrair_frequencias_trimeros(seq_com_n)
        self.assertEqual(len(vetor_com_n), 64)
        self.assertAlmostEqual(sum(vetor_com_n), 1.0, places=6)
        self.assertFalse(any(np.isnan(vetor_com_n)))

    def test_extrair_frequencias_caso_controlado(self):
        """Verifica a exatidão matemática para uma sequência artificial com proporção conhecida."""
        seq_poli_a = "A" * 10000
        vetor_poli_a = extrair_frequencias_trimeros(seq_poli_a)
        idx_aaa = TRIMEROS_64.index("AAA")
        self.assertEqual(vetor_poli_a[idx_aaa], 1.0)
        self.assertEqual(sum(vetor_poli_a), 1.0)
        for i, val in enumerate(vetor_poli_a):
            if i != idx_aaa:
                self.assertEqual(val, 0.0)

    def test_invariancia_rna_dna(self):
        """Garante que a presença de Uracila ('U') seja convertida em Timina ('T') sem alteração."""
        seq_dna = "ACGTACTGACGT" * 900
        seq_rna = seq_dna.replace("T", "U")
        vetor_dna = extrair_frequencias_trimeros(seq_dna)
        vetor_rna = extrair_frequencias_trimeros(seq_rna)
        np.testing.assert_allclose(vetor_dna, vetor_rna, rtol=1e-5)

    def test_calculo_parametros_pmc(self):
        """Valida o cálculo formal do número de parâmetros treináveis da rede neural."""
        # 1 Camada (64): 64*64 + 64 (oculta) + 64*4 + 4 (saída) = 4.096 + 64 + 256 + 4 = 4.420
        self.assertEqual(calcular_qtd_parametros_pmc((64,)), 4420)
        # 2 Camadas (64, 32): 64*64+64 + 64*32+32 + 32*4+4 = 4.160 + 2.080 + 132 = 6.372
        self.assertEqual(calcular_qtd_parametros_pmc((64, 32)), 6372)


class TestPMCPipeline(unittest.TestCase):
    """Testes de integração com o modelo treinado persistido."""

    def test_predicao_pipeline_carregado(self):
        """Verifica se o pipeline serializado carrega e realiza inferência válida."""
        pipe = carregar_pipeline()
        self.assertIsNotNone(pipe)
        # Amostra de teste válida
        seq_teste = "ATGC" * 2600
        pred, probas = predizer_amostra(seq_teste, pipe)
        self.assertIn(pred, ["DENV-1", "DENV-2", "DENV-3", "DENV-4"])
        self.assertAlmostEqual(sum(probas.values()), 100.0, places=1)


if __name__ == "__main__":
    unittest.main()
