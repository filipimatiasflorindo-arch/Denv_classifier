"""
================================================================================
INTERFACE STREAMLIT: CLASSIFICADOR PMC DE SOROTIPOS DO VÍRUS DA DENGUE
Abordagem por Fragmentos Genômicos Curtos (Simulação NGS / Contigs)
Trabalho Acadêmico de Graduação em Inteligência Artificial / Bioinformática
================================================================================
"""

import datetime
import importlib
import os
import sys
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Garante que o diretório raiz esteja no sys.path e recarrega o módulo PMC caso o servidor
# Streamlit já estivesse em execução antes da reorganização das subpastas (hot-reload seguro)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import PMC as _pmc_mod
if not hasattr(_pmc_mod, "CAMINHO_PIPELINE_PADRAO"):
    _pmc_mod = importlib.reload(_pmc_mod)

# Importa o pipeline e funções de suporte do módulo PMC principal
from PMC import (
    CAMINHO_PIPELINE_PADRAO,
    DIR_GRAFICOS,
    DIR_RESULTADOS,
    K_MER_SIZE,
    NUM_KMERS,
    aplicar_ruido_estocastico,
    carregar_pipeline,
    extrair_frequencias_kmers,
)

# Configuração da Página
st.set_page_config(
    page_title="Classificador PMC - Fragmentos DENV",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

HISTORICO_CSV = os.path.join(DIR_RESULTADOS, "historico_predicoes.csv")
CORES_SOROTIPOS = {
    "DENV-1": "#1f77b4",  # Azul
    "DENV-2": "#2ca02c",  # Verde
    "DENV-3": "#ff7f0e",  # Laranja
    "DENV-4": "#d62728",  # Vermelho
}


# ==============================================================================
# FUNÇÕES DE GERENCIAMENTO DO HISTÓRICO
# ==============================================================================


def carregar_historico():
    """Carrega o histórico de análises salvas ou cria um DataFrame vazio."""
    colunas = [
        "Data/Hora",
        "Arquivo",
        "ID Sequência",
        "Tamanho (pb)",
        "Sorotipo Predito",
        "Confiança (%)",
        "DENV-1 (%)",
        "DENV-2 (%)",
        "DENV-3 (%)",
        "DENV-4 (%)",
    ]
    if os.path.exists(HISTORICO_CSV):
        try:
            return pd.read_csv(HISTORICO_CSV)
        except Exception:
            return pd.DataFrame(columns=colunas)
    return pd.DataFrame(columns=colunas)


def salvar_no_historico(novos_registros):
    """Adiciona novas predições ao arquivo de histórico persistente."""
    if not novos_registros:
        return
    os.makedirs(os.path.dirname(HISTORICO_CSV), exist_ok=True)
    df_novo = pd.DataFrame(novos_registros)
    if os.path.exists(HISTORICO_CSV):
        df_novo.to_csv(HISTORICO_CSV, mode="a", header=False, index=False)
    else:
        df_novo.to_csv(HISTORICO_CSV, index=False)


# ==============================================================================
# PARSER FASTA PARA O STREAMLIT
# ==============================================================================


def processar_texto_fasta(texto):
    """Lê o conteúdo textual de um FASTA (ou sequência bruta sem cabeçalho) e extrai sequências."""
    linhas = texto.splitlines()
    sequencias = []
    seq_atual = []
    header_atual = ""
    header_completo = ""

    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue
        if linha.startswith(">"):
            if seq_atual:
                id_final = header_atual or f"Sequencia_Manual_{len(sequencias) + 1:02d}"
                desc_final = header_completo or id_final
                sequencias.append({
                    "id": id_final,
                    "descricao": desc_final,
                    "seq": "".join(seq_atual),
                })
                seq_atual = []
            header_completo = linha[1:].strip() or f"Sequencia_Manual_{len(sequencias) + 1:02d}"
            partes = header_completo.split()
            header_atual = partes[0] if partes else header_completo
        else:
            seq_atual.append(linha)

    if seq_atual:
        id_final = header_atual or f"Sequencia_Manual_{len(sequencias) + 1:02d}"
        desc_final = header_completo or id_final
        sequencias.append({
            "id": id_final,
            "descricao": desc_final,
            "seq": "".join(seq_atual),
        })

    return sequencias


# ==============================================================================
# CARREGAMENTO DO MODELO EM CACHE
# ==============================================================================


@st.cache_resource(show_spinner="Carregando modelo PMC e normalizador...")
def obter_pipeline(caminho_modelo: str = CAMINHO_PIPELINE_PADRAO):
    """Carrega o pipeline serializado uma única vez em cache a partir de modelos/."""
    return carregar_pipeline(caminho_modelo)


try:
    pipeline = obter_pipeline(CAMINHO_PIPELINE_PADRAO)
    classes_modelo = list(pipeline.named_steps["pmc"].classes_)
except FileNotFoundError as erro_modelo:
    st.error(
        f"⚠️ **Modelo treinado não encontrado:** `{erro_modelo}`\n\n"
        f"Certifique-se de que o arquivo exista em `{CAMINHO_PIPELINE_PADRAO}` "
        f"ou execute `python PMC.py` na raiz do projeto para treiná-lo."
    )
    st.stop()


# ==============================================================================
# BARRA LATERAL (SIDEBAR)
# ==============================================================================

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Dengue_virus_cryo-EM.png/320px-Dengue_virus_cryo-EM.png",
        caption="Estrutura do Vírus da Dengue (Flaviviridae)",
    )
    st.title("PMC - Fragmentos DENV")
    st.markdown(f"""
    **Trabalho Acadêmico de Graduação (Versão 3.0)**
    * **Rede:** Perceptron Multicamadas Enxuto (`LeanMLPDropout`)
    * **Arquitetura:** `(32,)` neurônios ocultos
    * **Regularização:** Inverted Dropout ($p=0.3$) + $L_2$ ($\alpha=0.01$)
    * **Entrada:** {NUM_KMERS:,} K-mers canônicos ($K={K_MER_SIZE}$)
    * **Resolução:** Fragmentos curtos / Reads brutas NGS / Amplicons
    * **Classes:** DENV-1, DENV-2, DENV-3, DENV-4
    """)
    st.divider()

    st.subheader("⚙️ Robustez Metodológica")
    st.markdown("""
    * **Zero Data Leakage:** Partição estrita ao nível de genomas completos antes do fatiamento.
    * **Resiliência a Ruído:** Testado contra mutações e indels aleatórios de basecalling.
    * **Normalização:** Z-score (`StandardScaler`) integrado no Pipeline.
    """)


# ==============================================================================
# CORPO PRINCIPAL E ABAS
# ==============================================================================

st.title("🧬 Classificação de Fragmentos Genômicos da Dengue via PMC")
st.markdown("Sistema inteligente para diagnóstico genômico e tipagem de contigs e leituras parciais via assinaturas de 6-mers canônicos (4.096 características).")

tab_classificar, tab_dataset, tab_modelo = st.tabs([
    "🔍 Classificar Nova Amostra / Fragmento",
    "📊 Conjunto de Dados & Histórico",
    "🧠 Arquitetura, Diagnóstico & Auditoria Biológica",
])


# ------------------------------------------------------------------------------
# ABA 1: CLASSIFICAÇÃO DE NOVAS AMOSTRAS
# ------------------------------------------------------------------------------
with tab_classificar:
    st.subheader("1. Carregar Arquivo FASTA ou Colar Fragmento/Genoma")
    
    tipo_entrada = st.radio(
        "Formato de Entrada:",
        ["📁 Upload de Arquivo FASTA (.fasta / .fa)", "✏️ Colar Sequência de Texto"],
        horizontal=True,
    )

    amostras_para_analise = []
    nome_origem = "Entrada Manual"

    if "Upload" in tipo_entrada:
        arquivo_upload = st.file_uploader(
            "Selecione um arquivo FASTA contendo fragmentos ou genomas completos:",
            type=["fasta", "fa", "txt", "fna"],
            help="Fragmentos curtos, contigs ou genomas completos no formato FASTA padrão.",
        )
        if arquivo_upload is not None:
            nome_origem = arquivo_upload.name
            conteudo_str = arquivo_upload.getvalue().decode("utf-8", errors="ignore")
            amostras_para_analise = processar_texto_fasta(conteudo_str)
            st.info(f"Arquivo **{nome_origem}** lido com sucesso! Contém **{len(amostras_para_analise)}** sequência(s).")
    else:
        texto_seq = st.text_area(
            "Cole o conteúdo FASTA completo (incluindo o cabeçalho >):",
            height=200,
            placeholder=">Fragmento_01 Contig_NS5\nGGCCAGTTCGGAAGCTTGCTTAACACAGTTCTAACAGTTTGTTTGAATAGAGAGCAGATC...",
        )
        if texto_seq.strip():
            amostras_para_analise = processar_texto_fasta(texto_seq)

    simular_ruido_ngs = st.checkbox(
        "🧪 Simular Leituras Brutas de Sequenciamento NGS (Injetar 4% de ruído estocástico: mutações pontuais e indels)",
        value=False,
        help="Simula sequenciamento de leituras brutas (Oxford Nanopore / Illumina sem polimento de basecalling) para auditar a resiliência do modelo.",
    )

    # Botão de Execução
    if st.button("🚀 Executar Classificação pelo PMC", type="primary", width="stretch"):
        if not amostras_para_analise:
            st.warning("Por favor, carregue um arquivo FASTA ou insira uma sequência para continuar.")
        else:
            novos_registros_historico = []
            resultados_exibicao = []
            timestamp_agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with st.spinner(f"Extraindo {NUM_KMERS:,} K-mers e processando inferência pela rede neural..."):
                for amostra in amostras_para_analise:
                    id_seq = amostra["id"]
                    seq_str = amostra["seq"]

                    if simular_ruido_ngs:
                        seq_str = aplicar_ruido_estocastico(seq_str, taxa_erro=0.04)

                    comp_pb = len(seq_str)

                    if comp_pb < K_MER_SIZE:
                        st.error(f"❌ **[{id_seq}] Rejeitada:** Sequência muito curta ({comp_pb} pb < {K_MER_SIZE} pb).")
                        continue

                    # Extração dos 4.096 K-mers canônicos (K=6)
                    vetor_features = extrair_frequencias_kmers(seq_str, K_MER_SIZE).reshape(1, -1)
                    pred = pipeline.predict(vetor_features)[0]
                    probas = pipeline.predict_proba(vetor_features)[0]

                    prob_dict = {c: float(np.round(p * 100, 2)) for c, p in zip(classes_modelo, probas)}
                    confianca = prob_dict[str(pred)]

                    reg_hist = {
                        "Data/Hora": timestamp_agora,
                        "Arquivo": nome_origem,
                        "ID Sequência": id_seq,
                        "Tamanho (pb)": comp_pb,
                        "Sorotipo Predito": str(pred),
                        "Confiança (%)": confianca,
                        "DENV-1 (%)": prob_dict.get("DENV-1", 0.0),
                        "DENV-2 (%)": prob_dict.get("DENV-2", 0.0),
                        "DENV-3 (%)": prob_dict.get("DENV-3", 0.0),
                        "DENV-4 (%)": prob_dict.get("DENV-4", 0.0),
                    }
                    novos_registros_historico.append(reg_hist)
                    resultados_exibicao.append(reg_hist)

            # Persistência no histórico local
            if novos_registros_historico:
                salvar_no_historico(novos_registros_historico)
                st.success(f"✅ {len(novos_registros_historico)} amostra(s) classificada(s) e registradas no banco de dados!")

                # Se for apenas 1 sequência, mostra visualização em destaque
                if len(resultados_exibicao) == 1:
                    res = resultados_exibicao[0]
                    cor_classe = CORES_SOROTIPOS.get(res["Sorotipo Predito"], "#333333")

                    st.markdown("---")
                    st.markdown(f"""
                    <div style="background-color: {cor_classe}15; border-left: 6px solid {cor_classe}; padding: 16px; border-radius: 8px;">
                        <h2 style="color: {cor_classe}; margin: 0;">Sorotipo Identificado: {res['Sorotipo Predito']}</h2>
                        <h4 style="margin: 4px 0 0 0; color: #555;">Confiança da Rede Neural: <b>{res['Confiança (%)']:.2f}%</b> | ID: <code>{res['ID Sequência']}</code> ({res['Tamanho (pb)']:,} pb)</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Distribuição de Probabilidades:")
                    df_probs = pd.DataFrame({
                        "Sorotipo": classes_modelo,
                        "Probabilidade (%)": [res[f"{c} (%)"] for c in classes_modelo],
                    })

                    chart = (
                        alt.Chart(df_probs)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                        .encode(
                            x=alt.X("Sorotipo:N", title="Sorotipo da Dengue"),
                            y=alt.Y("Probabilidade (%):Q", scale=alt.Scale(domain=[0, 100])),
                            color=alt.Color(
                                "Sorotipo:N",
                                scale=alt.Scale(
                                    domain=list(CORES_SOROTIPOS.keys()),
                                    range=list(CORES_SOROTIPOS.values()),
                                ),
                                legend=None,
                            ),
                            tooltip=["Sorotipo", "Probabilidade (%)"],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(chart, width="stretch")

                # Se forem múltiplas sequências, mostra em tabela interativa
                else:
                    st.markdown("### Tabela de Classificação das Amostras:")
                    df_res = pd.DataFrame(resultados_exibicao)
                    st.dataframe(df_res, width="stretch")


# ------------------------------------------------------------------------------
# ABA 2: CONJUNTO DE DADOS & HISTÓRICO ACUMULADO
# ------------------------------------------------------------------------------
with tab_dataset:
    st.subheader("📈 Painel do Conjunto de Dados Acumulado")
    st.markdown("Monitoramento de todas as amostras/fragmentos submetidos e classificados pelo modelo até o momento.")

    df_hist = carregar_historico()

    if df_hist.empty:
        st.info("Nenhuma amostra foi analisada ainda. Carregue um arquivo FASTA na primeira aba para iniciar o rastreamento.")
    else:
        # Contagem por sorotipo
        contagem = df_hist["Sorotipo Predito"].value_counts()
        total_analisadas = len(df_hist)

        # Cards de métricas
        c_tot, c1, c2, c3, c4 = st.columns(5)
        c_tot.metric("Total Analisadas", f"{total_analisadas:,}")
        c1.metric("DENV-1", contagem.get("DENV-1", 0))
        c2.metric("DENV-2", contagem.get("DENV-2", 0))
        c3.metric("DENV-3", contagem.get("DENV-3", 0))
        c4.metric("DENV-4", contagem.get("DENV-4", 0))

        st.divider()

        col_graf, col_info = st.columns([3, 2])

        with col_graf:
            st.markdown("#### Proporção Atual de Vírus por Sorotipo:")
            df_contagem = pd.DataFrame({
                "Sorotipo": ["DENV-1", "DENV-2", "DENV-3", "DENV-4"],
                "Quantidade": [contagem.get(s, 0) for s in ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]],
            })

            chart_hist = (
                alt.Chart(df_contagem)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Sorotipo:N", title="Sorotipo"),
                    y=alt.Y("Quantidade:Q", title="Total de Amostras"),
                    color=alt.Color(
                        "Sorotipo:N",
                        scale=alt.Scale(
                            domain=list(CORES_SOROTIPOS.keys()),
                            range=list(CORES_SOROTIPOS.values()),
                        ),
                        legend=None,
                    ),
                    tooltip=["Sorotipo", "Quantidade"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_hist, width="stretch")

        with col_info:
            st.markdown("#### Resumo Estatístico:")
            st.write(f"* **Total Geral de Amostras:** {total_analisadas}")
            for s in ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]:
                qtd = contagem.get(s, 0)
                pct = (qtd / total_analisadas * 100) if total_analisadas > 0 else 0
                st.write(f"* **{s}:** {qtd} amostras ({pct:.1f}%)")

            # Botão para download do histórico em CSV
            csv_data = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Baixar Histórico Completo (CSV)",
                data=csv_data,
                file_name="historico_amostras_denv.csv",
                mime="text/csv",
                width="stretch",
            )

        st.markdown("### Registro Detalhado das Amostras:")
        st.dataframe(df_hist, width="stretch")

        # Opção de resetar histórico com confirmação
        with st.expander("⚠️ Opções de Manutenção do Banco de Dados"):
            if st.button("🗑️ Limpar Todo o Histórico de Amostras", type="secondary"):
                if os.path.exists(HISTORICO_CSV):
                    os.remove(HISTORICO_CSV)
                    st.rerun()


# ------------------------------------------------------------------------------
# ABA 3: INFORMAÇÕES DO MODELO PMC & AUDITORIA BIOLÓGICA
# ------------------------------------------------------------------------------
with tab_modelo:
    st.subheader("🧠 Metodologia, Evolução das Versões e Auditoria Biológica")
    st.markdown("""
    O classificador opera com a **Versão 3.0 (Definitiva)** do projeto: modelo **PMC Enxuto (`LeanMLPDropout`)**
    com **4.096 hexâmeros canônicos ($K=6$)**, camada oculta única de **32 neurônios**, **Inverted Dropout ($p=0.3$)**
    e **regularização $L_2$ ($\alpha=0.01$)** em NumPy puro (sem dependência de PyTorch).
    """)

    st.markdown("### 📜 Histórico Evolutivo do Projeto e Comparativo de Versões")

    df_versoes = pd.DataFrame([
        {
            "Versão": "V1.0 (Legado)",
            "Alvo": "Genomas Completos (~10.700 pb)",
            "Features": "64 Trímeros (K=3)",
            "Arquitetura": "PMC (64, 32)",
            "Acurácia Global": "99,9%",
            "Diagnóstico Metodológico": "Separação linearmente trivial. PMC injustificado (overengineering)."
        },
        {
            "Versão": "V2.0",
            "Alvo": "Fragmentos (2.000 pb)",
            "Features": "1.024 Pentâmeros (K=5)",
            "Arquitetura": "PMC (256, 128)",
            "Acurácia Global": "96,15%",
            "Diagnóstico Metodológico": "Zero Data Leakage a nível de genoma. Queda em reads curtas (<300 pb: 69,1%)."
        },
        {
            "Versão": "V2.5",
            "Alvo": "Fragmentos (2.000 pb) e Reads",
            "Features": "4.096 Hexâmeros (K=6)",
            "Arquitetura": "PMC Piramidal (512, 128)",
            "Acurácia Global": "97,51%",
            "Diagnóstico Metodológico": "Alta especificidade alélica. Modelo pesado (2,16M pesos) e sensível a ruído NGS."
        },
        {
            "Versão": "V3.0 (Definitiva)",
            "Alvo": "Fragmentos, Amplicons e Reads NGS",
            "Features": "4.096 Hexâmeros (K=6)",
            "Arquitetura": "PMC Enxuto (32,) + Dropout (p=0.3) + L2",
            "Acurácia Global": "97,56% (Limpo) | 97,40% (Ruído NGS)",
            "Diagnóstico Metodológico": "131.200 pesos sinápticos (131.236 parâmetros com vieses, -94%). Supera a Regressão Logística sob ruído NGS."
        },
    ])
    st.dataframe(df_versoes, width="stretch", hide_index=True)

    st.markdown("### 🔬 Duelo Experimental: PMC Enxuto vs Regressão Logística (Cenários Limpo e Ruidoso)")
    st.markdown("""
    Sob simulação de **leitura bruta de sequenciamento NGS com 4% de erro estocástico (substituições e indels)**,
    o modelo linear aditivo sofre degradação acentuada (-5,83%), enquanto o **PMC com Dropout e regularização L2
    permanece resiliente e supera a Regressão Logística em amplicons (90,98% vs 85,66%)**:
    """)

    df_comp_ruido = pd.DataFrame([
        {"Modelo": "PMC (32,) + Dropout (p=0.3) + L2", "Limpo (Global)": "97,56%", "Limpo (300-1000pb)": "91,80%", "Ruído 4% (Global)": "97,40%", "Ruído 4% (300-1000pb)": "90,98%", "Queda em Reads (<300pb)": "-0,97% (Resiliente)"},
        {"Modelo": "PMC (64, 16) + Dropout (p=0.3) + L2", "Limpo (Global)": "97,45%", "Limpo (300-1000pb)": "91,39%", "Ruído 4% (Global)": "97,13%", "Ruído 4% (300-1000pb)": "90,57%", "Queda em Reads (<300pb)": "-3,88%"},
        {"Modelo": "Regressão Logística (L2)", "Limpo (Global)": "97,51%", "Limpo (300-1000pb)": "89,34%", "Ruído 4% (Global)": "96,70%", "Ruído 4% (300-1000pb)": "85,66%", "Queda em Reads (<300pb)": "-5,83% (Sensível)"},
        {"Modelo": "PMC Grande (512, 128) [V2.5]", "Limpo (Global)": "97,51%", "Limpo (300-1000pb)": "91,39%", "Ruído 4% (Global)": "97,02%", "Ruído 4% (300-1000pb)": "89,34%", "Queda em Reads (<300pb)": "-3,88%"},
    ])
    st.dataframe(df_comp_ruido, width="stretch", hide_index=True)

    with st.expander("📊 Novos Experimentos do Artigo (Ablação K=4..6, Rede Neural RBF e CNN 1D / Posição Genômica)", expanded=False):
        caminho_img_k46 = os.path.join(DIR_GRAFICOS, "comparativo_regressao_logistica_k4_k6.png")
        caminho_img_rbf = os.path.join(DIR_GRAFICOS, "comparativo_rbf_vs_pmc_rl.png")
        caminho_img_cnn = os.path.join(DIR_GRAFICOS, "comparativo_cnn1d_posicao_genomica.png")
        caminho_img_stress = os.path.join(DIR_GRAFICOS, "curva_degradacao_ruido_extremo.png")

        if os.path.exists(caminho_img_k46):
            st.markdown("#### 1. Ablação da Ordem do K-mer ($K=4, 5, 6$) na Regressão Logística vs PMC V3.0")
            st.image(caminho_img_k46, caption="Impacto da Ordem do K-mer (256, 1.024 e 4.096 features) na Regressão Logística e no PMC Enxuto")
        if os.path.exists(caminho_img_rbf):
            st.markdown("#### 2. Rede Neural RBF (Moody-Darken) & Kernel RBF vs PMC V3.0")
            st.image(caminho_img_rbf, caption="Comparativo entre Localidade Euclidiana (RBF) e Projeção Direcional (PMC V3.0)")
        if os.path.exists(caminho_img_cnn):
            st.markdown("#### 3. CNN 1D (Posicional vs Global) e Influência da Posição Genômica ($5' \\to 3'$)")
            st.image(caminho_img_cnn, caption="Varredura posicional de amplicons de 500 pb ao longo das 10 regiões funcionais do DENV")
        if os.path.exists(caminho_img_stress):
            st.markdown("#### 4. Teste de Estresse de Ruído NGS (0% a 15%)")
            st.image(caminho_img_stress, caption="Curva de degradação sob ruído estocástico crescente (0%, 4%, 8%, 10% e 15%)")

    st.divider()

    caminho_matriz = os.path.join(DIR_GRAFICOS, "matriz_confusao_fragmentos.png")
    caminho_auditoria = os.path.join(DIR_RESULTADOS, "auditoria_erros_fragmentos.csv")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists(caminho_matriz):
            st.image(caminho_matriz, caption="Matriz de Confusão em Fragmentos Curtos de 2.000 pb (V3.0)")

    with col_img2:
        st.markdown("#### 🧬 Auditoria Biológica de Incerteza e Erros:")
        st.markdown("""
        Fragmentos de sequenciamento que apresentaram menor confiança ou divergência de predição
        foram mapeados contra as coordenadas canônicas da poliproteína viral:
        * **Região mais crítica:** **NS5 (RNA Polimerase dependente de RNA / Metiltransferase)**
          (Posições 7.638 - 10.340 pb).
        * **Fundamentação Biológica:** A enzima NS5 é o gene mais conservado evolutivamente entre todos os Flavivírus,
          o que diminui a distância de K-mers entre os sorotipos nessa região específica do genoma.
        """)

    if os.path.exists(caminho_auditoria):
        st.markdown("#### Tabela de Fragmentos com Incerteza (<85%) ou Erro no Teste Cego:")
        df_audit = pd.read_csv(caminho_auditoria)
        st.dataframe(df_audit, width="stretch")
