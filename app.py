"""
================================================================================
INTERFACE STREAMLIT: CLASSIFICADOR PMC DE SOROTIPOS DO VÍRUS DA DENGUE
Trabalho Acadêmico de Graduação em Inteligência Artificial / Bioinformática
================================================================================
"""

import datetime
import io
import os
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Importa o pipeline e funções de suporte do módulo PMC principal
from PMC import (
    TRIMEROS_64,
    carregar_pipeline,
    extrair_frequencias_trimeros,
    validar_sequencia,
)

# Configuração da Página
st.set_page_config(
    page_title="Classificador PMC - Sorotipos DENV",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

HISTORICO_CSV = "historico_predicoes.csv"
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
            df = pd.read_csv(HISTORICO_CSV)
            return df
        except Exception:
            return pd.DataFrame(columns=colunas)
    return pd.DataFrame(columns=colunas)


def salvar_no_historico(novos_registros):
    """Adiciona novas predições ao arquivo de histórico persistente."""
    if not novos_registros:
        return
    df_novo = pd.DataFrame(novos_registros)
    if os.path.exists(HISTORICO_CSV):
        df_novo.to_csv(HISTORICO_CSV, mode="a", header=False, index=False)
    else:
        df_novo.to_csv(HISTORICO_CSV, index=False)


# ==============================================================================
# PARSER FASTA PARA O STREAMLIT
# ==============================================================================


def processar_texto_fasta(texto):
    """Lê o conteúdo textual de um FASTA e extrai cabeçalhos e sequências."""
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
                sequencias.append({
                    "id": header_atual,
                    "descricao": header_completo,
                    "seq": "".join(seq_atual),
                })
                seq_atual = []
            header_completo = linha[1:].strip()
            header_atual = header_completo.split()[0]
        else:
            seq_atual.append(linha)

    if seq_atual:
        sequencias.append({
            "id": header_atual,
            "descricao": header_completo,
            "seq": "".join(seq_atual),
        })

    return sequencias


# ==============================================================================
# CARREGAMENTO DO MODELO EM CACHE
# ==============================================================================


@st.cache_resource(show_spinner="Carregando modelo PMC e normalizador...")
def obter_pipeline():
    """Carrega o pipeline serializado uma única vez em cache."""
    return carregar_pipeline()


pipeline = obter_pipeline()
classes_modelo = list(pipeline.named_steps["pmc"].classes_)


# ==============================================================================
# BARRA LATERAL (SIDEBAR)
# ==============================================================================

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Dengue_virus_cryo-EM.png/320px-Dengue_virus_cryo-EM.png",
        caption="Estrutura do Vírus da Dengue (Flaviviridae)",
    )
    st.title("PMC - Classificador DENV")
    st.markdown("""
    **Trabalho Acadêmico de Graduação**
    * **Rede:** Perceptron Multicamadas (PMC)
    * **Entrada:** 64 Trímeros Canônicos ($k=3$)
    * **Classes:** DENV-1, DENV-2, DENV-3, DENV-4
    """)
    st.divider()

    st.subheader("⚙️ Critérios de Qualidade")
    st.markdown("""
    * **Tamanho do genoma:** $10.000$ a $11.500\\text{ pb}$
    * **Bases ambíguas (N):** Máximo de $1,0\\%$
    * **Normalização:** Z-score (`StandardScaler`)
    """)


# ==============================================================================
# CORPO PRINCIPAL E ABAS
# ==============================================================================

st.title("🧬 Classificação dos Sorotipos da Dengue via PMC")
st.markdown("Sistema inteligente para diagnóstico genômico e rastreamento de sorotipos virais via assinaturas de trímeros.")

tab_classificar, tab_dataset, tab_modelo = st.tabs([
    "🔍 Classificar Nova Amostra",
    "📊 Conjunto de Dados & Histórico",
    "🧠 Arquitetura & Diagnóstico do PMC",
])


# ------------------------------------------------------------------------------
# ABA 1: CLASSIFICAÇÃO DE NOVAS AMOSTRAS
# ------------------------------------------------------------------------------
with tab_classificar:
    st.subheader("1. Carregar Arquivo FASTA ou Colar Sequência")
    
    tipo_entrada = st.radio(
        "Formato de Entrada:",
        ["📁 Upload de Arquivo FASTA (.fasta / .fa)", "✏️ Colar Sequência de Texto"],
        horizontal=True,
    )

    amostras_para_analise = []
    nome_origem = "Entrada Manual"

    if "Upload" in tipo_entrada:
        arquivo_upload = st.file_uploader(
            "Selecione um arquivo FASTA de genoma completo:",
            type=["fasta", "fa", "txt", "fna"],
            help="Genomas completos (~10.500 pb) no formato FASTA padrão.",
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
            placeholder=">Amostra_01 Exemplo\nAGTTGTTAGTCTACGTGGACCGACAAGAACAGTTTCGAATCGGAAGCTTG...",
        )
        if texto_seq.strip():
            amostras_para_analise = processar_texto_fasta(texto_seq)

    # Botão de Execução
    if st.button("🚀 Executar Classificação pelo PMC", type="primary", use_container_width=True):
        if not amostras_para_analise:
            st.warning("Por favor, carregue um arquivo FASTA ou insira uma sequência para continuar.")
        else:
            novos_registros_historico = []
            resultados_exibicao = []
            timestamp_agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with st.spinner("Extraindo trímeros e processando inferência pela rede neural..."):
                for amostra in amostras_para_analise:
                    id_seq = amostra["id"]
                    seq_str = amostra["seq"]
                    comp_pb = len(seq_str)

                    valida, motivo = validar_sequencia(seq_str)
                    if not valida:
                        st.error(f"❌ **[{id_seq}] Rejeitada:** {motivo} (Tamanho lido: {comp_pb} pb)")
                        continue

                    # Extração dos 64 trímeros canônicos
                    vetor_features = np.array([extrair_frequencias_trimeros(seq_str)])
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
                st.success(f"✅ {len(novos_registros_historico)} sequência(s) classificada(s) e registradas no banco de dados!")

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
                    st.altair_chart(chart, use_container_width=True)

                # Se forem múltiplas sequências, mostra em tabela interativa
                else:
                    st.markdown("### Tabela de Classificação das Amostras:")
                    df_res = pd.DataFrame(resultados_exibicao)
                    st.dataframe(df_res, use_container_width=True)


# ------------------------------------------------------------------------------
# ABA 2: CONJUNTO DE DADOS & HISTÓRICO ACUMULADO
# ------------------------------------------------------------------------------
with tab_dataset:
    st.subheader("📈 Painel do Conjunto de Dados Acumulado")
    st.markdown("Monitoramento de todas as amostras submetidas e classificadas pelo modelo até o momento.")

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
            st.altair_chart(chart_hist, use_container_width=True)

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
                use_container_width=True,
            )

        st.markdown("### Registro Detalhado das Amostras:")
        st.dataframe(df_hist, use_container_width=True)

        # Opção de resetar histórico com confirmação
        with st.expander("⚠️ Opções de Manutenção do Banco de Dados"):
            if st.button("🗑️ Limpar Todo o Histórico de Amostras", type="secondary"):
                if os.path.exists(HISTORICO_CSV):
                    os.remove(HISTORICO_CSV)
                    st.rerun()


# ------------------------------------------------------------------------------
# ABA 3: INFORMAÇÕES DO MODELO PMC & GRÁFICOS CIENTÍFICOS
# ------------------------------------------------------------------------------
with tab_modelo:
    st.subheader("🧠 Metodologia e Desempenho do Perceptron Multicamadas (PMC)")
    st.markdown("""
    Este modelo foi treinado em um conjunto curado de mais de **3.100 genomas independentes**
    do NCBI GenBank e BV-BRC, avaliado em validação cruzada estratificada e teste cego ($100\\%$ de acurácia).
    """)

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists("matriz_confusao_final.png"):
            st.image("matriz_confusao_final.png", caption="Matriz de Confusão no Teste Cego (Absoluta e Normalizada)")
        if os.path.exists("curva_perda.png"):
            st.image("curva_perda.png", caption="Convergência da Função de Perda (Loss)")

    with col_img2:
        if os.path.exists("importancia_trimeros.png"):
            st.image("importancia_trimeros.png", caption="Importância das Características (Permutation Importance)")
        if os.path.exists("distribuicao_confianca.png"):
            st.image("distribuicao_confianca.png", caption="Distribuição de Calibração das Probabilidades")
