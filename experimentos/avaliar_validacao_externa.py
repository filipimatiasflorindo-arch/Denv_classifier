import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PMC import (
    DIR_DADOS_TESTES,
    DIR_PREDICOES,
    K_MER_SIZE,
    carregar_pipeline,
    extrair_frequencias_kmers,
)

def parse_header_organism(header: str):
    """
    Extrai com alta precisão o sorotipo a partir do cabeçalho FASTA,
    dando prioridade à definição formal do organismo do NCBI (ex: 'Dengue virus type 2').
    """
    h_lower = header.lower()
    
    # 1. Definição formal do NCBI no início da descrição
    # 'dengue virus type X' ou 'dengue virus type [I,II,III,IV]'
    m_ncbi = re.search(r'dengue\s+virus\s+type\s+([1-4]|iv|iii|ii|i)\b', h_lower)
    if m_ncbi:
        val = m_ncbi.group(1)
        mapa = {'1': 'DENV-1', 'i': 'DENV-1',
                '2': 'DENV-2', 'ii': 'DENV-2',
                '3': 'DENV-3', 'iii': 'DENV-3',
                '4': 'DENV-4', 'iv': 'DENV-4'}
        return mapa.get(val, 'UNKNOWN'), False

    # 2. 'dengue virus X' ou 'dengue-X virus'
    m_dengue_num = re.search(r'dengue(?:-|\s+virus\s+|\s+)([1-4])\b', h_lower)
    if m_dengue_num and 'isolate' not in h_lower[:m_dengue_num.start()]:
        val = m_dengue_num.group(1)
        return f'DENV-{val}', False

    # 3. 'serotype X' ou 'serX'
    m_ser = re.search(r'ser(?:otype)?\s*([1-4])\b', h_lower)
    if m_ser:
        return f'DENV-{m_ser.group(1)}', False

    # 4. 'isolate denv-X' ou 'isolate denX' ou 'd1/', 'd2/'
    m_denv = re.search(r'\bdenv-?([1-4])\b', h_lower)
    if m_denv:
        return f'DENV-{m_denv.group(1)}', False

    # 5. Isolate dX/
    m_dx = re.search(r'\bd([1-4])/', h_lower)
    if m_dx:
        return f'DENV-{m_dx.group(1)}', False

    # Checar se é patente ou diagnóstico
    is_patent = any(w in h_lower for w in ['patent', 'jp 20', 'kr 10', 'us 1', 'wo20', 'crispr', 'assay', 'therapeutic', 'vaccine', 'unverified'])
    return 'UNKNOWN', is_patent

def main():
    print("=" * 80)
    print("BENCHMARK COMPLETO: AVALIAÇÃO DE 'teste_aleatorio.fasta' NO PMC")
    print("=" * 80)

    pipe = carregar_pipeline()
    classes = list(pipe.named_steps['pmc'].classes_)

    caminho_fasta = os.path.join(DIR_DADOS_TESTES, 'teste_aleatorio.fasta')
    records = []
    with open(caminho_fasta, 'r', encoding='utf-8', errors='ignore') as f:
        cur_id, cur_desc, cur_seq = '', '', []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur_id:
                    records.append((cur_id, cur_desc, ''.join(cur_seq)))
                    cur_seq = []
                parts = line[1:].split(None, 1)
                cur_id = parts[0]
                cur_desc = line[1:]
            else:
                cur_seq.append(line)
        if cur_id:
            records.append((cur_id, cur_desc, ''.join(cur_seq)))

    print(f"Total de registros lidos no FASTA: {len(records):,}")

    rows = []
    for r_id, desc, seq in records:
        comp = len(seq)
        label_real, is_patent = parse_header_organism(desc)
        
        if comp < K_MER_SIZE:
            rows.append({
                'ID': r_id, 'Descricao': desc, 'Tamanho': comp, 'Label_Real': label_real,
                'Is_Patent': is_patent, 'Pred_PMC': f'N/A (< {K_MER_SIZE}pb)', 'Confianca': 0.0,
                **{f'P({c})': 0.0 for c in classes}
            })
            continue

        feat = extrair_frequencias_kmers(seq, K_MER_SIZE).reshape(1, -1)
        pred = pipe.predict(feat)[0]
        probas = pipe.predict_proba(feat)[0]
        p_dict = {c: float(np.round(p * 100, 2)) for c, p in zip(classes, probas)}
        conf = p_dict[pred]

        rows.append({
            'ID': r_id, 'Descricao': desc, 'Tamanho': comp, 'Label_Real': label_real,
            'Is_Patent': is_patent, 'Pred_PMC': pred, 'Confianca': conf,
            **{f'P({c})': p_dict[c] for c in classes}
        })

    os.makedirs(DIR_PREDICOES, exist_ok=True)
    caminho_csv = os.path.join(DIR_PREDICOES, 'predicoes_fragmentos_teste_aleatorio.csv')
    df = pd.DataFrame(rows)
    df.to_csv(caminho_csv, index=False)
    print(f"Predições salvas com sucesso em '{caminho_csv}'.")

    # Análise Estatística
    df_val = df[df['Label_Real'].isin(classes) & (~df['Is_Patent'])].copy()
    print(f"\nSequências validadas com anotação taxonômica não-ambígua: {len(df_val):,} / {len(df):,}")

    acc = accuracy_score(df_val['Label_Real'], df_val['Pred_PMC']) * 100
    bal_acc = balanced_accuracy_score(df_val['Label_Real'], df_val['Pred_PMC']) * 100
    print(f"Acurácia Global (Geral):     {acc:.2f}%")
    print(f"Acurácia Balanceada (Geral): {bal_acc:.2f}%")

    print("\nRelatório de Classificação por Sorotipo:")
    print(classification_report(df_val['Label_Real'], df_val['Pred_PMC'], target_names=classes, digits=4))

    cm = confusion_matrix(df_val['Label_Real'], df_val['Pred_PMC'], labels=classes)
    cm_df = pd.DataFrame(cm, index=[f'Real {c}' for c in classes], columns=[f'Pred {c}' for c in classes])
    print("Matriz de Confusão:")
    print(cm_df)

    # Estratificação por comprimento (intervalos semiabertos [a, b) padronizados com os demais scripts)
    bins = [0, 300, 1000, 5000, np.inf]
    strata_labels = [
        '< 300 pb (Ultra-curtos / Reads parciais)',
        '300 - 1.000 pb (Amplicons / PCRs)',
        '1.000 - 5.000 pb (Genes completos / E, NS1)',
        '>= 5.000 pb (Genomas quase completos / completos)'
    ]
    df_val['Faixa_Tamanho'] = pd.cut(df_val['Tamanho'], bins=bins, labels=strata_labels, right=False)

    print("\n" + "=" * 80)
    print("DESEMPENHO DO PMC ESTRATIFICADO POR COMPRIMENTO:")
    print("=" * 80)

    for strata in strata_labels:
        sub = df_val[df_val['Faixa_Tamanho'] == strata]
        if len(sub) == 0:
            continue
        sub_acc = accuracy_score(sub['Label_Real'], sub['Pred_PMC']) * 100
        sub_bal_acc = balanced_accuracy_score(sub['Label_Real'], sub['Pred_PMC']) * 100
        sub_conf = sub['Confianca'].mean()
        erros = len(sub[sub['Label_Real'] != sub['Pred_PMC']])
        print(f"Faixa: {strata}")
        print(f"  - Total: {len(sub):,} sequências | Acertos: {len(sub)-erros:,} | Erros: {erros:,}")
        print(f"  - Acurácia: {sub_acc:.2f}% | Acurácia Balanceada: {sub_bal_acc:.2f}% | Confiança Média: {sub_conf:.2f}%\n")

    # Detalhar os erros em sequências longas (> 1.000 pb)
    erros_longos = df_val[(df_val['Tamanho'] >= 1000) & (df_val['Label_Real'] != df_val['Pred_PMC'])]
    print("=" * 80)
    print(f"AUDITORIA DETALHADA: ERROS EM SEQUÊNCIAS LONGAS (>= 1.000 pb): {len(erros_longos)}")
    print("=" * 80)
    for idx, r in erros_longos.iterrows():
        print(f"ID: {r['ID']} | Tam: {r['Tamanho']} pb | Real: {r['Label_Real']} | Pred: {r['Pred_PMC']} | Conf: {r['Confianca']}%")
        print(f"   Descrição: {r['Descricao']}")
        print(f"   Probabilidades: P(DENV-1)={r['P(DENV-1)']}% | P(DENV-2)={r['P(DENV-2)']}% | P(DENV-3)={r['P(DENV-3)']}% | P(DENV-4)={r['P(DENV-4)']}%")
        print("-" * 60)

    # Análise das sequências UNKNOWN e Patentes
    outros = df[(df['Label_Real'] == 'UNKNOWN') | (df['Is_Patent'])]
    print(f"\nSequências Atípicas / Patentes / Sem sorotipo explícito no header: {len(outros):,}")
    print("Distribuição das predições do PMC para sequências atípicas/patentes:")
    print(outros['Pred_PMC'].value_counts())

if __name__ == '__main__':
    main()

