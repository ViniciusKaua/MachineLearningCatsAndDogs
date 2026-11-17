"""
Comparação final entre os melhores modelos dos 4 comitês de classificadores
(Bagging, Random Forest, Stacking e Voting) - Prática de Comitês.

Etapas (conforme o último parágrafo do PDF):
  1) Lê, em cada pasta resultados_*, a tabela final (*_tabela_final.csv) e a análise
     estatística (*_analise_estatistica.csv) e identifica o modelo marcado como
     'Escolhido' em cada estrutura de comitê.
  2) Monta a tabela comparativa dos 4 campeões nas 12 bases (10-fold CV e 70/30),
     com média e desvio padrão gerais calculados sobre as 24 linhas
     (10-fold + 70/30), igual às tabelas anteriores.
  3) Análise estatística entre os 4 campeões:
       - Teste de Friedman (há diferença estatística?)
       - Havendo diferença: melhor modelo = melhor rank médio (pós-teste de
         Nemenyi informa quais pares diferem).
       - Não havendo: melhor modelo = menor complexidade, medida pelo nº de
         classificadores base do comitê (o número entre parênteses no nome do
         modelo, ex.: 'k-NN (10)', 'Gini (100)', 'Classificadores (5)'); entre os
         de mesma complexidade, maior acurácia média.

Entrada : resultados_bagging/   Bagging_tabela_final.csv   Bagging_analise_estatistica.csv
          resultados_rf/        RF_tabela_final.csv        RF_analise_estatistica.csv
          resultados_stacking/  Stacking_tabela_final.csv  Stacking_analise_estatistica.csv
          resultados_voting/    Voting_tabela_final.csv    Voting_analise_estatistica.csv
Saída   : resultados_comparacao/
            Comparacao_tabela_final.csv          (tabela comparativa dos 4 campeões)
            Comparacao_campeoes.csv              (qual modelo foi escolhido em cada comitê)
            Comparacao_analise_estatistica.csv   (rank médio, acurácia média, escolhido)
            Comparacao_analise_estatistica.txt   (Friedman, Nemenyi e decisão)

Requisitos: pip install pandas numpy scipy
"""
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range

# ----------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------
PASTA_SAIDA = "resultados_comparacao"

# nome do comitê -> (pasta de resultados, prefixo dos arquivos)
COMITES = {
    "Bagging":       ("resultados_bagging",  "Bagging"),
    "Random Forest": ("resultados_rf",       "RF"),
    "Stacking":      ("resultados_stacking", "Stacking"),
    "Voting":        ("resultados_voting",   "Voting"),
}

COLS_ID = ["#", "Base", "Atributos", "Treinamento/Teste"]
ROTULO_GERAL = "Geral (10-fold + 70/30)"

# Análise estatística
ALFA = 0.05
# Quais acurácias entram nos testes (1 valor por base e por modelo):
#   "10-fold CV" | "70/30" | "media" (média dos dois)
DADOS_TESTE = "10-fold CV"


# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES (mesmo padrão dos scripts anteriores)
# ----------------------------------------------------------------------
def n_classificadores(modelo):
    """Complexidade = nº de classificadores base: o número entre parênteses no fim
    do nome do modelo ('k-NN (10)' -> 10, 'Gini (100)' -> 100)."""
    m = re.search(r"\((\d+)\)\s*$", modelo)
    if not m:
        raise SystemExit(f"Não consegui extrair o nº de classificadores de '{modelo}'.")
    return int(m.group(1))


def carregar_comite(nome, pasta, prefixo):
    """Retorna (modelo escolhido, linhas das 12 bases x 2 tipos de teste)."""
    cam_tab = os.path.join(pasta, f"{prefixo}_tabela_final.csv")
    cam_est = os.path.join(pasta, f"{prefixo}_analise_estatistica.csv")
    for cam in (cam_tab, cam_est):
        if not os.path.exists(cam):
            raise SystemExit(f"Arquivo não encontrado: '{cam}' (rode antes o script do {nome}).")

    est = pd.read_csv(cam_est)
    escolhido = est[est["Escolhido"].astype(str).str.lower() == "true"]
    if len(escolhido) != 1:
        raise SystemExit(f"'{cam_est}' deve ter exatamente 1 modelo com Escolhido = True "
                         f"(encontrei {len(escolhido)}).")
    modelo = str(escolhido.iloc[0]["Modelo"])

    tab = pd.read_csv(cam_tab)
    bases = tab[tab["#"].notna()].copy()  # descarta as linhas de Média e Desv. Pad.
    if modelo not in bases.columns:
        raise SystemExit(f"Coluna '{modelo}' não existe em '{cam_tab}'. "
                         f"Colunas: {[c for c in bases.columns if c not in COLS_ID]}")
    return modelo, bases


def matriz_para_testes(df, colunas):
    """Matriz (bases x modelos) com a acurácia usada nos testes."""
    cv = df[df["Treinamento/Teste"] == "10-fold CV"][colunas].reset_index(drop=True)
    ho = df[df["Treinamento/Teste"] == "70/30"][colunas].reset_index(drop=True)
    if DADOS_TESTE == "10-fold CV":
        return cv
    if DADOS_TESTE == "70/30":
        return ho
    return (cv + ho) / 2


# ----------------------------------------------------------------------
# ANÁLISE ESTATÍSTICA
# ----------------------------------------------------------------------
def analise_estatistica(matriz, campeoes):
    """Friedman + Nemenyi + decisão. Retorna (tabela_modelos, texto)."""
    colunas = list(matriz.columns)
    N, k = matriz.shape
    stat, p = friedmanchisquare(*[matriz[c].values for c in colunas])

    # Rank 1 = melhor (maior acurácia); empates recebem o rank médio
    ranks = np.vstack([rankdata(-linha) for linha in matriz.values])
    rank_medio = ranks.mean(axis=0)

    # Distância crítica de Nemenyi
    q_alfa = studentized_range.ppf(1 - ALFA, k, np.inf) / np.sqrt(2)
    cd = q_alfa * np.sqrt(k * (k + 1) / (6 * N))

    tab = pd.DataFrame({
        "Modelo": colunas,
        "Comite": [c["comite"] for c in campeoes],
        "Modelo_escolhido": [c["modelo"] for c in campeoes],
        "n_classificadores": [c["n_class"] for c in campeoes],
        "Acc_media": matriz.mean().values,
        "Desv_Pad": matriz.std(ddof=1).values,
        "Rank_medio": rank_medio,
        "Complexidade": [c["n_class"] for c in campeoes],
    })
    melhor_rank = tab["Rank_medio"].min()
    tab["Dif_rank_p_melhor"] = tab["Rank_medio"] - melhor_rank
    tab["Equivalente_ao_melhor_rank"] = tab["Dif_rank_p_melhor"] <= cd

    houve_diferenca = p < ALFA
    if houve_diferenca:
        idx = tab.sort_values(["Rank_medio", "Acc_media", "Complexidade"],
                              ascending=[True, False, True]).index[0]
        criterio = "Friedman significativo -> melhor rank médio."
    else:
        idx = tab.sort_values(["Complexidade", "Acc_media"],
                              ascending=[True, False]).index[0]
        criterio = ("Sem diferença estatística -> menor complexidade (menor nº de "
                    "classificadores base); entre os de mesma complexidade, maior "
                    "acurácia média.")
    tab["Escolhido"] = False
    tab.loc[idx, "Escolhido"] = True
    escolhido = tab.loc[idx, "Modelo"]
    tab = tab.sort_values("Rank_medio").reset_index(drop=True)

    # Pares que diferem pelo Nemenyi (só faz sentido se Friedman for significativo)
    pares = []
    if houve_diferenca:
        for i in range(k):
            for j in range(i + 1, k):
                d = abs(rank_medio[i] - rank_medio[j])
                if d > cd:
                    melhor, pior = (colunas[i], colunas[j]) if rank_medio[i] < rank_medio[j] \
                        else (colunas[j], colunas[i])
                    pares.append(f"  {melhor} > {pior} (dif. de rank = {d:.2f})")

    linhas = [
        "ANÁLISE ESTATÍSTICA - COMPARAÇÃO ENTRE OS 4 COMITÊS",
        f"Dados usados: acurácia {DADOS_TESTE} | {N} bases x {k} modelos | alfa = {ALFA}",
        "",
        "Modelos escolhidos em cada etapa anterior:",
    ]
    linhas += [f"  - {c['comite']}: {c['modelo']}" for c in campeoes]
    linhas += [
        "",
        f"Teste de Friedman: estatística = {stat:.4f} | p-valor = {p:.6f}",
        "Há diferença estatística entre os modelos: " + ("SIM" if houve_diferenca else "NÃO"),
        f"Distância crítica de Nemenyi (CD) = {cd:.3f}",
    ]
    if houve_diferenca:
        linhas.append(f"Pares com diferença significativa (|dif. de rank| > CD): {len(pares)}")
        linhas += pares if pares else ["  (nenhum par individual ultrapassou o CD)"]
    linhas += [
        "",
        f"Critério de escolha: {criterio}",
        f"MELHOR MODELO GERAL: {escolhido}",
        "",
        "Obs.: a complexidade considerada é o nº de classificadores base do comitê.",
    ]
    return tab, "\n".join(linhas)


# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------
def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    campeoes, colunas, df = [], [], None
    for nome, (pasta, prefixo) in COMITES.items():
        modelo, bases = carregar_comite(nome, pasta, prefixo)
        coluna = f"{nome} [{modelo}]"
        campeoes.append({"comite": nome, "modelo": modelo, "coluna": coluna,
                         "n_class": n_classificadores(modelo)})
        colunas.append(coluna)

        if df is None:  # a 1ª estrutura define a ordem das bases
            df = bases[COLS_ID].copy().reset_index(drop=True)

        # Alinha pela chave (Base, Treinamento/Teste) para garantir correspondência exata
        mapa = bases.set_index(["Base", "Treinamento/Teste"])[modelo].astype(float)
        chaves = pd.MultiIndex.from_frame(df[["Base", "Treinamento/Teste"]])
        valores = mapa.reindex(chaves).values
        if np.isnan(valores).any():
            raise SystemExit(f"As bases de '{nome}' não coincidem com as do Bagging.")
        df[coluna] = valores

    print(f"{df['Base'].nunique()} bases | {len(colunas)} comitês campeões:")
    for c in campeoes:
        print(f"  - {c['comite']}: {c['modelo']}")
    print()
    pd.DataFrame(campeoes).rename(columns={"comite": "Comitê", "modelo": "Modelo escolhido",
                                           "coluna": "Coluna", "n_class": "n_classificadores"}
                                  ).to_csv(os.path.join(PASTA_SAIDA, "Comparacao_campeoes.csv"),
                                           index=False)

    # ---- Tabela final com média e desvio padrão gerais (24 linhas) ----
    geral = df[colunas]
    resumo = pd.DataFrame([
        {"Base": f"Média ({ROTULO_GERAL})", **geral.mean().to_dict()},
        {"Base": f"Desv. Pad. ({ROTULO_GERAL})", **geral.std(ddof=1).to_dict()},
    ])
    tabela = pd.concat([df, resumo], ignore_index=True)
    tabela["#"] = tabela["#"].astype("Int64")
    tabela["Atributos"] = tabela["Atributos"].astype("Int64")
    tabela.round(4).to_csv(os.path.join(PASTA_SAIDA, "Comparacao_tabela_final.csv"), index=False)

    # ---- Análise estatística ----
    tab_est, texto = analise_estatistica(matriz_para_testes(df, colunas), campeoes)
    tab_est.round(4).to_csv(os.path.join(PASTA_SAIDA, "Comparacao_analise_estatistica.csv"),
                            index=False)
    with open(os.path.join(PASTA_SAIDA, "Comparacao_analise_estatistica.txt"), "w",
              encoding="utf-8") as f:
        f.write(texto + "\n")

    print(f"Concluído. Arquivos em '{PASTA_SAIDA}/'.\n")
    print(tabela.tail(2).round(4).to_string(index=False))
    print("\n" + texto)


if __name__ == "__main__":
    main()