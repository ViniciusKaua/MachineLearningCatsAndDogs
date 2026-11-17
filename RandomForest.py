"""
Random Forest (RandomForestClassifier) - Prática de Comitês de Classificadores.

Etapas (conforme o PDF, Figura 2):
  1) criterion = gini com n_estimators em {10, 20, 30, 100}; repete com
     criterion = entropy -> 8 modelos. Demais parâmetros em valores default.
  2) Cada modelo é avaliado nas 12 bases (pasta 12bases) com 10-fold CV e
     holdout 70/30 -> tabela da Figura 2 (com média e desvio padrão gerais,
     calculados sobre as 24 linhas: 10-fold + 70/30).
  3) Análise estatística entre os 8 modelos:
       - Teste de Friedman (há diferença estatística?)
       - Havendo diferença: melhor modelo = melhor rank médio (pós-teste de
         Nemenyi informa quais pares diferem).
       - Não havendo: melhor modelo = menor complexidade, medida pelo nº de
         árvores (n_estimators); entre os de mesma complexidade, maior
         acurácia média.

Entrada : 12bases/ (CSVs com a coluna 'classe')
Saída   : resultados_rf/
            RF_tabela_final.csv           (Figura 2 preenchida)
            RF_analise_estatistica.csv    (rank médio, acurácia média, escolhido)
            RF_analise_estatistica.txt    (Friedman, Nemenyi e decisão)
            RF_parcial.csv                (cache interno p/ retomar; pode apagar no fim)

Requisitos: pip install scikit-learn pandas numpy scipy
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# ----------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------
PASTA_BASES = "12bases"
PASTA_SAIDA = "resultados_rf"

# Opcional: se existir, a ordem das bases na tabela segue a do k-NN.
TABELA_KNN = os.path.join("resultados_selecao", "tabela_final.csv")

CRITERIOS = ["gini", "entropy"]
N_ESTIMADORES = [10, 20, 30, 100]
N_FOLDS = 10
TAM_TESTE = 0.30            # holdout 70/30
SEMENTE = 42                # folds, holdout e Random Forest (reprodutível)
N_JOBS = -1

# Análise estatística
ALFA = 0.05
# Quais acurácias entram nos testes (1 valor por base e por modelo):
#   "10-fold CV" | "70/30" | "media" (média dos dois)
DADOS_TESTE = "10-fold CV"
# Complexidade do modelo = n_estimators (menos árvores = mais simples);
# usada só se NÃO houver diferença estatística. Sem pesos por critério.

COL_CLASSE = "classe"
ROTULO_GERAL = "Geral (10-fold + 70/30)"

# Ordem das colunas igual à Figura 2 do PDF: Gini (10, 20, 30, 100), Entropy (...)
CONFIGS = [(crit, n) for crit in CRITERIOS for n in N_ESTIMADORES]
COLUNAS = [f"{crit.capitalize()} ({n})" for crit, n in CONFIGS]


# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES (mesmo padrão dos scripts anteriores)
# ----------------------------------------------------------------------
def listar_bases(pasta):
    """CSVs da pasta que tenham a coluna 'classe' (ignora outros arquivos)."""
    nomes = []
    for nome in sorted(os.listdir(pasta)):
        if not nome.lower().endswith(".csv"):
            continue
        colunas = pd.read_csv(os.path.join(pasta, nome), nrows=0).columns
        if COL_CLASSE in colunas:
            nomes.append(nome)
    return nomes


def ordenar_bases(nomes):
    """Segue a ordem do k-NN se a tabela existir; senão, ordem alfabética."""
    if os.path.exists(TABELA_KNN):
        ordem = pd.read_csv(TABELA_KNN)["Base"].dropna().drop_duplicates().tolist()
        ordenadas = [n for n in ordem if n in nomes]
        ordenadas += [n for n in nomes if n not in ordenadas]
        return ordenadas
    print(f"[Aviso] '{TABELA_KNN}' não encontrada: bases em ordem alfabética.")
    return list(nomes)


def carregar_base(caminho):
    df = pd.read_csv(caminho)
    y = df[COL_CLASSE].values
    X = df.drop(columns=[COL_CLASSE]).values
    return X, y


def floresta(criterio, n_estimators):
    # demais parâmetros ficam no default
    return RandomForestClassifier(criterion=criterio, n_estimators=n_estimators,
                                  random_state=SEMENTE)


def acc_10fold(X, y, criterio, n_estimators):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
    return cross_val_score(floresta(criterio, n_estimators), X, y, cv=cv,
                           scoring="accuracy", n_jobs=N_JOBS).mean()


def acc_holdout(divisao, criterio, n_estimators):
    X_tr, X_te, y_tr, y_te = divisao
    return floresta(criterio, n_estimators).fit(X_tr, y_tr).score(X_te, y_te)


# ----------------------------------------------------------------------
# ANÁLISE ESTATÍSTICA
# ----------------------------------------------------------------------
def matriz_para_testes(df):
    """Matriz (bases x modelos) com a acurácia usada nos testes."""
    cv = df[df["Treinamento/Teste"] == "10-fold CV"][COLUNAS].reset_index(drop=True)
    ho = df[df["Treinamento/Teste"] == "70/30"][COLUNAS].reset_index(drop=True)
    if DADOS_TESTE == "10-fold CV":
        return cv
    if DADOS_TESTE == "70/30":
        return ho
    return (cv + ho) / 2


def analise_estatistica(matriz):
    """Friedman + Nemenyi + decisão. Retorna (tabela_modelos, texto)."""
    N, k = matriz.shape
    stat, p = friedmanchisquare(*[matriz[c].values for c in COLUNAS])

    # Rank 1 = melhor (maior acurácia); empates recebem o rank médio
    ranks = np.vstack([rankdata(-linha) for linha in matriz.values])
    rank_medio = ranks.mean(axis=0)

    # Distância crítica de Nemenyi
    q_alfa = studentized_range.ppf(1 - ALFA, k, np.inf) / np.sqrt(2)
    cd = q_alfa * np.sqrt(k * (k + 1) / (6 * N))

    tab = pd.DataFrame({
        "Modelo": COLUNAS,
        "criterion": [c[0] for c in CONFIGS],
        "n_estimators": [c[1] for c in CONFIGS],
        "Acc_media": matriz.mean().values,
        "Rank_medio": rank_medio,
        "Complexidade": [n for _, n in CONFIGS],
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
        criterio = ("Sem diferença estatística -> menor complexidade (menor "
                    "n_estimators); entre os de mesma complexidade, maior "
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
                    melhor, pior = (COLUNAS[i], COLUNAS[j]) if rank_medio[i] < rank_medio[j] \
                        else (COLUNAS[j], COLUNAS[i])
                    pares.append(f"  {melhor} > {pior} (dif. de rank = {d:.2f})")

    linhas = [
        "ANÁLISE ESTATÍSTICA - RANDOM FOREST",
        f"Dados usados: acurácia {DADOS_TESTE} | {N} bases x {k} modelos | alfa = {ALFA}",
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
        f"MODELO ESCOLHIDO: Random Forest {escolhido}",
        "",
        "Obs.: a complexidade considerada é o nº de árvores (n_estimators).",
    ]
    return tab, "\n".join(linhas)


# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------
def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    nomes = listar_bases(PASTA_BASES)
    if not nomes:
        raise SystemExit(f"Nenhuma base (CSV com coluna '{COL_CLASSE}') em '{PASTA_BASES}'.")
    nomes = ordenar_bases(nomes)
    print(f"{len(nomes)} bases em '{PASTA_BASES}' | {len(CONFIGS)} modelos de Random Forest\n")

    # ---- Avaliação (com retomada) ----
    parcial = os.path.join(PASTA_SAIDA, "RF_parcial.csv")
    if os.path.exists(parcial):
        linhas = pd.read_csv(parcial).to_dict("records")
        feitas = {l["Base"] for l in linhas}
        print(f"Retomando: {len(feitas)} bases já avaliadas.\n")
    else:
        linhas, feitas = [], set()

    for pos, nome in enumerate(nomes, 1):
        if nome in feitas:
            continue
        t0 = time.time()
        X, y = carregar_base(os.path.join(PASTA_BASES, nome))
        divisao = train_test_split(X, y, test_size=TAM_TESTE, stratify=y,
                                   random_state=SEMENTE)
        accs_cv = [acc_10fold(X, y, c, n) for c, n in CONFIGS]
        accs_ho = [acc_holdout(divisao, c, n) for c, n in CONFIGS]

        base = {"#": pos, "Base": nome, "Atributos": X.shape[1]}
        linhas.append({**base, "Treinamento/Teste": "10-fold CV", **dict(zip(COLUNAS, accs_cv))})
        linhas.append({**base, "Treinamento/Teste": "70/30", **dict(zip(COLUNAS, accs_ho))})
        pd.DataFrame(linhas).to_csv(parcial, index=False)  # salva a cada base
        print(f"[{pos}/{len(nomes)}] {nome} ({X.shape[1]} atrib.) "
              f"10-fold: {np.mean(accs_cv):.4f} | 70/30: {np.mean(accs_ho):.4f} "
              f"({time.time() - t0:.0f}s)")

    # ---- Tabela final (Figura 2) com média e desvio padrão gerais ----
    df = pd.DataFrame(linhas)
    geral = df[COLUNAS]  # as 24 linhas (10-fold + 70/30)
    resumo = pd.DataFrame([
        {"Base": f"Média ({ROTULO_GERAL})", **geral.mean().to_dict()},
        {"Base": f"Desv. Pad. ({ROTULO_GERAL})", **geral.std(ddof=1).to_dict()},
    ])
    tabela = pd.concat([df, resumo], ignore_index=True)
    tabela["#"] = tabela["#"].astype("Int64")
    tabela["Atributos"] = tabela["Atributos"].astype("Int64")
    tabela.round(4).to_csv(os.path.join(PASTA_SAIDA, "RF_tabela_final.csv"), index=False)

    # ---- Análise estatística ----
    tab_est, texto = analise_estatistica(matriz_para_testes(df))
    tab_est.round(4).to_csv(os.path.join(PASTA_SAIDA, "RF_analise_estatistica.csv"),
                            index=False)
    with open(os.path.join(PASTA_SAIDA, "RF_analise_estatistica.txt"), "w",
              encoding="utf-8") as f:
        f.write(texto + "\n")

    print(f"\nConcluído. Arquivos em '{PASTA_SAIDA}/'.\n")
    print(tabela.tail(2).round(4).to_string(index=False))
    print("\n" + texto)


if __name__ == "__main__":
    main()