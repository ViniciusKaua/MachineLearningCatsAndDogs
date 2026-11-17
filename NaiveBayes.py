"""
Naive Bayes (GaussianNB, MultinomialNB e ComplementNB).

Etapas:
  1) Avalia os três classificadores Naive Bayes (com parâmetros default) em TODAS
     as bases da pasta 12bases, utilizando 10-fold CV e holdout 70/30.
  2) Para MultinomialNB e ComplementNB, aplica MinMaxScaler() via Pipeline
     (ajustado apenas no treino) para garantir valores não-negativos nas bases
     com PCA sem vazamento de dados.
  3) Gera a tabela final com média geral e desvio padrão geral (as 24 linhas:
     10-fold + 70/30), no mesmo formato do CSV do Decision Tree e da planilha limponb.xlsx.

Entrada : 12bases/ (CSVs com a coluna 'classe')
Saída   : resultados_nb/
            NB_tabela_final.csv  (tabela final: 12 bases x 3 classificadores)

Requisitos: pip install scikit-learn pandas numpy
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------
PASTA_BASES = "12bases"
PASTA_SAIDA = "resultados_nb"

# Opcional: se existir, segue a ordem das bases da tabela de referência.
TABELA_REFERENCIA = os.path.join("resultados_selecao", "tabela_final.csv")

N_FOLDS = 10
TAM_TESTE = 0.30       # holdout 70/30
SEMENTE = 42           # divisão dos folds e do holdout (reprodutível)
N_JOBS = -1

COL_CLASSE = "classe"
COLS_MODELOS = ["GaussianNB", "MultinomialNB", "ComplementNB"]
ROTULO_GERAL = "Geral (10-fold + 70/30)"


# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES
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
    """Segue a ordem da tabela de referência se existir; senão, ordem alfabética."""
    if os.path.exists(TABELA_REFERENCIA):
        ordem = pd.read_csv(TABELA_REFERENCIA)["Base"].dropna().drop_duplicates().tolist()
        ordenadas = [n for n in ordem if n in nomes]
        ordenadas += [n for n in nomes if n not in ordenadas]
        return ordenadas
    print(f"[Aviso] '{TABELA_REFERENCIA}' não encontrada: bases em ordem alfabética.")
    return list(nomes)


def carregar_base(caminho):
    df = pd.read_csv(caminho)
    y = df[COL_CLASSE].values
    X = df.drop(columns=[COL_CLASSE]).values
    return X, y


def criar_modelos():
    """
    Retorna os três classificadores Naive Bayes com parâmetros default.
    MultinomialNB e ComplementNB usam MinMaxScaler() no Pipeline porque bases
    com PCA possuem valores negativos, o que geraria erro nesses modelos.
    """
    return {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": make_pipeline(MinMaxScaler(), MultinomialNB()),
        "ComplementNB": make_pipeline(MinMaxScaler(), ComplementNB()),
    }


def acc_10fold(X, y, modelo):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
    return cross_val_score(
        modelo, X, y, cv=cv, scoring="accuracy", n_jobs=N_JOBS
    ).mean()


def acc_holdout(divisao, modelo):
    X_tr, X_te, y_tr, y_te = divisao
    return modelo.fit(X_tr, y_tr).score(X_te, y_te)


# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------
def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    nomes = listar_bases(PASTA_BASES)
    if not nomes:
        raise SystemExit(f"Nenhuma base (CSV com coluna '{COL_CLASSE}') em '{PASTA_BASES}'.")
    nomes = ordenar_bases(nomes)
    print(f"{len(nomes)} bases encontradas em '{PASTA_BASES}'.\n")

    # ---- 1) Avalia GaussianNB, MultinomialNB e ComplementNB em todas as bases ----
    linhas = []
    for pos, nome in enumerate(nomes, 1):
        Xb, yb = carregar_base(os.path.join(PASTA_BASES, nome))
        divisao = train_test_split(
            Xb, yb, test_size=TAM_TESTE, stratify=yb, random_state=SEMENTE
        )

        accs_cv = {}
        accs_ho = {}
        for nome_modelo, modelo in criar_modelos().items():
            accs_cv[nome_modelo] = acc_10fold(Xb, yb, modelo)
            accs_ho[nome_modelo] = acc_holdout(divisao, modelo)

        base = {"#": pos, "Base": nome, "Atributos": Xb.shape[1]}
        linhas.append({**base, "Treinamento/Teste": "10-fold CV", **accs_cv})
        linhas.append({**base, "Treinamento/Teste": "70/30", **accs_ho})

        print(
            f"[{pos}/{len(nomes)}] {nome} ({Xb.shape[1]} atrib.) | "
            f"10-fold -> GNB: {accs_cv['GaussianNB']:.4f}, MNB: {accs_cv['MultinomialNB']:.4f}, CNB: {accs_cv['ComplementNB']:.4f} | "
            f"70/30 -> GNB: {accs_ho['GaussianNB']:.4f}, MNB: {accs_ho['MultinomialNB']:.4f}, CNB: {accs_ho['ComplementNB']:.4f}"
        )

    # ---- 2) Tabela final com média e desvio padrão gerais ----
    df = pd.DataFrame(linhas)
    geral = df[COLS_MODELOS]  # as 24 linhas (10-fold + 70/30)
    resumo = pd.DataFrame([
        {"Base": f"Média ({ROTULO_GERAL})", **geral.mean().to_dict()},
        {"Base": f"Desv. Pad. ({ROTULO_GERAL})", **geral.std(ddof=1).to_dict()},
    ])
    tabela = pd.concat([df, resumo], ignore_index=True)
    tabela["#"] = tabela["#"].astype("Int64")
    tabela["Atributos"] = tabela["Atributos"].astype("Int64")

    saida = os.path.join(PASTA_SAIDA, "NB_tabela_final.csv")
    tabela.round(4).to_csv(saida, index=False)

    print(f"\nConcluído. Arquivo salvo em '{saida}'.\n")
    print(tabela.tail(2).round(4).to_string(index=False))


if __name__ == "__main__":
    main()