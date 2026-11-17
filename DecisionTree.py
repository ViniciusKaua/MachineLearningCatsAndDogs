"""
Árvore de Decisão (DecisionTreeClassifier) - Aula 16.

Etapas (conforme o PDF):
  1) GridSearch (criterion x max_depth = 3 x 14 = 42 configurações) com 10-fold CV
     na MELHOR base do k-NN (MELHOR_BASE).
  2) Ordena as 42 acurácias de forma decrescente e fica com as 10 melhores
     configurações (Conf. 1 a Conf. 10).
  3) Roda essas 10 configurações em TODAS as bases da pasta 12bases, com
     10-fold CV e com holdout 70/30.
  4) Tabela final (Figura 5 do PDF) com média geral e desvio padrão geral
     (as 24 linhas: 10-fold + 70/30), no mesmo formato do CSV do k-NN.

Entrada : 12bases/ (CSVs com a coluna 'classe')
Saída   : resultados_dt/
            DT_tabela_final.csv              (tabela final: 12 bases x 10 configurações)
            DT_top10_configuracoes.csv       (Conf., Acc, criterion, max_depth)
            DT_gridsearch_42_resultados.csv  (as 42 acurácias do grid, em ordem decrescente)

Requisitos: pip install scikit-learn pandas numpy   (scikit-learn >= 1.3 para 'log_loss')
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------
PASTA_BASES = "12bases"
PASTA_SAIDA = "resultados_dt"
MELHOR_BASE = "HOG_128_32x32_PCA-75.csv"   # melhor base do k-NN

# Opcional: se existir, a ordem das bases na tabela segue a do k-NN.
TABELA_KNN = os.path.join("resultados_selecao", "tabela_final.csv")

# GridSearch (Figura 1 do PDF)
par_criterion = ["gini", "entropy", "log_loss"]
par_max_depth = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
N_CONFIGS = 10

N_FOLDS = 10
TAM_TESTE = 0.30       # holdout 70/30
SEMENTE = 42           # divisão dos folds e do holdout (reprodutível)
# Semente da própria árvore. ATENÇÃO: em scikit-learn, 'entropy' e 'log_loss' são o
# MESMO critério; com semente fixa os dois dão resultados idênticos.
# Use None para deixar a árvore sem semente (como no exemplo do PDF, onde os dois
# diferem um pouco), ao custo de resultados não reproduzíveis a cada execução.
SEMENTE_ARVORE = 42
N_JOBS = -1

COL_CLASSE = "classe"
COLS_CONF = [f"Conf. {i}" for i in range(1, N_CONFIGS + 1)]
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


def arvore(criterio, profundidade):
    # Demais parâmetros em valores default
    return DecisionTreeClassifier(criterion=criterio, max_depth=profundidade,
                                  random_state=SEMENTE_ARVORE)


def acc_10fold(X, y, criterio, profundidade):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
    return cross_val_score(arvore(criterio, profundidade), X, y, cv=cv,
                           scoring="accuracy", n_jobs=N_JOBS).mean()


def acc_holdout(divisao, criterio, profundidade):
    X_tr, X_te, y_tr, y_te = divisao
    return arvore(criterio, profundidade).fit(X_tr, y_tr).score(X_te, y_te)


# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------
def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    nomes = listar_bases(PASTA_BASES)
    if not nomes:
        raise SystemExit(f"Nenhuma base (CSV com coluna '{COL_CLASSE}') em '{PASTA_BASES}'.")
    if MELHOR_BASE not in nomes:
        raise SystemExit(f"'{MELHOR_BASE}' não está em '{PASTA_BASES}'.")
    nomes = ordenar_bases(nomes)
    print(f"{len(nomes)} bases em '{PASTA_BASES}'. Melhor base do k-NN: {MELHOR_BASE}\n")

    # ---- 1) GridSearch na melhor base (10-fold CV) ----
    X, y = carregar_base(os.path.join(PASTA_BASES, MELHOR_BASE))
    resultados = []
    for i in par_criterion:
        for j in par_max_depth:
            acc = acc_10fold(X, y, i, j)
            resultados.append({"criterion": i, "max_depth": j, "Acc": acc})
            print(f"Accuracy: {acc:.3f} - {i} - max_depth = {j}")

    # ---- 2) Ordena (decrescente) e separa as 10 melhores ----
    # Empate: menor profundidade primeiro (árvore mais simples); depois ordem do grid.
    grid = (pd.DataFrame(resultados)
            .sort_values(["Acc", "max_depth"], ascending=[False, True], kind="stable")
            .reset_index(drop=True))
    grid.insert(0, "#", range(1, len(grid) + 1))
    top = grid.head(N_CONFIGS).copy()
    top.insert(0, "Conf.", range(1, N_CONFIGS + 1))

    print(f"\n=== {N_CONFIGS} melhores configurações (10-fold CV em {MELHOR_BASE}) ===")
    print(top[["Conf.", "Acc", "criterion", "max_depth"]].round(3).to_string(index=False))

    grid.round(4).to_csv(os.path.join(PASTA_SAIDA, "DT_gridsearch_42_resultados.csv"), index=False)
    top[["Conf.", "Acc", "criterion", "max_depth"]].round(4).to_csv(
        os.path.join(PASTA_SAIDA, "DT_top10_configuracoes.csv"), index=False)

    configs = list(zip(top["criterion"], top["max_depth"]))

    # ---- 3) 10 configurações em todas as bases: 10-fold CV e holdout 70/30 ----
    linhas = []
    for pos, nome in enumerate(nomes, 1):
        Xb, yb = carregar_base(os.path.join(PASTA_BASES, nome))
        divisao = train_test_split(Xb, yb, test_size=TAM_TESTE, stratify=yb,
                                   random_state=SEMENTE)
        if nome == MELHOR_BASE:
            accs_cv = top["Acc"].tolist()      # mesmos valores do grid (Figura 3)
        else:
            accs_cv = [acc_10fold(Xb, yb, c, d) for c, d in configs]
        accs_ho = [acc_holdout(divisao, c, d) for c, d in configs]

        base = {"#": pos, "Base": nome, "Atributos": Xb.shape[1]}
        linhas.append({**base, "Treinamento/Teste": "10-fold CV", **dict(zip(COLS_CONF, accs_cv))})
        linhas.append({**base, "Treinamento/Teste": "70/30", **dict(zip(COLS_CONF, accs_ho))})
        print(f"[{pos}/{len(nomes)}] {nome} ({Xb.shape[1]} atrib.) "
              f"10-fold: {np.mean(accs_cv):.4f} | 70/30: {np.mean(accs_ho):.4f}")

    # ---- 4) Tabela final com média e desvio padrão gerais ----
    df = pd.DataFrame(linhas)
    geral = df[COLS_CONF]  # as 24 linhas (10-fold + 70/30), como na Figura 5 do PDF
    resumo = pd.DataFrame([
        {"Base": f"Média ({ROTULO_GERAL})", **geral.mean().to_dict()},
        {"Base": f"Desv. Pad. ({ROTULO_GERAL})", **geral.std(ddof=1).to_dict()},
    ])
    tabela = pd.concat([df, resumo], ignore_index=True)
    tabela["#"] = tabela["#"].astype("Int64")
    tabela["Atributos"] = tabela["Atributos"].astype("Int64")

    saida = os.path.join(PASTA_SAIDA, "DT_tabela_final.csv")
    tabela.round(4).to_csv(saida, index=False)

    print(f"\nConcluído. Arquivos em '{PASTA_SAIDA}/'.\n")
    print(tabela.tail(2).round(4).to_string(index=False))


if __name__ == "__main__":
    main()