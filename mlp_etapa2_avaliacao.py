"""
Multi Layer Perceptron (MLPClassifier) - Etapa 2: Avaliação nas 12 bases.

Etapas:
  1) Lê as 10 melhores configurações de 'resultados_mlp/MLP_top10_configuracoes.csv'.
  2) Avalia todas as bases de '12bases/' usando 10-fold CV e holdout 70/30
     (com StandardScaler dentro do Pipeline para evitar data leakage).
  3) Gera 'resultados_mlp/MLP_tabela_final.csv' no mesmo formato padrão
     (DT_tabela_final.csv e NB_tabela_final.csv) para preencher a planilha.

Entrada : 12bases/ e resultados_mlp/MLP_top10_configuracoes.csv
Saída   : resultados_mlp/MLP_tabela_final.csv
"""
import ast
import os
import time
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------
PASTA_BASES = "12bases"
PASTA_SAIDA = "resultados_mlp"
ARQ_TOP10 = os.path.join(PASTA_SAIDA, "MLP_top10_configuracoes.csv")
TABELA_REFERENCIA = os.path.join("resultados_selecao", "tabela_final.csv")

N_FOLDS = 10
TAM_TESTE = 0.30
SEMENTE = 42
SEMENTE_MLP = 42
N_JOBS = -1

COL_CLASSE = "classe"
ROTULO_GERAL = "Geral (10-fold + 70/30)"


def listar_bases(pasta):
    nomes = []
    for nome in sorted(os.listdir(pasta)):
        if not nome.lower().endswith(".csv"):
            continue
        colunas = pd.read_csv(os.path.join(pasta, nome), nrows=0).columns
        if COL_CLASSE in colunas:
            nomes.append(nome)
    return nomes


def ordenar_bases(nomes):
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


def carregar_top10(caminho_csv):
    if not os.path.exists(caminho_csv):
        raise SystemExit(
            f"Arquivo '{caminho_csv}' não encontrado. Rode 'mlp_etapa1_gridsearch.py' primeiro."
        )
    df = pd.read_csv(caminho_csv)
    configs = []
    for _, row in df.iterrows():
        hls = ast.literal_eval(str(row["hidden_layer_sizes"]))
        if isinstance(hls, int):
            hls = (hls,)
        configs.append({
            "hidden_layer_sizes": hls,
            "activation": str(row["activation"]),
            "solver": str(row["solver"]),
            "learning_rate_init": float(row["learning_rate_init"]),
            "max_iter": int(row["max_iter"]),
        })
    return configs


def criar_modelo(conf):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(**conf, random_state=SEMENTE_MLP),
    )


def acc_10fold(X, y, conf):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
    return cross_val_score(
        criar_modelo(conf), X, y, cv=cv, scoring="accuracy", n_jobs=N_JOBS
    ).mean()


def acc_holdout(divisao, conf):
    X_tr, X_te, y_tr, y_te = divisao
    modelo = criar_modelo(conf).fit(X_tr, y_tr)
    return accuracy_score(y_te, modelo.predict(X_te))


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    configs = carregar_top10(ARQ_TOP10)
    cols_conf = [f"Conf. {i}" for i in range(1, len(configs) + 1)]

    nomes = listar_bases(PASTA_BASES)
    if not nomes:
        raise SystemExit(f"Nenhuma base encontrada em '{PASTA_BASES}'.")
    nomes = ordenar_bases(nomes)

    print(f"Avaliando {len(nomes)} bases x {len(configs)} configurações (10-fold CV e 70/30)...\n")
    t0 = time.time()
    linhas = []

    for pos, nome in enumerate(nomes, 1):
        Xb, yb = carregar_base(os.path.join(PASTA_BASES, nome))
        divisao = train_test_split(
            Xb, yb, test_size=TAM_TESTE, stratify=yb, random_state=SEMENTE
        )

        accs_cv = [acc_10fold(Xb, yb, c) for c in configs]
        accs_ho = [acc_holdout(divisao, c) for c in configs]

        base_info = {"#": pos, "Base": nome, "Atributos": Xb.shape[1]}
        linhas.append({**base_info, "Treinamento/Teste": "10-fold CV", **dict(zip(cols_conf, accs_cv))})
        linhas.append({**base_info, "Treinamento/Teste": "70/30", **dict(zip(cols_conf, accs_ho))})

        print(
            f"[{pos:02d}/{len(nomes)}] {nome} ({Xb.shape[1]} atrib.) | "
            f"10-fold: {np.mean(accs_cv):.4f} | 70/30: {np.mean(accs_ho):.4f}"
        )

    df = pd.DataFrame(linhas)
    geral = df[cols_conf]
    resumo = pd.DataFrame([
        {"Base": f"Média ({ROTULO_GERAL})", **geral.mean().to_dict()},
        {"Base": f"Desv. Pad. ({ROTULO_GERAL})", **geral.std(ddof=1).to_dict()},
    ])
    tabela = pd.concat([df, resumo], ignore_index=True)
    tabela["#"] = tabela["#"].astype("Int64")
    tabela["Atributos"] = tabela["Atributos"].astype("Int64")

    saida = os.path.join(PASTA_SAIDA, "MLP_tabela_final.csv")
    tabela.round(4).to_csv(saida, index=False)

    print(f"\nConcluído em {time.time() - t0:.2f}s. Arquivo salvo em '{saida}'.\n")
    print(tabela.tail(2).round(4).to_string(index=False))


if __name__ == "__main__":
    main()