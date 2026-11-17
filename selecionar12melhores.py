"""
Seleção das 12 melhores bases com k-NN.

Etapas (conforme o PDF da Aula 13):
  1) k-NN (metric='euclidean', k = 1..10) com 10-fold CV em TODAS as bases.
  2) Critério: as N_MELHORES bases com maior acurácia média (média dos k=1..10
     no 10-fold CV). Empate: fica a base com menos atributos.
  3) As bases escolhidas são COPIADAS para a pasta resultados_selecao/.
  4) Holdout 70/30 nas bases escolhidas.
  5) Saída única: tabela_final.csv (10-fold CV e 70/30 por base, com a
     média geral e o desvio padrão geral das 24 linhas).

Entrada : todasasbases/ (hog_pura/ e pca/), cada CSV com a coluna 'classe' por último
Saída   : resultados_selecao/
            tabela_final.csv                  (resultado final)
            <as 12 bases selecionadas>.csv    (cópias)
            resultados_cv_todas_as_bases.csv  (cache interno que permite retomar; pode apagar no fim)

Requisitos: pip install scikit-learn pandas numpy
"""
import os
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------
PASTA_BASES = "todasasbases"
PASTA_SAIDA = "resultados_selecao"

K_VALORES = list(range(1, 11))   # k = 1..10
METRICA = "euclidean"
N_MELHORES = 12
N_FOLDS = 10
TAM_TESTE = 0.30                 # holdout 70/30
SEMENTE = 42                     # reprodutibilidade
N_JOBS = -1                      # -1 = usa todos os núcleos

COL_CLASSE = "classe"
COLS_K = [f"{k}k" for k in K_VALORES]


# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ----------------------------------------------------------------------
def listar_bases(pasta):
    """Todos os CSVs de bases (hog_pura/ e pca/), ignorando ordem_imagens.csv."""
    caminhos = []
    for raiz, _, arquivos in os.walk(pasta):
        for nome in arquivos:
            if nome.lower().endswith(".csv") and nome != "ordem_imagens.csv":
                caminhos.append(os.path.join(raiz, nome))
    return sorted(caminhos, key=lambda p: os.path.basename(p))


def carregar_base(caminho):
    """Retorna X (atributos) e y (classe)."""
    df = pd.read_csv(caminho)
    y = df[COL_CLASSE].values
    X = df.drop(columns=[COL_CLASSE]).values
    return X, y


def knn(k):
    return KNeighborsClassifier(n_neighbors=k, metric=METRICA)


def acuracias_10fold(X, y):
    """Acurácia média do 10-fold CV (estratificado) para cada k."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
    return [
        cross_val_score(knn(k), X, y, cv=cv, scoring="accuracy", n_jobs=N_JOBS).mean()
        for k in K_VALORES
    ]


def acuracias_holdout(X, y):
    """Acurácia no holdout 70/30 (estratificado) para cada k."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TAM_TESTE, stratify=y, random_state=SEMENTE
    )
    return [knn(k).fit(X_tr, y_tr).score(X_te, y_te) for k in K_VALORES]


def tipo_base(nome):
    if "PCA-75" in nome:
        return "PCA 75%"
    if "PCA-90" in nome:
        return "PCA 90%"
    return "HOG"


# ----------------------------------------------------------------------
# ETAPA 1: 10-fold CV em todas as bases (com retomada)
# ----------------------------------------------------------------------
def etapa_cv_todas(caminhos, arquivo_parcial):
    if os.path.exists(arquivo_parcial):
        df_ok = pd.read_csv(arquivo_parcial)
        feitas = set(df_ok["base"])
        linhas = df_ok.to_dict("records")
        print(f"Retomando: {len(feitas)} bases já avaliadas.")
    else:
        feitas, linhas = set(), []

    for i, caminho in enumerate(caminhos, 1):
        nome = os.path.basename(caminho)
        if nome in feitas:
            continue
        t0 = time.time()
        X, y = carregar_base(caminho)
        accs = acuracias_10fold(X, y)
        linha = {"base": nome, "tipo": tipo_base(nome), "atributos": X.shape[1]}
        linha.update(dict(zip(COLS_K, accs)))
        linha["media"] = float(np.mean(accs))
        linhas.append(linha)
        pd.DataFrame(linhas).to_csv(arquivo_parcial, index=False)  # salva a cada base
        print(f"[{i}/{len(caminhos)}] {nome} ({X.shape[1]} atrib.) "
              f"média = {linha['media']:.4f}  ({time.time() - t0:.1f}s)")

    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------
# ETAPA 3/4: holdout nas melhores e tabela final
# ----------------------------------------------------------------------
def montar_tabela_final(melhores, caminho_por_nome):
    linhas = []
    for pos, (_, r) in enumerate(melhores.iterrows(), 1):
        X, y = carregar_base(caminho_por_nome[r["base"]])
        accs_ho = acuracias_holdout(X, y)
        base = {"#": pos, "Base": r["base"], "Atributos": int(r["atributos"])}
        linhas.append({**base, "Treinamento/Teste": "10-fold CV",
                       **{c: r[c] for c in COLS_K}})
        linhas.append({**base, "Treinamento/Teste": "70/30",
                       **dict(zip(COLS_K, accs_ho))})
        print(f"holdout {r['base']}: média = {np.mean(accs_ho):.4f}")

    df = pd.DataFrame(linhas)
    geral = df[COLS_K]  # as 24 linhas (10-fold + 70/30), como na Figura 5 do PDF
    resumo = pd.DataFrame([
        {"Base": "Média (Geral (10-fold + 70/30))", **geral.mean().to_dict()},
        {"Base": "Desv. Pad. (Geral (10-fold + 70/30))", **geral.std(ddof=1).to_dict()},
    ])
    tabela = pd.concat([df, resumo], ignore_index=True)
    tabela["#"] = tabela["#"].astype("Int64")            # inteiros (sem '1.0')
    tabela["Atributos"] = tabela["Atributos"].astype("Int64")
    return tabela


# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------
def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    caminhos = listar_bases(PASTA_BASES)
    if not caminhos:
        raise SystemExit(f"Nenhuma base encontrada em '{PASTA_BASES}'.")
    por_nome = {os.path.basename(c): c for c in caminhos}
    print(f"{len(caminhos)} bases encontradas.\n")

    # 1) 10-fold CV em todas
    parcial = os.path.join(PASTA_SAIDA, "resultados_cv_todas_as_bases.csv")
    todas = etapa_cv_todas(caminhos, parcial)

    # 2) Ranking e seleção (maior média; empate -> menos atributos)
    todas = todas.sort_values(["media", "atributos"], ascending=[False, True]).reset_index(drop=True)
    todas.insert(0, "ranking", range(1, len(todas) + 1))
    melhores = todas[todas["ranking"] <= N_MELHORES].copy()

    print(f"\n=== {N_MELHORES} melhores bases (10-fold CV, média de k=1..10) ===")
    print(melhores[["ranking", "base", "atributos", "media"]].to_string(index=False))

    # 3) Copia as bases selecionadas para resultados_selecao/
    for nome in melhores["base"]:
        shutil.copy2(por_nome[nome], os.path.join(PASTA_SAIDA, nome))
    print(f"\n{len(melhores)} bases copiadas para '{PASTA_SAIDA}/'.")

    # 4) Holdout 70/30 + tabela final (um único CSV)
    print("\nHoldout 70/30 nas bases selecionadas...")
    tabela = montar_tabela_final(melhores, por_nome)
    saida = os.path.join(PASTA_SAIDA, "tabela_final.csv")
    tabela.round(4).to_csv(saida, index=False)

    print(f"\nConcluído. Resultado em '{saida}'.")
    print("\n", tabela.tail(2).round(4).to_string(index=False))


if __name__ == "__main__":
    main()