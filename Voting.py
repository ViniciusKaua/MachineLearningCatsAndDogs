"""
Voting (VotingClassifier) heterogêneo - Prática de Comitês de Classificadores.

Etapas (conforme o PDF, Figura 4):
  1) Monta comitês com 5, 10, 15 e 20 classificadores base, usando 4 algoritmos:
     k-NN, DT, NB e MLP. Cada algoritmo tem uma lista de configurações em ordem
     de qualidade: as melhores das atividades anteriores (planilhas K-NN, DT, NB
     e MLP), ranqueadas pela linha "Média =>" (média geral das 24 linhas).
     As listas são combinadas em rodízio (k-NN, DT, NB, MLP, k-NN, DT, ...),
     formando um "pool" de 20 classificadores; o comitê de tamanho N usa os N
     primeiros do pool (comitês aninhados). Quando um algoritmo esgota suas
     configurações (NB tem 2; MLP tem 4), o rodízio segue com os demais.
     MLP: como o PDF pede para evitar configurações complexas (demoradas),
     só entram as configurações do Top 10 com max_iter <= 1000 (todas com sgd).
       5  -> k-NN x2, DT x1, NB x1, MLP x1
       10 -> k-NN x3, DT x3, NB x2, MLP x2
       15 -> k-NN x5, DT x4, NB x2, MLP x4
       20 -> k-NN x7, DT x7, NB x2, MLP x4
     VotingClassifier com parâmetros default (voting="hard": maioria dos votos).
  2) Cada comitê é avaliado nas 12 bases (pasta 12bases) com 10-fold CV e
     holdout 70/30 -> tabela da Figura 4 (com média e desvio padrão gerais,
     calculados sobre as 24 linhas: 10-fold + 70/30).
  3) Análise estatística entre os 4 comitês:
       - Teste de Friedman (há diferença estatística?)
       - Havendo diferença: melhor modelo = melhor rank médio (pós-teste de
         Nemenyi informa quais pares diferem).
       - Não havendo: melhor modelo = menor complexidade, medida pelo nº de
         classificadores base; entre os de mesma complexidade, maior acurácia média.

Entrada : 12bases/ (CSVs com a coluna 'classe')
Saída   : resultados_voting/
            Voting_tabela_final.csv          (Figura 4 preenchida)
            Voting_composicao.csv            (quais classificadores formam cada comitê)
            Voting_analise_estatistica.csv   (rank médio, acurácia média, escolhido)
            Voting_analise_estatistica.txt   (Friedman, Nemenyi e decisão)
            Voting_parcial.csv               (cache interno p/ retomar; pode apagar no fim)

Requisitos: pip install scikit-learn pandas numpy scipy
"""
import os
import warnings

# Silencia ConvergenceWarning do MLP também nos processos paralelos (joblib/loky)
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import time

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------
PASTA_BASES = "12bases"
PASTA_SAIDA = "resultados_voting"

# Opcional: se existir, a ordem das bases na tabela segue a do k-NN.
TABELA_KNN = os.path.join("resultados_selecao", "tabela_final.csv")

TAMANHOS = [5, 10, 15, 20]  # nº de classificadores base de cada comitê
N_FOLDS = 10
TAM_TESTE = 0.30            # holdout 70/30
SEMENTE = 42                # folds, holdout e modelos (reprodutível)
N_JOBS = -1
VOTO = "hard"             # "hard" (maioria; default do sklearn) | "soft" (média das probabilidades)

# Análise estatística
ALFA = 0.05
# Quais acurácias entram nos testes (1 valor por base e por modelo):
#   "10-fold CV" | "70/30" | "media" (média dos dois)
DADOS_TESTE = "10-fold CV"

COL_CLASSE = "classe"
ROTULO_GERAL = "Geral (10-fold + 70/30)"

# Padronização (StandardScaler) dentro de cada classificador base (sem vazamento).
# MLP: True, pois a melhor configuração foi encontrada com StandardScaler no Pipeline.
# k-NN: ajuste conforme foi feito na atividade do k-NN.
ESCALAR = {"k-NN": False, "DT": False, "NB": False, "MLP": True}

# Ordem do rodízio entre os algoritmos
ORDEM_ALGORITMOS = ["k-NN", "DT", "NB", "MLP"]


def _knn(k):
    return lambda: KNeighborsClassifier(n_neighbors=k, metric="euclidean")


def _dt(criterio, prof):
    return lambda: DecisionTreeClassifier(criterion=criterio, max_depth=prof,
                                          random_state=SEMENTE)


def _mlp(hls, ativ, solver, lr, it):
    return lambda: MLPClassifier(hidden_layer_sizes=hls, activation=ativ, solver=solver,
                                 learning_rate_init=lr, max_iter=it, random_state=SEMENTE)


# Cada entrada: (descrição, fábrica, origem na planilha). A 1ª de cada lista é a MELHOR.
# Ranking pela linha "Média =>" das planilhas (média das 24 linhas: 10-fold + 70/30).
CONFIGS = {
    "k-NN": [
        ("k=4, euclidean",  _knn(4),  "K-NN, coluna k4 (média 0,6714) - MELHOR"),
        ("k=8, euclidean",  _knn(8),  "K-NN, coluna k8 (média 0,6704)"),
        ("k=6, euclidean",  _knn(6),  "K-NN, coluna k6 (média 0,6689)"),
        ("k=2, euclidean",  _knn(2),  "K-NN, coluna k2 (média 0,6666)"),
        ("k=10, euclidean", _knn(10), "K-NN, coluna k10 (média 0,6666)"),
        ("k=3, euclidean",  _knn(3),  "K-NN, coluna k3 (média 0,6515)"),
        ("k=1, euclidean",  _knn(1),  "K-NN, coluna k1 (média 0,6487)"),
    ],
    "DT": [
        ("gini, max_depth=9",     _dt("gini", 9),      "DT, Conf. 6 (média 0,6516) - MELHOR"),
        ("gini, max_depth=10",    _dt("gini", 10),     "DT, Conf. 1 (média 0,6470)"),
        ("entropy, max_depth=12", _dt("entropy", 12),  "DT, Conf. 7 (média 0,6464); a Conf. 8 (log_loss) é idêntica"),
        ("entropy, max_depth=15", _dt("entropy", 15),  "DT, Conf. 9 (média 0,6445); a Conf. 10 (log_loss) é idêntica"),
        ("gini, max_depth=12",    _dt("gini", 12),     "DT, Conf. 3 (média 0,6414)"),
        ("gini, max_depth=15",    _dt("gini", 15),     "DT, Conf. 2 (média 0,6403)"),
        ("gini, max_depth=14",    _dt("gini", 14),     "DT, Conf. 4 (média 0,6398)"),
    ],
    "NB": [
        ("GaussianNB()", lambda: GaussianNB(), "NB, GaussianNB (média 0,7324) - MELHOR"),
        # MultinomialNB não aceita valores negativos (PCA): MinMaxScaler dentro do Pipeline.
        # O ComplementNB teve resultados idênticos aos do MultinomialNB em todas as bases
        # e foi descartado por ser redundante.
        ("MultinomialNB() + MinMaxScaler",
         lambda: make_pipeline(MinMaxScaler(clip=True), MultinomialNB()),
         "NB, MultinomialNB (média 0,7100)"),
    ],
    # Evitando configurações complexas do MLP (PDF): só max_iter <= 1000, todas com sgd.
    # Ficam de fora as Conf. 4, 6, 1 e 7 (adam, 1500-2000 iterações).
    "MLP": [
        ("(64,32), relu, sgd, lr=0.001, it=500",   _mlp((64, 32), "relu", "sgd", 0.001, 500),
         "MLP, Conf. 3 (média 0,7309) - MELHOR"),
        ("(64,32), relu, sgd, lr=0.001, it=1000",  _mlp((64, 32), "relu", "sgd", 0.001, 1000),
         "MLP, Conf. 5 (média 0,7282)"),
        ("(16,), logistic, sgd, lr=0.01, it=1000", _mlp((16,), "logistic", "sgd", 0.01, 1000),
         "MLP, Conf. 2 (média 0,7279)"),
        ("(64,), relu, sgd, lr=0.01, it=1000",     _mlp((64,), "relu", "sgd", 0.01, 1000),
         "MLP, Conf. 9 (média 0,7248)"),
    ],
}
SIGLA = {"k-NN": "knn", "DT": "dt", "NB": "nb", "MLP": "mlp"}


def montar_pool(n_total):
    """Rodízio entre os algoritmos; pula quem já esgotou suas configurações."""
    pool, rodada = [], 0
    while len(pool) < n_total:
        entrou = False
        for alg in ORDEM_ALGORITMOS:
            if rodada < len(CONFIGS[alg]) and len(pool) < n_total:
                pool.append((alg, rodada))
                entrou = True
        if not entrou:
            raise SystemExit("Configurações insuficientes para formar o maior comitê.")
        rodada += 1
    return pool


POOL = montar_pool(max(TAMANHOS))
COLUNAS = [f"Classificadores ({n})" for n in TAMANHOS]


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


def criar_base(algoritmo, variante):
    est = CONFIGS[algoritmo][variante][1]()
    return make_pipeline(StandardScaler(), est) if ESCALAR[algoritmo] else est


def voting(n_classificadores):
    estimadores = [(f"{SIGLA[alg]}_{v + 1}", criar_base(alg, v))
                   for alg, v in POOL[:n_classificadores]]
    return VotingClassifier(estimators=estimadores, voting=VOTO)


def descricao(algoritmo, variante):
    texto = f"{algoritmo}: {CONFIGS[algoritmo][variante][0]}"
    return texto + (" + StandardScaler" if ESCALAR[algoritmo] else "")


def tabela_composicao():
    linhas = []
    for n in TAMANHOS:
        for pos, (alg, v) in enumerate(POOL[:n], 1):
            linhas.append({"Comitê": f"Classificadores ({n})", "#": pos,
                           "Algoritmo": alg, "Variante": v + 1,
                           "Configuração": descricao(alg, v),
                           "Origem": CONFIGS[alg][v][2]})
    return pd.DataFrame(linhas)


def acc_10fold(X, y, n):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
    return cross_val_score(voting(n), X, y, cv=cv,
                           scoring="accuracy", n_jobs=N_JOBS).mean()


def acc_holdout(divisao, n):
    X_tr, X_te, y_tr, y_te = divisao
    return voting(n).fit(X_tr, y_tr).score(X_te, y_te)


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
        "n_classificadores": TAMANHOS,
        "Acc_media": matriz.mean().values,
        "Rank_medio": rank_medio,
        "Complexidade": TAMANHOS,
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
                    melhor, pior = (COLUNAS[i], COLUNAS[j]) if rank_medio[i] < rank_medio[j] \
                        else (COLUNAS[j], COLUNAS[i])
                    pares.append(f"  {melhor} > {pior} (dif. de rank = {d:.2f})")

    linhas = [
        "ANÁLISE ESTATÍSTICA - STACKING",
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
        f"MODELO ESCOLHIDO: Voting {escolhido}",
        "",
        "Obs.: a complexidade considerada é o nº de classificadores base do comitê.",
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
    print(f"{len(nomes)} bases em '{PASTA_BASES}' | {len(TAMANHOS)} comitês de Voting\n")

    comp = tabela_composicao()
    comp.to_csv(os.path.join(PASTA_SAIDA, "Voting_composicao.csv"), index=False)
    print("Composição dos comitês (salva em Voting_composicao.csv):")
    print(comp.groupby("Comitê", sort=False)["Algoritmo"]
          .value_counts().unstack(fill_value=0)[ORDEM_ALGORITMOS].to_string())
    print()

    # ---- Avaliação (com retomada) ----
    parcial = os.path.join(PASTA_SAIDA, "Voting_parcial.csv")
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
        accs_cv = [acc_10fold(X, y, n) for n in TAMANHOS]
        accs_ho = [acc_holdout(divisao, n) for n in TAMANHOS]

        base = {"#": pos, "Base": nome, "Atributos": X.shape[1]}
        linhas.append({**base, "Treinamento/Teste": "10-fold CV", **dict(zip(COLUNAS, accs_cv))})
        linhas.append({**base, "Treinamento/Teste": "70/30", **dict(zip(COLUNAS, accs_ho))})
        pd.DataFrame(linhas).to_csv(parcial, index=False)  # salva a cada base
        print(f"[{pos}/{len(nomes)}] {nome} ({X.shape[1]} atrib.) "
              f"10-fold: {np.mean(accs_cv):.4f} | 70/30: {np.mean(accs_ho):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- Tabela final (Figura 4) com média e desvio padrão gerais ----
    df = pd.DataFrame(linhas)
    geral = df[COLUNAS]  # as 24 linhas (10-fold + 70/30)
    resumo = pd.DataFrame([
        {"Base": f"Média ({ROTULO_GERAL})", **geral.mean().to_dict()},
        {"Base": f"Desv. Pad. ({ROTULO_GERAL})", **geral.std(ddof=1).to_dict()},
    ])
    tabela = pd.concat([df, resumo], ignore_index=True)
    tabela["#"] = tabela["#"].astype("Int64")
    tabela["Atributos"] = tabela["Atributos"].astype("Int64")
    tabela.round(4).to_csv(os.path.join(PASTA_SAIDA, "Voting_tabela_final.csv"), index=False)

    # ---- Análise estatística ----
    tab_est, texto = analise_estatistica(matriz_para_testes(df))
    tab_est.round(4).to_csv(os.path.join(PASTA_SAIDA, "Voting_analise_estatistica.csv"),
                            index=False)
    with open(os.path.join(PASTA_SAIDA, "Voting_analise_estatistica.txt"), "w",
              encoding="utf-8") as f:
        f.write(texto + "\n")

    print(f"\nConcluído. Arquivos em '{PASTA_SAIDA}/'.\n")
    print(tabela.tail(2).round(4).to_string(index=False))
    print("\n" + texto)


if __name__ == "__main__":
    main()