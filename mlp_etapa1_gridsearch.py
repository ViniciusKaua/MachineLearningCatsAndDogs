"""
Multi Layer Perceptron (MLPClassifier) - Etapa 1: GridSearch e Top 10.

Etapas (conforme Aula_Pratica_MLP-2.pdf):
  1) Identifica automaticamente na pasta '12bases' a base com MENOR quantidade
     de atributos e a base com MAIOR quantidade de atributos.
  2) Executa GridSearch (480 configurações) com Stratified 10-fold CV nas duas bases,
     utilizando StandardScaler() dentro de Pipeline (sem vazamento de dados).
  3) Seleciona as 15 melhores configurações de cada base (Top 15).
  4) Monta as 10 melhores configurações finais priorizando a INTERSEÇÃO entre as
     duas listas Top 15 e completando com as melhores acurácias de cada base.

Entrada : 12bases/ (CSVs com a coluna 'classe')
Saída   : resultados_mlp/
            MLP_grid_menor_<base>.csv
            MLP_grid_maior_<base>.csv
            MLP_top15_menor_<base>.csv
            MLP_top15_maior_<base>.csv
            MLP_top10_configuracoes.csv
"""
import ast
import os
import time
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------
PASTA_BASES = "12bases"
PASTA_SAIDA = "resultados_mlp"
COL_CLASSE = "classe"

# Grade da Figura 1 do PDF (5 x 4 x 2 x 3 x 4 = 480 configurações)
# Caso prefira a sua grade reduzida, basta substituir par_hls por [(16,), (32,), (64,), (32, 16), (64, 32)]
GRID = {
    "hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)],
    "activation": ["identity", "logistic", "relu", "tanh"],
    "solver": ["adam", "sgd"],
    "learning_rate_init": [0.0001, 0.001, 0.01],
    "max_iter": [500, 1000, 1500, 2000],
}
PARAMS = list(GRID.keys())

N_FOLDS = 10
SEMENTE = 42
SEMENTE_MLP = 42
N_JOBS = -1
TOP_K_BASE = 15
N_CONFIGS_FINAIS = 10


def fmt_tempo(seg):
    seg = int(seg)
    return f"{seg // 3600:02d}:{(seg % 3600) // 60:02d}:{seg % 60:02d}"


def listar_bases_com_atributos(pasta):
    """Retorna lista de tuplas (nome_arquivo, n_atributos) ordenada por n_atributos."""
    bases = []
    for nome in sorted(os.listdir(pasta)):
        if not nome.lower().endswith(".csv"):
            continue
        cols = pd.read_csv(os.path.join(pasta, nome), nrows=0).columns
        if COL_CLASSE in cols:
            n_atrib = len(cols) - 1
            bases.append((nome, n_atrib))
    bases.sort(key=lambda x: (x[1], x[0]))
    return bases


def carregar_base(caminho):
    df = pd.read_csv(caminho)
    y = df[COL_CLASSE].values
    X = df.drop(columns=[COL_CLASSE]).values
    return X, y


def rodar_grid_base(nome_base, rotulo_tamanho):
    caminho = os.path.join(PASTA_BASES, nome_base)
    X, y = carregar_base(caminho)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
    configs = list(ParameterGrid(GRID))
    total = len(configs)

    print("\n" + "=" * 72)
    print(f"[{rotulo_tamanho.upper()}] Base: {nome_base} | Atributos: {X.shape[1]} | Amostras: {X.shape[0]}")
    print(f"Executando {total} configurações ({N_FOLDS}-fold CV)...")
    print("=" * 72)

    resultados = []
    melhor_acc = 0.0
    t0 = time.time()

    for idx, conf in enumerate(configs, start=1):
        modelo = make_pipeline(
            StandardScaler(),
            MLPClassifier(**conf, random_state=SEMENTE_MLP),
        )
        acc = cross_val_score(
            modelo, X, y, scoring="accuracy", cv=cv, n_jobs=N_JOBS
        ).mean()

        melhor_acc = max(melhor_acc, acc)
        decorrido = time.time() - t0
        eta = (decorrido / idx) * (total - idx)
        hls_str = str(conf["hidden_layer_sizes"])

        print(
            f"  [{idx:03d}/{total}] Acc={acc:.4f} (best={melhor_acc:.4f}) | "
            f"hls={hls_str:<8} act={conf['activation']:<8} sol={conf['solver']:<4} "
            f"lr={conf['learning_rate_init']:<6} it={conf['max_iter']:<4} | ETA: {fmt_tempo(eta)}",
            flush=True,
        )

        resultados.append({
            "hidden_layer_sizes": hls_str,
            "activation": conf["activation"],
            "solver": conf["solver"],
            "learning_rate_init": conf["learning_rate_init"],
            "max_iter": conf["max_iter"],
            "Acc": acc,
        })

    tempo_total = time.time() - t0
    df_grid = (
        pd.DataFrame(resultados)
        .sort_values(["Acc", "max_iter"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    df_grid.insert(0, "#", range(1, len(df_grid) + 1))

    df_top15 = df_grid.head(TOP_K_BASE).copy()

    # Persistência
    sufixo = nome_base.replace(".csv", "")
    df_grid.round(4).to_csv(
        os.path.join(PASTA_SAIDA, f"MLP_grid_{rotulo_tamanho}_{sufixo}.csv"), index=False
    )
    df_top15.round(4).to_csv(
        os.path.join(PASTA_SAIDA, f"MLP_top15_{rotulo_tamanho}_{sufixo}.csv"), index=False
    )

    print(f"\n>>> Concluído em {tempo_total:.2f}s ({fmt_tempo(tempo_total)}). Top 5 de {nome_base}:")
    print(df_top15.head(5).round(4).to_string(index=False))

    return df_top15, tempo_total


def selecionar_top10_com_intersecao(top15_menor, nome_menor, top15_maior, nome_maior):
    """
    Regra do PDF (Figuras 2 e 3):
      1. Começa pelas configurações presentes na interseção das duas listas Top 15.
      2. Completa até 10 configurações intercalando as melhores restantes de cada base.
    """
    def chave(row):
        return (
            row["hidden_layer_sizes"],
            row["activation"],
            row["solver"],
            float(row["learning_rate_init"]),
            int(row["max_iter"]),
        )

    mapa_menor = {chave(r): r for _, r in top15_menor.iterrows()}
    mapa_maior = {chave(r): r for _, r in top15_maior.iterrows()}

    selecionadas = []
    chaves_usadas = set()

    # 1) Interseção entre Top 15 da menor e Top 15 da maior
    intersecao = [k for k in mapa_menor if k in mapa_maior]
    intersecao.sort(
        key=lambda k: (mapa_menor[k]["Acc"] + mapa_maior[k]["Acc"]) / 2.0,
        reverse=True,
    )

    for k in intersecao:
        if len(selecionadas) >= N_CONFIGS_FINAIS:
            break
        r_men = mapa_menor[k]
        r_mai = mapa_maior[k]
        selecionadas.append({
            "Acc": max(r_men["Acc"], r_mai["Acc"]),
            "Acc_Menor": r_men["Acc"],
            "Acc_Maior": r_mai["Acc"],
            "Origem": "Interseção",
            "hidden_layer_sizes": r_men["hidden_layer_sizes"],
            "activation": r_men["activation"],
            "solver": r_men["solver"],
            "learning_rate_init": r_men["learning_rate_init"],
            "max_iter": int(r_men["max_iter"]),
        })
        chaves_usadas.add(k)

    # 2) Completa com as melhores restantes de cada base (alternando menor e maior)
    rest_menor = [r for _, r in top15_menor.iterrows() if chave(r) not in chaves_usadas]
    rest_maior = [r for _, r in top15_maior.iterrows() if chave(r) not in chaves_usadas]

    idx_men, idx_mai = 0, 0
    while len(selecionadas) < N_CONFIGS_FINAIS and (idx_men < len(rest_menor) or idx_mai < len(rest_maior)):
        if idx_men < len(rest_menor):
            r = rest_menor[idx_men]
            idx_men += 1
            k = chave(r)
            if k not in chaves_usadas:
                selecionadas.append({
                    "Acc": r["Acc"],
                    "Acc_Menor": r["Acc"],
                    "Acc_Maior": np.nan,
                    "Origem": f"Menor ({nome_menor})",
                    "hidden_layer_sizes": r["hidden_layer_sizes"],
                    "activation": r["activation"],
                    "solver": r["solver"],
                    "learning_rate_init": r["learning_rate_init"],
                    "max_iter": int(r["max_iter"]),
                })
                chaves_usadas.add(k)

        if len(selecionadas) >= N_CONFIGS_FINAIS:
            break

        if idx_mai < len(rest_maior):
            r = rest_maior[idx_mai]
            idx_mai += 1
            k = chave(r)
            if k not in chaves_usadas:
                selecionadas.append({
                    "Acc": r["Acc"],
                    "Acc_Menor": np.nan,
                    "Acc_Maior": r["Acc"],
                    "Origem": f"Maior ({nome_maior})",
                    "hidden_layer_sizes": r["hidden_layer_sizes"],
                    "activation": r["activation"],
                    "solver": r["solver"],
                    "learning_rate_init": r["learning_rate_init"],
                    "max_iter": int(r["max_iter"]),
                })
                chaves_usadas.add(k)

    df_top10 = pd.DataFrame(selecionadas)
    df_top10.insert(0, "Conf.", range(1, len(df_top10) + 1))
    return df_top10


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    bases = listar_bases_com_atributos(PASTA_BASES)
    if len(bases) < 2:
        raise SystemExit(f"São necessárias pelo menos 2 bases CSV em '{PASTA_BASES}'.")

    nome_menor, atrib_menor = bases[0]
    nome_maior, atrib_maior = bases[-1]

    print(f"Base com MENOR nº de atributos: {nome_menor} ({atrib_menor} atributos)")
    print(f"Base com MAIOR nº de atributos: {nome_maior} ({atrib_maior} atributos)")

    top15_menor, t_men = rodar_grid_base(nome_menor, "menor")
    top15_maior, t_mai = rodar_grid_base(nome_maior, "maior")

    df_top10 = selecionar_top10_com_intersecao(
        top15_menor, nome_menor, top15_maior, nome_maior
    )

    saida_top10 = os.path.join(PASTA_SAIDA, "MLP_top10_configuracoes.csv")
    df_top10.round(4).to_csv(saida_top10, index=False)

    print("\n" + "=" * 72)
    print(f"=== 10 MELHORES CONFIGURAÇÕES SELECIONADAS (Tempo total: {fmt_tempo(t_men + t_mai)}) ===")
    print("=" * 72)
    print(df_top10.round(4).to_string(index=False))
    print(f"\nArquivo salvo em: {saida_top10}")


if __name__ == "__main__":
    main()