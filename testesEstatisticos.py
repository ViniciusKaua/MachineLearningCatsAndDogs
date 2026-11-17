"""
Testes estatísticos (Friedman + pós-teste de Nemenyi) - Prática de Comitês.

Gera, SEM reexecutar nenhum modelo, a visualização dos testes estatísticos para o
relatório: a matriz de p-valores do pós-teste de Nemenyi (mesmo formato da
atividade do k-NN) de cada estrutura de comitê e da comparação final.

Análises geradas:
  Bagging        -> todos os modelos da Figura 1 (12 colunas)
  Random Forest  -> todos os modelos da Figura 2 (8 colunas)
  Stacking       -> todos os modelos da Figura 3 (4 colunas)
  Voting         -> todos os modelos da Figura 4 (4 colunas)
  Comparação     -> os 4 modelos escolhidos (um por comitê)

Entrada : resultados_bagging/   Bagging_tabela_final.csv   (+ Bagging_analise_estatistica.csv)
          resultados_rf/        RF_tabela_final.csv        (+ RF_analise_estatistica.csv)
          resultados_stacking/  Stacking_tabela_final.csv  (+ Stacking_analise_estatistica.csv)
          resultados_voting/    Voting_tabela_final.csv    (+ Voting_analise_estatistica.csv)
          (o *_analise_estatistica.csv só é usado para marcar com ★ o modelo escolhido)
Saída   : resultados_testes/
            <Nome>_testes_estatisticos.txt    (Friedman, ranks, CD e matriz do Nemenyi)
            <Nome>_nemenyi_pvalores.csv       (matriz de p-valores)
            <Nome>_nemenyi_heatmap.png        (figura para o relatório)
          com <Nome> = Bagging, RF, Stacking, Voting e Comparacao.

Os p-valores do Nemenyi seguem o cálculo padrão (o mesmo de scikit-posthocs,
posthoc_nemenyi_friedman): q = |R_i - R_j| / sqrt(k(k+1)/(6N)) * sqrt(2), com
p-valor dado pela distribuição da amplitude studentizada (infinitos graus de liberdade).

Requisitos: pip install pandas numpy scipy matplotlib
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import friedmanchisquare, rankdata, studentized_range

# ----------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------
PASTA_SAIDA = "resultados_testes"

# nome do comitê -> (pasta de resultados, prefixo dos arquivos)
COMITES = {
    "Bagging":       ("resultados_bagging",  "Bagging"),
    "Random Forest": ("resultados_rf",       "RF"),
    "Stacking":      ("resultados_stacking", "Stacking"),
    "Voting":        ("resultados_voting",   "Voting"),
}

COLS_ID = ["#", "Base", "Atributos", "Treinamento/Teste"]

ALFA = 0.05
# Mesmo critério usado nos scripts dos comitês (1 acurácia por base e por modelo):
#   "10-fold CV" | "70/30" | "media" (média dos dois)
DADOS_TESTE = "10-fold CV"

# Cores do heatmap: diagonal, não significativo, e 3 níveis de significância
CORES = ["#e9e9e9", "#ffffff", "#c6dbef", "#6baed6", "#2171b5"]


# ----------------------------------------------------------------------
# LEITURA DOS RESULTADOS JÁ GERADOS
# ----------------------------------------------------------------------
def carregar_comite(nome, pasta, prefixo):
    """Retorna (linhas das bases, colunas dos modelos, modelo escolhido ou None)."""
    cam_tab = os.path.join(pasta, f"{prefixo}_tabela_final.csv")
    cam_est = os.path.join(pasta, f"{prefixo}_analise_estatistica.csv")
    if not os.path.exists(cam_tab):
        raise SystemExit(f"Arquivo não encontrado: '{cam_tab}' (rode antes o script do {nome}).")

    tab = pd.read_csv(cam_tab)
    bases = tab[tab["#"].notna()].copy()  # descarta as linhas de Média e Desv. Pad.
    modelos = [c for c in bases.columns if c not in COLS_ID]

    escolhido = None
    if os.path.exists(cam_est):
        est = pd.read_csv(cam_est)
        marcados = est[est["Escolhido"].astype(str).str.lower() == "true"]
        if len(marcados) == 1:
            escolhido = str(marcados.iloc[0]["Modelo"])
    return bases, modelos, escolhido


def matriz_para_testes(df, colunas):
    """Matriz (bases x modelos) com a acurácia usada nos testes."""
    cv = df[df["Treinamento/Teste"] == "10-fold CV"][colunas].astype(float).reset_index(drop=True)
    ho = df[df["Treinamento/Teste"] == "70/30"][colunas].astype(float).reset_index(drop=True)
    if DADOS_TESTE == "10-fold CV":
        return cv
    if DADOS_TESTE == "70/30":
        return ho
    return (cv + ho) / 2


def matriz_comparacao_final():
    """Matriz (bases x 4 comitês) com o modelo escolhido de cada estrutura."""
    df, colunas = None, []
    for nome, (pasta, prefixo) in COMITES.items():
        bases, _, escolhido = carregar_comite(nome, pasta, prefixo)
        if escolhido is None:
            raise SystemExit(f"Não achei o modelo 'Escolhido' em '{pasta}/{prefixo}"
                             f"_analise_estatistica.csv'.")
        coluna = f"{nome}\n{escolhido}"
        if df is None:  # a 1ª estrutura define a ordem das bases
            df = bases[COLS_ID].copy().reset_index(drop=True)
        mapa = bases.set_index(["Base", "Treinamento/Teste"])[escolhido].astype(float)
        chaves = pd.MultiIndex.from_frame(df[["Base", "Treinamento/Teste"]])
        valores = mapa.reindex(chaves).values
        if np.isnan(valores).any():
            raise SystemExit(f"As bases de '{nome}' não coincidem com as do Bagging.")
        df[coluna] = valores
        colunas.append(coluna)
    return matriz_para_testes(df, colunas)


# ----------------------------------------------------------------------
# TESTES ESTATÍSTICOS
# ----------------------------------------------------------------------
def testes(matriz):
    """Friedman + Nemenyi. Retorna (stat, p, rank_medio, cd, matriz_pvalores)."""
    N, k = matriz.shape
    colunas = list(matriz.columns)
    stat, p = friedmanchisquare(*[matriz[c].values for c in colunas])

    # Rank 1 = melhor (maior acurácia); empates recebem o rank médio
    ranks = np.vstack([rankdata(-linha) for linha in matriz.values])
    rank_medio = pd.Series(ranks.mean(axis=0), index=colunas)

    # Distância crítica de Nemenyi
    q_alfa = studentized_range.ppf(1 - ALFA, k, np.inf) / np.sqrt(2)
    cd = q_alfa * np.sqrt(k * (k + 1) / (6 * N))

    # Matriz de p-valores do pós-teste de Nemenyi
    erro = np.sqrt(k * (k + 1) / (6 * N))
    pvals = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            q = abs(rank_medio.iloc[i] - rank_medio.iloc[j]) / erro * np.sqrt(2)
            pvals[i, j] = pvals[j, i] = float(studentized_range.sf(q, k, np.inf))
    return stat, p, rank_medio, cd, pd.DataFrame(pvals, index=colunas, columns=colunas)


def texto_testes(titulo, matriz, stat, p, rank_medio, cd, pvals):
    N, k = matriz.shape
    houve = p < ALFA
    # rótulos de uma linha só (na comparação final o nome do comitê e o modelo
    # vêm separados por quebra de linha, que é usada apenas na figura)
    uma_linha = lambda s: s.replace("\n", " ")  # noqa: E731
    with pd.option_context("display.width", 110, "display.precision", 10,
                           "display.max_columns", None, "display.max_rows", None):
        ranks_txt = rank_medio.rename(index=uma_linha).sort_values().round(4).to_string()
        pvals_txt = pvals.rename(index=uma_linha, columns=uma_linha).to_string()
    linhas = [
        f"TESTES ESTATÍSTICOS - {titulo.upper()}",
        f"Dados usados: acurácia {DADOS_TESTE} | {N} bases x {k} modelos | alfa = {ALFA}",
        "",
        f"Teste de Friedman: estatística = {stat:.4f} | p-valor = {p:.6f}",
        "Há diferença estatística entre os modelos: " + ("SIM" if houve else "NÃO"),
        f"Distância crítica de Nemenyi (CD) = {cd:.3f}",
        "",
        "Rank médio (1 = melhor):",
        ranks_txt,
        "",
        "Teste Nemenyi (pós-hoc):",
        pvals_txt,
    ]
    if not houve:
        linhas += ["", "Obs.: Friedman não significativo; a matriz do Nemenyi é apenas informativa."]
    return "\n".join(linhas)


# ----------------------------------------------------------------------
# FIGURA
# ----------------------------------------------------------------------
def fmt_p(p):
    return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def desenhar_heatmap(titulo, matriz, stat, p, pvals, escolhido, caminho):
    N, k = matriz.shape
    P = pvals.values
    cat = np.zeros((k, k), int)
    for i in range(k):
        for j in range(k):
            if i == j:
                cat[i, j] = 0
            elif P[i, j] >= ALFA:
                cat[i, j] = 1
            elif P[i, j] >= 0.01:
                cat[i, j] = 2
            elif P[i, j] >= 0.001:
                cat[i, j] = 3
            else:
                cat[i, j] = 4

    quebra = any("\n" in c for c in pvals.index)  # rótulos em 2 linhas (comparação final)
    lado = max(5.0, 0.85 * k + 2.5)
    largura = max(lado + 0.5, 2.6 * k + 2.5) if quebra else lado + 0.5
    fig, ax = plt.subplots(figsize=(largura, lado))
    ax.imshow(cat, cmap=ListedColormap(CORES), norm=BoundaryNorm(np.arange(-0.5, 5.5, 1), 5),
              aspect="auto" if quebra else "equal")  # células mais largas p/ rótulos longos

    tam = max(6, 11 - 0.4 * k)
    for i in range(k):
        for j in range(k):
            texto = "—" if i == j else fmt_p(P[i, j])
            ax.text(j, i, texto, ha="center", va="center", fontsize=tam,
                    color="white" if cat[i, j] == 4 else "black")

    rotulos = [(c + " ★") if c == escolhido else c for c in pvals.index]
    # inclina os rótulos do eixo x quando há muitos modelos ou nomes longos (ex.: "Classificadores (10)")
    girar = k > 6 or (not quebra and max(len(r) for r in rotulos) > 12)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(rotulos, rotation=45 if girar else 0, ha="right" if girar else "center",
                       fontsize=max(7, 10 - 0.2 * k))
    ax.set_yticklabels(rotulos, fontsize=max(7, 10 - 0.2 * k))
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="#999999", linewidth=0.6)
    ax.tick_params(which="both", length=0)

    ax.set_title(f"Pós-teste de Nemenyi (p-valores) - {titulo}", fontsize=13,
                 fontweight="bold", pad=34)
    p_txt = "< 0,001" if p < 0.001 else f"= {p:.4f}".replace(".", ",")
    sub = (f"Friedman: χ² = {stat:.3f}, p {p_txt}  |  {N} bases × {k} modelos  |  "
           f"α = {str(ALFA).replace('.', ',')}  |  acurácia {DADOS_TESTE}")
    if p >= ALFA:
        sub += "\n(Friedman não significativo: matriz apenas informativa)"
    ax.text(0.5, 1.01, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)

    alfa_txt = str(ALFA).replace(".", ",")
    legenda = [
        Patch(facecolor=CORES[1], edgecolor="#999999", label=f"p ≥ {alfa_txt} (sem diferença)"),
        Patch(facecolor=CORES[2], edgecolor="#999999", label=f"{alfa_txt} > p ≥ 0,01"),
        Patch(facecolor=CORES[3], edgecolor="#999999", label="0,01 > p ≥ 0,001"),
        Patch(facecolor=CORES[4], edgecolor="#999999", label="p < 0,001"),
    ]
    if escolhido is not None:
        legenda.append(Line2D([], [], linestyle="none", marker="*", color="black",
                              label="modelo escolhido"))
    # legenda logo abaixo do rótulo mais baixo do eixo x (qualquer tamanho de nome)
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    y_min = min(inv.transform((0, t.get_window_extent().y0))[1] for t in ax.get_xticklabels())
    deslocamento = y_min - 0.02
    ax.legend(handles=legenda, loc="upper center", ncol=len(legenda), fontsize=8,
              frameon=False, bbox_to_anchor=(0.5, deslocamento))
    fig.savefig(caminho, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------
def analisar(titulo, prefixo_arquivo, matriz, escolhido):
    stat, p, rank_medio, cd, pvals = testes(matriz)
    texto = texto_testes(titulo, matriz, stat, p, rank_medio, cd, pvals)

    base = os.path.join(PASTA_SAIDA, prefixo_arquivo)
    pvals.to_csv(f"{base}_nemenyi_pvalores.csv", float_format="%.10f")
    with open(f"{base}_testes_estatisticos.txt", "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    desenhar_heatmap(titulo, matriz, stat, p, pvals, escolhido, f"{base}_nemenyi_heatmap.png")
    print(texto + "\n" + "-" * 70)


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    for nome, (pasta, prefixo) in COMITES.items():
        bases, modelos, escolhido = carregar_comite(nome, pasta, prefixo)
        analisar(nome, prefixo, matriz_para_testes(bases, modelos), escolhido)

    analisar("Comparação final entre os comitês", "Comparacao", matriz_comparacao_final(), None)
    print(f"\nConcluído. Arquivos em '{PASTA_SAIDA}/'.")


if __name__ == "__main__":
    main()