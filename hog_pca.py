"""
Geração de bases HOG + PCA (grid search sobre resolução x pixels por célula).
Segue a estrutura do notebook HOG_FeatureExtraction.ipynb:
  imread -> resize (anti_aliasing=True) -> rgb2gray -> hog -> classe dog/cat

Entrada : pasta "Imagens" (todas as imagens juntas)
Saída   : todasasbases/
            hog_pura/  -> HOG_<res>_<cel>x<cel>.csv
            pca/       -> HOG_<res>_<cel>x<cel>_PCA-75.csv e _PCA-90.csv
            ordem_imagens.csv -> arquivo e classe de cada linha (mesma ordem dos CSVs)

Em todos os CSVs a coluna 'classe' é a última.

Requisitos: pip install scikit-image scikit-learn pandas numpy
"""
import os
import time

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import hog
from sklearn.decomposition import PCA


PASTA_IMAGENS = "Imagens"
PASTA_SAIDA = "todasasbases"
SUBPASTA_HOG = "hog_pura"
SUBPASTA_PCA = "pca"

# Grid search
RESOLUCOES = [256, 128, 64]
PIXELS_POR_CELULA = [8, 12, 16, 20, 24, 32]

# Parâmetros fixos do HOG (iguais ao notebook)
ORIENTACOES = 9
CELULAS_POR_BLOCO = (2, 2)

# Variância explicada mantida pelo PCA
PCA_VARIANCIAS = {"PCA-75": 0.75, "PCA-90": 0.90}

EXTENSOES = (".jpg", ".jpeg", ".png")


def listar_imagens(pasta):
    """Lista os arquivos de imagem da pasta, em ordem alfabética (reprodutível)."""
    return sorted(f for f in os.listdir(pasta) if f.lower().endswith(EXTENSOES))


def definir_classe(nome_arquivo):
    """Mesma regra do notebook: raças -> 'dog' ou 'cat'.
    Se não bater com nenhuma raça, mantém o nome do arquivo (como no notebook)."""
    if "american_bulldog" in nome_arquivo or "yorkshire_terrier" in nome_arquivo:
        return "dog"
    if "Sphynx" in nome_arquivo or "Abyssinian" in nome_arquivo:
        return "cat"
    return nome_arquivo


def carregar_imagem(caminho):
    """Lê a imagem; remove canal alfa se houver (RGBA -> RGB)."""
    img = imread(caminho)
    if img.ndim == 3 and img.shape[2] == 4:
        img = rgba2rgb(img)
    return img


def redimensionar_cinza(img, res):
    """resize (anti_aliasing=True) e depois rgb2gray, como no notebook.
    Imagens que já são em tons de cinza (2D) não passam pelo rgb2gray."""
    img = resize(img, (res, res), anti_aliasing=True)
    if img.ndim == 3:
        img = rgb2gray(img)
    return img


# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------
def main():
    pasta_hog = os.path.join(PASTA_SAIDA, SUBPASTA_HOG)
    pasta_pca = os.path.join(PASTA_SAIDA, SUBPASTA_PCA)
    os.makedirs(pasta_hog, exist_ok=True)
    os.makedirs(pasta_pca, exist_ok=True)

    arquivos = listar_imagens(PASTA_IMAGENS)
    if not arquivos:
        raise SystemExit(f"Nenhuma imagem encontrada em '{PASTA_IMAGENS}'.")

    classes = [definir_classe(f) for f in arquivos]
    n_sem_classe = sum(c not in ("dog", "cat") for c in classes)
    print(f"{len(arquivos)} imagens | dog: {classes.count('dog')} | "
          f"cat: {classes.count('cat')} | sem classe reconhecida: {n_sem_classe}")
    if n_sem_classe:
        print("   [AVISO] Há imagens cujo nome não bate com nenhuma raça da regra "
              "(basset_hound, saint_bernard, Bombay, Abyssinian). "
              "Nelas a classe ficou igual ao nome do arquivo.")

    # Guarda a ordem das imagens (linha i dos CSVs = arquivo i desta lista)
    pd.DataFrame({"arquivo": arquivos, "classe": classes}).to_csv(
        os.path.join(PASTA_SAIDA, "ordem_imagens.csv"), index=False
    )

    imagens = [carregar_imagem(os.path.join(PASTA_IMAGENS, f)) for f in arquivos]

    inicio_total = time.time()
    total_bases = 0

    for res in RESOLUCOES:
        # Redimensiona + converte para cinza uma vez por resolução
        preparadas = [redimensionar_cinza(img, res) for img in imagens]

        for cel in PIXELS_POR_CELULA:
            t0 = time.time()
            nome_base = f"HOG_{res}_{cel}x{cel}"
            print(f"\n>> {nome_base}")

            try:
                feats = np.array([
                    hog(
                        im,
                        orientations=ORIENTACOES,
                        pixels_per_cell=(cel, cel),
                        cells_per_block=CELULAS_POR_BLOCO,
                        visualize=False,
                    )
                    for im in preparadas
                ])
            except Exception as e:
                print(f"   [ERRO] HOG falhou: {e}")
                continue

            #Base HOG pura
            colunas = [f"hog_feature_{i + 1}" for i in range(feats.shape[1])]
            df_hog = pd.DataFrame(feats, columns=colunas)
            df_hog["classe"] = classes
            df_hog.to_csv(os.path.join(pasta_hog, f"{nome_base}.csv"),
                          sep=",", index=False)
            total_bases += 1
            print(f"   HOG: {feats.shape[1]} atributos (sem contar a coluna classe)")

            #Bases PCA
            for rotulo, var in PCA_VARIANCIAS.items():
                try:
                    pca = PCA(n_components=var, svd_solver="full", random_state=0)
                    comp = pca.fit_transform(feats)
                except Exception as e:
                    print(f"   [ERRO] {rotulo} falhou: {e}")
                    continue

                cols_pca = [f"pca_{i + 1}" for i in range(comp.shape[1])]
                df_pca = pd.DataFrame(comp, columns=cols_pca)
                df_pca["classe"] = classes
                df_pca.to_csv(os.path.join(pasta_pca, f"{nome_base}_{rotulo}.csv"),
                              sep=",", index=False)
                total_bases += 1
                print(f"   {rotulo}: {comp.shape[1]} atributos")

            print(f"   ({time.time() - t0:.1f}s)")

    print(f"\nConcluído: {total_bases} bases em '{PASTA_SAIDA}/' "
          f"({(time.time() - inicio_total) / 60:.1f} min)")


if __name__ == "__main__":
    main()