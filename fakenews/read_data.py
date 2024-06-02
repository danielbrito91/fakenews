import glob
import os
import pathlib
from typing import Literal

import polars as pl


def read_fakebr_corpus(
    glob_path: Literal[
        "data/fakebr-corpus/fake/*.txt",
        "data/fakebr-corpus/true/*.txt",
        "data/fakebr-corpus/size_normalized_texts/fake/*.txt",
        "data/fakebr-corpus/size_normalized_texts/true/*.txt",
    ],
) -> pl.DataFrame:
    """Leitura dos arquivos de texto do FakeBR Corpus.

    https://github.com/roneysco/Fake.br-Corpus/tree/master

    Args:
        glob_path (Literal[str]): Local onde estão armazenados os
        txt com as notícias.

    Returns:
        pl.DataFrame: Polars DataFrame contenedo textos, ids e label (`is_fake`: 0 para false, 1 para true)
    """
    txt_files = glob.glob(glob_path)
    if "true" in glob_path:
        is_fake = 0
    else:
        is_fake = 1

    ids = []
    texts = []
    for file in txt_files:
        file_name = os.path.basename(file)
        id = (file_name.split(".")[0]) + f"-{is_fake}"
        with open(file, "r") as f:
            txt = f.read()

        ids.append(id)
        texts.append(txt)

    return pl.DataFrame(dict(id=ids, texts=texts)).with_columns(
        pl.lit(is_fake).alias("label")
    )


def read_fake_recogna(
        fake_recogna_path: str = "data/fakerecogna/FakeRecogna.xlsx"
) -> pl.DataFrame:

    return (pl.read_excel(fake_recogna_path)
        .select(["Noticia", "Classe"])
        .rename({"Noticia": "text", "Classe": "label"})
        )

    
    