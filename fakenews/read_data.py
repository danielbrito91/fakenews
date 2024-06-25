import glob
import os
import pandas as pd
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
    fake_recogna_path: str = "data/fakerecogna/FakeRecogna.xlsx",
) -> pl.DataFrame:
    return (
        pl.read_excel(fake_recogna_path)
        .select(["Noticia", "Classe"])
        .rename({"Noticia": "text", "Classe": "label"})
    )


def read_fake_true_br(
    url="https://raw.githubusercontent.com/jpchav98/FakeTrue.Br/main/FakeTrueBr_corpus.csv",
):
    df = pl.from_pandas(pd.read_csv(url))

    # Mesmo comprimento para fake e true

    df = (
        df.with_columns(pl.col("fake").str.strip(), pl.col("true").str.strip())
        .with_columns(
            pl.col("fake").str.len_chars().alias("fake_len"),
            pl.col("true").str.len_chars().alias("true_len"),
        )
        .with_columns(
            pl.min_horizontal([pl.col("fake_len"), pl.col("true_len")]).alias("min_len")
        )
        .with_columns(
            pl.col("fake").str.slice(0, length=pl.col("min_len")).alias("fake"),
            pl.col("true").str.slice(0, length=pl.col("min_len")).alias("true"),
        )
        .with_row_count("id")
    )

    assert (
        df.with_columns(
            (pl.col("fake").str.len_chars() - pl.col("true").str.len_chars()).alias(
                "dif"
            )
        )
        .filter(pl.col("dif") != 0)
        .shape[0]
        == 0
    )

    fake = (
        df.select(["fake", "id"])
        .with_columns(pl.lit(1).alias("label"))
        .rename({"fake": "text"})
    )
    true = (
        df.select(["true", "id"])
        .with_columns(pl.lit(0).alias("label"))
        .rename({"true": "text"})
    )

    df = pl.concat([fake, true])

    # Existe um id para cada notícia fake e true
    assert df.shape[0] == (df["id"].max() + 1) * 2

    return df
