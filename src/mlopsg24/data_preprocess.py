from pathlib import Path
from loguru import logger
import typer
import torch
from sentence_transformers import SentenceTransformer
import polars as pl

class PreprocessData():

    def __init__(
        self,
        path_text_embedder:Path=Path("models/intfloat/multilingual-e5-large-instruct"),
        file_text_data:Path=Path("data/raw/training_jobopslag.parquet"),
        do_split:bool=True,
    ) -> None:

        self.path_text_embedder = path_text_embedder
        self.file_text_data = file_text_data
        self.do_split = do_split


    def init_text_embedder(self):
        """Initialize SentenceTransformer model for text embeddings."""

        self.text_embedder = SentenceTransformer(
            model_name_or_path = self.path_text_embedder,
            device = 'cuda',
        )

    def embed(self):
        df_jobopslag = pl.read_parquet(self.file_text_data)

        list_sentences = (
            df_jobopslag
            .select("ann_id")
            .to_series()
            .to_list()
        )

        return self.text_embedder.encode(
            sentences = list_sentences,
            prompt="Instruct: Retrieve semantically similar text.\n Query: ",
            convert_to_tensor=True,
        )

    def __call__(self):
        self.init_text_embedder()

        if self.do_split:
            # split targets, features into training, test and training
            pass
        else:
            # return targets, features



if __name__ == "__main__":
    # typer.run(embed_texts)
