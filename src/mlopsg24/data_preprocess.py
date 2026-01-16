# %%
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class PreprocessData:
    def __init__(
        self,
        path_text_embedder: str | Path = Path("models/intfloat/multilingual-e5-large-instruct"),
        file_data_raw: Path = Path("data/raw/training_jobopslag.parquet"),
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        batch_size: int = 32,
        path_output: Path = Path("data/processed"),
        column_target_class: str = "erhvervsomraade_txt",
        column_text: str = "annonce_tekst",
        embedding_prefix: str = (
            "query: "
            "Classify the following extracted texts of occupation, skills "
            "and tasks from a Danish job vacancy into job category"
        ),
    ) -> None:
        self.path_text_embedder = path_text_embedder
        self.file_data_raw = file_data_raw
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.path_output = path_output
        self.column_target_class = column_target_class
        self.column_text = column_text
        self.embedding_prefix = embedding_prefix
        self.batch_size = batch_size

        if not self.path_output.exists():
            logger.info(f"Creating data processed output directory: {self.path_output}")
            self.path_output.mkdir(parents=True, exist_ok=True)

    def extract_input_data(self) -> None:
        """
        Mount raw data to dataframe
        """

        logger.info(f"Reading data from {self.file_data_raw}")
        self.df_jobopslag = pl.read_parquet(self.file_data_raw)

    def init_text_embedder(self, gpu: bool = True):
        """
        Initialize SentenceTransformer model for text embeddings.
        Default tries to use GPU
        """
        logger.info(f"Loading text embedder from {self.path_text_embedder}")

        self.text_embedder = SentenceTransformer(
            model_name_or_path=str(self.path_text_embedder),
            device="cuda" if (torch.cuda.is_available() and gpu) else "cpu",
        )

    def create_x_features(self):
        """
        Create text embeddings used as x features.
        Processes in batches to manage GPU memory efficiently.
        """

        list_sentences = self.df_jobopslag.select(self.column_text).to_series().to_list()

        logger.info(f"Creating embeddings for {len(list_sentences)} observations in batches of {self.batch_size}")

        # Initialize list to store embeddings from each batch
        all_embeddings = []

        # Calculate number of batches
        num_batches = (len(list_sentences) + self.batch_size - 1) // self.batch_size

        # Process in batches with progress bar
        for i in tqdm(range(0, len(list_sentences), self.batch_size), desc="Creating embeddings", total=num_batches):
            # Get current batch
            batch_sentences = list_sentences[i : i + self.batch_size]

            # Generate embeddings for batch
            batch_embeddings = self.text_embedder.encode(
                sentences=batch_sentences,
                convert_to_tensor=True,
                show_progress_bar=False,  # Disable inner progress bar
                prompt=self.embedding_prefix,
                normalize_embeddings=True,
            )

            # Move to CPU and convert to numpy to free GPU memory
            batch_embeddings_cpu = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings_cpu)

            # Clear GPU cache
            if torch.cuda.is_available():
                del batch_embeddings
                torch.cuda.empty_cache()

        # Concatenate all batches and convert back to tensor
        self.x_features = torch.from_numpy(np.vstack(all_embeddings))

        logger.info(f"x features embeddings shape: {self.x_features.shape}")

    def create_y_target(self):
        """
        Create categorical y target features
        Saves mapping of y idx to classes to parquet table
        """

        y_categories = self.df_jobopslag.select(self.column_target_class).to_series()

        # Create mapping from categories to indices
        unique_categories = y_categories.unique().sort()
        category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}

        # Convert to tensor
        self.y_targets = torch.tensor([category_to_idx[cat] for cat in y_categories.to_list()], dtype=torch.long)

        # save mapping of categories
        (
            pl.DataFrame(
                data=list(category_to_idx.items()),
                schema=("categori", "idx"),
                orient="row",
            ).write_parquet(self.path_output / "category_mapping.parquet")
        )

        logger.info(f"Targets shape: {self.y_targets.shape}")
        logger.info(f"Number of classes: {len(unique_categories)}")

    def split_data(self) -> None:
        """
        Split data into train, validation, and test sets.
        """
        logger.info("Splitting data into train/val/test sets")

        x_temp, self.x_test, y_temp, self.y_test = train_test_split(
            self.x_features,
            self.y_targets,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y_targets,  # Maintain class distribution
        )

        # Second split: separate validation from training
        # Adjust val_size relative to temp size to ensure equal distribution
        val_size_adjusted = self.val_size / (1 - self.test_size)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=y_temp
        )

        logger.info(f"Train set: {self.x_train.shape[0]} samples")
        logger.info(f"Validation set: {self.x_val.shape[0]} samples")
        logger.info(f"Test set: {self.x_test.shape[0]} samples")

    def save_data(self) -> None:
        """
        Save train,test,validation for x and y as tensors to .pt files.
        """
        logger.info("Saving data")

        torch.save(self.x_train, self.path_output / "x_train.pt")
        torch.save(self.x_val, self.path_output / "x_val.pt")
        torch.save(self.x_test, self.path_output / "x_test.pt")
        torch.save(self.y_train, self.path_output / "y_train.pt")
        torch.save(self.y_val, self.path_output / "y_val.pt")
        torch.save(self.y_test, self.path_output / "y_test.pt")

        logger.info(f"All tensors saved successfully to {self.path_output}")

    def main(self):
        """
        Main preprocessing pipeline.
        """
        logger.info("Starting preprocessing pipeline")

        self.extract_input_data()
        self.init_text_embedder()
        self.create_x_features()
        self.create_y_target()
        self.split_data()
        self.save_data()

        logger.info("Preprocessing complete with train/val/test split")


if __name__ == "__main__":
    # Run preprocessing
    PreprocessData(
        batch_size=256,
        path_text_embedder="intfloat/multilingual-e5-large-instruct",
    ).main()

# %%
