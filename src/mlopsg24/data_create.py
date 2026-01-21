# %%[markdown]
# This module is not meant to be run by others than Esben Opstrup,
# because it requres access to a private database
# and database credentials stored in the internal package `bmdb``
# %%
import html
import uuid
from pathlib import Path

import polars as pl
import torch
from gliner2 import GLiNER2
from loguru import logger
from tqdm import tqdm

# Internal package, not publicly avaiable
# from bmdb import db_uri


def prepare_text(
    df_jobopslag_raw: pl.DataFrame,
    model_gliner2: GLiNER2,
) -> pl.DataFrame:
    # Non batched version as prototype. Later Vibecode refactored into batched version
    def entities_to_natural_language(jobopslag_text: str, model_gliner2: GLiNER2) -> str:
        entities = ["stillingsbetegnelser", "kompetencer", "arbejdsopgaver"]

        dict_extracted = model_gliner2.extract_entities(
            text=jobopslag_text,
            entity_types=entities,
        )

        str_natural_language = (
            f"{', '.join(dict_extracted['entities']['stillingsbetegnelser'])} ."
            f"{', '.join(dict_extracted['entities']['kompetencer'])}, "
            f"{', '.join(dict_extracted['entities']['arbejdsopgaver'])} "
        )

        return str_natural_language

    df_results = (
        df_jobopslag_raw.with_columns(
            pl.col("ann_id").map_elements(lambda row: str(uuid.uuid4()), return_dtype=pl.String)
        )
        .with_columns(
            pl.col("annonce_tekst").map_elements(
                lambda text: entities_to_natural_language(jobopslag_text=text, model_gliner2=model_gliner2),
                return_dtype=pl.String,
            )
        )
        .drop_nulls()
        .filter(pl.col("annonce_tekst").str.len_chars() >= 10)
    )

    return df_results


def augment_jobopslag_text(
    text: str,
    model_gliner2: GLiNER2,
) -> str:
    """
    Process a single text through GLiNER2 extraction and formatting.

    Args:
        text: Input text to process
        model_gliner2: GLiNER2 model instance

    Returns:
        Formatted string with extracted entities or empty string on error
    """

    entities_to_extract: list = ["stillingsbetegnelser", "kompetencer", "arbejdsopgaver"]

    try:
        dict_extracted = model_gliner2.extract_entities(text, entities_to_extract)
        entities = dict_extracted.get("entities", {})

        entities_stil = entities.get("stillingsbetegnelser", None)
        entities_komp = entities.get("kompetencer")
        entities_opg = entities.get("arbejdsopgaver")

        if not any([entities_stil, entities_komp, entities_opg]):
            logger.error("FAILED to extract neither stillingsbetegnelser, kompetencer nor arbejdsopgaver")

        return f"{', '.join(entities_stil)}, {', '.join(entities_komp)}, {', '.join(entities_opg)}"

    except Exception as e:
        logger.error(f"FAILED extraction: {e}")
        return ""


def prepare_text_bactched(df_jobopslag_raw: pl.DataFrame, model_gliner2: GLiNER2, batch_size: int = 32) -> pl.DataFrame:
    """
    1) GLiNER2 cleans jobopslag text to just the parts of the text expected to be
    good signal for stillingsbetegnelse drop nulls just to make it easier
    to work with in the course. It also removes Personal Data information.
    Expects jobopslag in Danish. Does NER for ["stillingsbetegnelser", "kompetencer", "arbejdsopgaver"]
    on Danish jobopslag. This outputs a dict.
    2) Rewrites the GLINER2 dict into a text more like natural language,
    which fits better to a sentencetransformer text embedding model eg. E5
    3) Replaces jobopslag ID with UUID to make the data even more anonymous
    4) Drops nulls rows, unescape html and filter short strings
    for cleaner data to training of model.
    """

    # 1. Extract texts to process
    all_texts = df_jobopslag_raw["annonce_tekst"].to_list()
    results = []

    # 1. Batch processing loop of GLiNER2
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Processing Batches"):
        batch = all_texts[i : i + batch_size]

        # Inference
        for text in batch:
            result = augment_jobopslag_text(text, model_gliner2)
            results.append(result)

    # 3. UUID generation
    uuids = [str(uuid.uuid4()) for _ in range(df_jobopslag_raw.height)]

    df_cleaned_final = (
        df_jobopslag_raw.with_columns(
            [
                # 3. UUIDs
                pl.Series("ann_id", uuids),
                # Add rewritten text
                pl.Series("annonce_tekst", results),
            ]
        )
        # 4. final clean
        .drop_nulls()
        .filter(pl.col("annonce_tekst").str.len_chars() > 10)
        .with_columns(
            pl.col("annonce_tekst").map_elements(lambda x: html.unescape(x) if x else x, return_dtype=pl.String)
        )
    )

    return df_cleaned_final


def pipeline_data_create(
    path_model_gliner2: Path = Path("models/fastino/gliner2-multi-v1"),
    path_output: Path = Path("data/raw/training_jobopslag.parquet"),
    subset_size: int = 1000000,
):
    """
    1. Extract data from private database
    2. Transforms text on id's and rewrites job vacancy text
    3. Load / exports to parquet file
    """

    logger.info("START pipeline_data_create()")

    logger.info("Extracting data")
    df_jobopslag_raw = pl.read_database_uri(
        query=f"""--sql
            SELECT
                a.ann_id::INT,
                a.startdt,
                c.uri_aktiv,
                c.label,
                c.erhvervsgruppe,
                c.erhvervsgruppe_txt,
                c.erhvervsomraade,
                c.erhvervsomraade_txt,
                b.annonce_tekst
            FROM lg_voaoverv.j2_jjobtype AS a
            INNER JOIN (
                SELECT DISTINCT ON (ann_id) *
                FROM lg_voaoverv.janntext_total
                WHERE startdt >= slutdt - INTERVAL '1 year'
                ORDER BY ann_id, startdt DESC
                ) AS b ON a.ann_id=b.ann_id
            INNER JOIN (
                SELECT DISTINCT ON (uri_key) *
                FROM pr_dim_es_adhoc.dim_escostar_hist
                ORDER BY uri_key, slutdt DESC
                ) AS c ON a.concepturida = c.uri_key
            WHERE DATE_PART('year',a.startdt) BETWEEN 2022 AND 2024 -- NOTE:years with clean data

            LIMIT {subset_size}
        """,
        uri=db_uri(),
    )

    logger.info("Loading hf model")
    model_gliner2 = GLiNER2.from_pretrained(path_model_gliner2)

    if torch.cuda.is_available():
        model_gliner2.to("cuda")

    logger.info("Transforming data")

    df_cleaned = prepare_text_bactched(
        df_jobopslag_raw=df_jobopslag_raw,
        model_gliner2=model_gliner2,
    )

    logger.info("Exporting data")
    df_cleaned.write_parquet(path_output)

    logger.info("Diagnostics")
    logger.info(df_cleaned.group_by("erhvervsomraade_txt").len().sort(by="len"))
    df_cleaned.group_by("label").len().to_pandas().plot(kind="hist", bins=100)


if __name__ == "__main__":
    # cannot be run by others than Esben Opstrup
    pipeline_data_create()

# %%
