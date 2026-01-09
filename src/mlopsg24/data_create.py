#%%[markdown]
# This module is not meant to be run by others than Esben Opstrup,
# because it requres access to a private database
# and database credentials stored in the internal package `bmdb``
#%%
import polars as pl
import uuid
from gliner2 import GLiNER2
from pathlib import Path
from loguru import logger

#Internal package, not publicly avaiable
from bmdb import db_uri


def entities_to_natural_language(
    jobopslag_text:str,
    model_gliner2:GLiNER2
) -> str:
    """
    Expects jobopslag in Danish.
    1) Does NER for ["stillingsbetegnelser", "kompetencer", "arbejdsopgaver"]
    on danish jobopslag. This outputs a dict.
    2) Rewrites the dict into a text more like natural language,
    which fits better to a sentencetransformer text embedding model eg. E5
    Removes Personal Data information
    """

    entities = ["stillingsbetegnelser", "kompetencer", "arbejdsopgaver"]

    dict_extracted = model_gliner2.extract_entities(
        text = jobopslag_text,
        entity_types = entities,
    )

    str_natural_language = (
        f"{', '.join(dict_extracted['entities']['stillingsbetegnelser'])} ."
        f"{', '.join(dict_extracted['entities']['kompetencer'])}, "
        f"{', '.join(dict_extracted['entities']['arbejdsopgaver'])} "
    )

    return str_natural_language

def prepare_text(
    df_jobopslag_raw:pl.DataFrame,
    model_gliner2:GLiNER2,
) -> pl.DataFrame:
    """
    - Replaces jobopslag ID with UUID to make the data even more anonymous
    - Cleans jobopslag text to just the parts of the text expected to be
    good signal for stillingsbetegnelse drop nulls just to make it easier
    to work with in the course
    """

    return (

        df_jobopslag_raw

        .with_columns(
            pl.col('ann_id').map_elements(
                lambda row: str(uuid.uuid4()),
                return_dtype=pl.String
            )
        )

        .with_columns(
            pl.col('annonce_tekst').map_elements(
                lambda text: entities_to_natural_language(
                    jobopslag_text=text,
                    model_gliner2=model_gliner2
                ),
                return_dtype=pl.String
            )
        )

        .drop_nulls()

    )

def pipeline_data_create(
    path_model_gliner2:Path=Path("models/fastino/gliner2-multi-v1"),
    path_output:Path=Path("data/raw/training_jobopslag.parquet"),
):
    """
    1. Extract data from private database
    2. Transforms text on id's and rewrites job vacancy text
    3. Load / exports to parquet file
    """

    logger.info("START pipeline_data_create()")

    logger.info("Extracting data")
    df_jobopslag_raw = pl.read_database_uri(
        query="""--sql
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

            LIMIT 100 --DEV
        """,
        uri=db_uri()
    )

    logger.info("Loading hf model")
    model_gliner2 = GLiNER2.from_pretrained(path_model_gliner2)

    logger.info("Transforming data")
    df_cleaned = prepare_text(
        df_jobopslag_raw=df_jobopslag_raw,
        model_gliner2=model_gliner2,
    )

    logger.info("Exporting data")
    df_cleaned.write_parquet(path_output)

    logger.info("Diagnostics")
    # logger.info(df_cleaned.group_by('label').len().to_pandas().plot(kind='hist', bins=100))
    logger.info(df_cleaned.group_by('erhvervsomraade_txt').len().sort(by='len'))


if __name__ == "__main__":

    # pipeline_data_create()

    pipeline_data_create(
        path_model_gliner2=Path("/data/projects/overvaag/ESHA/hf_models/gliner2-multi-v1"),
        path_output=Path("/data/projects/overvaag/ESHA/mlops_course/MLOps_G24/data/raw/training_jobopslag.parquet"),
    )
