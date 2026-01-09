#%%[markdown]


#%%
import polars as pl
import uuid
from gliner2 import GLiNER2

#Internal
from bmdb import db_uri

#%% GLINER
extractor = GLiNER2.from_pretrained("/data/projects/overvaag/ESHA/hf_models/gliner2-multi-v1")

#%%
def entities_to_natural_language(jobopslag_text:str) -> str:
    """
    Expects jobopslag in Danish.
    1) Does NER for ["stillingsbetegnelser", "kompetencer", "arbejdsopgaver"]
    on danish jobopslag. This outputs a dict.
    2) Rewrites the dict into a text more like natural language,
    which fits better to a sentencetransformer text embedding model eg. E5
    Removes Personal Data information
    """

    entities = ["stillingsbetegnelser", "kompetencer", "arbejdsopgaver"]

    dict_extracted = extractor.extract_entities(
        text = jobopslag_text,
        entity_types = entities,
    )

    str_natural_language = (
        # "Stillingsbetegnelser er "
        f"{', '.join(dict_extracted['entities']['stillingsbetegnelser'])} ."
        # f". Kompetencer og arbejdsopgaver er "
        f"{', '.join(dict_extracted['entities']['kompetencer'])}, "
        f"{', '.join(dict_extracted['entities']['arbejdsopgaver'])} "
    )

    return str_natural_language

#%% EXTRACT form database
df_jobopslag_raw = pl.read_database_uri(query="""--sql
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
    WHERE DATE_PART('year',a.startdt) BETWEEN 2022 AND 2024 -- years with cleanest data

""", uri=db_uri())

#%%
df_cleaned = (

    df_jobopslag_raw

    .sample(100) #NOTE:DEV

    # replacing jobopslag ID make the data even more anonymous
    .with_columns(
        pl.col('ann_id').map_elements(
            lambda row: str(uuid.uuid4()),
            return_dtype=pl.String
        )
    )

    # clean jobopslag just the text expected to be good signal for stillingsbetegnelse
    .with_columns(
        pl.col('annonce_tekst').map_elements(
            lambda row: entities_to_natural_language(row),
            return_dtype=pl.String
        )
    )

    # drop nulls just to make it easier to work with in the course
    .drop_nulls()

)

#%%
df_cleaned['annonce_tekst'].sample(10)

#%%
# save output
df_cleaned.write_parquet("training_jobopslag.parquet")

#%%
df_cleaned.group_by('label').len().to_pandas().plot(kind='hist', bins=100)
#%%
df_cleaned.group_by('erhvervsomraade_txt').len().sort(by='len')
#%%


if __name__ == "__main__":
