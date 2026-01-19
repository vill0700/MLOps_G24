import dataframely as dy
import polars as pl
import pytest

from mlopsg24.data_preprocess import PreprocessData


def test_dummy():
    """
    This is just a dummy to follow if github workflows is triggered
    """
    assert 1==1


def test_existing_path_output_does_not_error():
    """
    Test if the output data directory exists.
    """
    assert PreprocessData().path_output.exists()


@pytest.mark.skipif(
    not PreprocessData().file_data_raw.exists(),
    reason="Data files not found. github does not have access to data file so CI will fail",
)
def test_validate_extracted_data():
    """
    Does a data validation check on the data loaded into the model.
    The design is unconventional. because it is meant as an excercise
    in using pytest for data validation using dataframely.
    """

    class TrainingDataSchema(dy.Schema):
        ann_id = dy.String(nullable=False, min_length=36, max_length=36)
        startdt = dy.Date()
        uri_aktiv = dy.String()
        label = dy.String()
        erhvervsgruppe = dy.Int32()
        erhvervsgruppe_txt = dy.String()
        erhvervsomraade = dy.Int32(nullable=False)
        erhvervsomraade_txt = dy.String(nullable=False)
        annonce_tekst = dy.String(nullable=False)

        @dy.rule()
        def target_has_exactly_22_categories(cls) -> pl.Expr:
            return pl.struct(["erhvervsomraade", "erhvervsomraade_txt"]).n_unique() == 22

    def is_valid(schema, df) -> bool:
        try:
            schema.validate(df, cast=True)
            return True
        except Exception:
            return False

    instance_preprocess = PreprocessData()
    instance_preprocess.extract_input_data()

    assert is_valid(TrainingDataSchema, instance_preprocess.df_jobopslag), "Data validation failed!"


@pytest.mark.skipif(
    not PreprocessData().file_data_raw.exists(),
    reason="Data files not found. github does not have access to data file so CI will fail",
)
def test_target_has_exactly_22_categories():
    """
    Standard pytest design for data validation,
    instad of using dataframely on a polars Dataframe
    """
    instance_preprocess = PreprocessData()
    instance_preprocess.extract_input_data()

    unique_count = instance_preprocess.df_jobopslag.select(["erhvervsomraade", "erhvervsomraade_txt"]).n_unique()

    assert unique_count == 22, "Data validation failed!"
