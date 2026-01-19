import dataframely as dy
import polars as pl
import pytest

from mlopsg24.data_preprocess import PreprocessData


def test_dummy():
    "This is just a dummy to see if github workflows is triggered"
    assert 1==1

def test_path_output_creation(tmp_path):
    """
    Test that the output directory is created if it does not exist
    upon initialization of PreprocessData.
    """
    # 1. Setup a path that definitely does not exist yet
    # tmp_path is a pathlib.Path object provided by pytest
    custom_output_dir = tmp_path / "new_processed_folder"

    # Verify it doesn't exist before we start
    assert not custom_output_dir.exists()

    # 2. Instantiate the class
    # We pass the non-existent path to the constructor
    preprocessor = PreprocessData(path_output=custom_output_dir)

    # 3. Assertions
    assert preprocessor.path_output.exists(), "The directory should have been created."
    assert preprocessor.path_output.is_dir(), "The path should be a directory."


def test_existing_path_output_does_not_error(tmp_path):
    """
    Test that if the directory already exists, the class initializes
    without raising an FileExistsError.
    """
    # 1. Pre-create the directory
    existing_dir = tmp_path / "already_here"
    existing_dir.mkdir()

    # 2. Instantiate (should not raise exception due to exist_ok=True)
    try:
        preprocessor = PreprocessData(path_output=existing_dir)
    except Exception as e:
        pytest.fail(f"Initialization raised an exception on existing directory: {e}")

    # 3. Final Check
    assert preprocessor.path_output.exists()


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
