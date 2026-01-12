import pytest
from pathlib import Path

from src.mlopsg24.data_preprocess import PreprocessData

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