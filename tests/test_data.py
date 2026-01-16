from torch.utils.data import Dataset
import os
from mlopsg24.data import MyDataset
import pytest

file_path = "data/raw"


@pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)
