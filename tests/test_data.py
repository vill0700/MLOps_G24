import os

import pytest
from torch.utils.data import Dataset

from mlopsg24.data import MyDataset

file_path = "data/raw"


@pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)
