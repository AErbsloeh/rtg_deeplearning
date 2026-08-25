import pytest

from copy import deepcopy

from denspp.offline.dnn import SettingsDataset, DefaultSettingsDataset
from src_dnn.call_dataset import DatasetLoader


@pytest.mark.parametrize("data_type, shape_data, shape_label, shape_dict", [
    ("mnist", (70000, 28, 28), (70000, ), 10),
    ("waveforms", (12000, 280), (12000, ), 12),
    ("martinez", (6409, 32), (6409, ), 0),
    ("quiroga", (15894, 32), (15894, ), 0),
])
def test_download_mnist(data_type: str, shape_data: tuple, shape_label: tuple, shape_dict: dict):
    sets: SettingsDataset = deepcopy(DefaultSettingsDataset)
    sets.data_type = data_type

    dataset = DatasetLoader(settings=sets).load_dataset()
    assert dataset.data.shape == shape_data
    assert dataset.label.shape == shape_label
    assert len(dataset.dict) == shape_dict
