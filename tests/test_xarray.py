import pytest
import numpy as np
import xarray as xr
from dianna.utils import to_xarray, move_axis


def test_xarray_conversion_list():
    data = np.zeros((1, 28, 28, 3))
    labels = ('batch', 'y', 'x', 'channels')

    data_x = to_xarray(data, labels)
    assert data_x.shape == data.shape
    assert data_x.dims == labels


def test_xarray_conversion_dict():
    data = np.zeros((1, 28, 28, 3))
    labels = {0: 'batch', -1: 'channels'}
    expected_labels = ('batch', 'dim_1', 'dim_2', 'channels')

    data_x = to_xarray(data, labels)
    assert data_x.shape == data.shape
    assert data_x.dims == expected_labels


def test_xarray_conversion_required_label():
    data = np.zeros((1, 28, 28, 3))
    labels = ('batch', 'y', 'x', 'channels')
    required = ['non-existent']

    with pytest.raises(AssertionError):
        to_xarray(data, labels, required_labels=required)


def test_xarray_move_axis():
    data = xr.DataArray(np.zeros((4, 1, 28, 28)), dims=('batch', 'channels', 'y', 'x'))
    expected_labels = ('batch', 'y', 'x', 'channels')
    expected_shape = (4, 28, 28, 1)

    data_moved = move_axis(data, 'channels', -1)
    assert data_moved.dims == expected_labels
    assert data_moved.shape == expected_shape
