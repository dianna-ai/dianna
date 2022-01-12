import numpy as np
import pytest
import xarray as xr
from dianna import utils


def test_xarray_conversion_list():
    """Tests if xarray has correct dimensions and labels after creation."""
    data = np.zeros((1, 28, 28, 3))
    labels = ('batch', 'y', 'x', 'channels')

    data_x = utils.to_xarray(data, labels)

    assert data_x.shape == data.shape
    assert data_x.dims == labels


def test_xarray_conversion_dict():
    """Tests if xarray has correct dimensions and labels after creation with incomplete label dict."""
    data = np.zeros((1, 28, 28, 3))
    labels = {0: 'batch', -1: 'channels'}
    expected_labels = ('batch', 'dim_1', 'dim_2', 'channels')

    data_x = utils.to_xarray(data, labels)

    assert data_x.shape == data.shape
    assert data_x.dims == expected_labels


def test_xarray_conversion_required_label():
    """Tests if error is raised when omitting some required label."""
    data = np.zeros((1, 28, 28, 3))
    labels = ('batch', 'y', 'x', 'channels')
    required = ['non-existent']

    with pytest.raises(AssertionError):
        utils.to_xarray(data, labels, required_labels=required)


def test_xarray_move_axis():
    """Tests if moving an axis results in expected shape and dimensions."""
    data = xr.DataArray(np.zeros((4, 1, 28, 28)), dims=('batch', 'channels', 'y', 'x'))
    expected_labels = ('batch', 'y', 'x', 'channels')
    expected_shape = (4, 28, 28, 1)

    data_moved = utils.move_axis(data, 'channels', -1)

    assert data_moved.dims == expected_labels
    assert data_moved.shape == expected_shape


def test_xarray_move_axis_nonexistent():
    """Tests if error is raised when trying to move some nonexistent axis."""
    data = xr.DataArray(np.zeros((4, 1, 28, 28)), dims=('batch', 'channels', 'y', 'x'))
    nonexistent_label = 'foo'

    with pytest.raises(ValueError):
        utils.move_axis(data, nonexistent_label, 0)
