import pytest


def pytest_runtest_setup(item):
    """Configure pytest per item."""
    # Do not test dashboard workflow by default
    if 'dashboard' in (mark.name for mark in item.iter_markers()):
        pytest.skip('Use `-m dashboard` to test dashboard workflow.')
