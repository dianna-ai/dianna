import pytest


def pytest_addoption(parser):
    """Add options to pytest."""
    parser.addoption('--dashboard',
                     action='store_true',
                     default=False,
                     help='Run dashboard workflow tests')


def pytest_collection_modifyitems(config, items):
    """Modify items for pytest."""
    if config.getoption('--dashboard'):
        return

    skip_dashboard = pytest.mark.skip(
        reason='Use `-m dashboard` to test dashboard workflow.')
    for item in items:
        if 'dashboard' in item.keywords:
            item.add_marker(skip_dashboard)
