import pytest

special_options = ('dashboard', 'downloader')


def pytest_addoption(parser):
    """Add options to pytest."""
    for option in special_options:
        parser.addoption(f'--{option}',
                         action='store_true',
                         default=False,
                         help=f'Run {option} workflow tests')


def pytest_collection_modifyitems(config, items):
    """Modify items for pytest."""
    for option in special_options:
        if not config.getoption(f'--{option}'):
            skip_mark = pytest.mark.skip(
                reason=f'Use `--{option}` to include testing {option} workflow.'
            )
            for item in items:
                if option in item.keywords:
                    item.add_marker(skip_mark)
