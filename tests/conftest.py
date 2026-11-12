import pytest

special_options = ('dashboard', 'downloader')


def pytest_addoption(parser):
    """Add options to pytest."""
    for option in special_options:
        parser.addoption(f'--{option}',
                         action='store_true',
                         default=False,
                         help=f'Run {option} workflow tests')
    parser.addoption('--web',
                     action='store_true',
                     default=False,
                     help='Run the dashboard tests on the browser version in web/_site '
                     '(build it with python web/build.py)')


@pytest.fixture(scope='session')
def dashboard(request):
    """Run the dashboard for the Playwright tests."""
    from tests.dashboard_helpers import run_dashboard
    with run_dashboard(web=request.config.getoption('--web')):
        yield


@pytest.fixture(scope='session')
def _web_page(browser):
    """One browser page for all tests of the browser version, so it loads Python and the packages only once."""
    context = browser.new_context()
    yield context.new_page()
    context.close()


@pytest.fixture
def page(page, request):
    """Playwright's page, or with --web the shared page (open_page() in tests/dashboard_helpers.py switches pages)."""
    if request.config.getoption('--web'):
        return request.getfixturevalue('_web_page')
    return page


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
