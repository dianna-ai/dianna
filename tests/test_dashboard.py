# pip install pytest-playwright
# playwright install --with-deps
# playwright install firefox
# pip install pytest-playwright-visual
# pytest tests/test_dashboard.py --browser firefox

import time
from contextlib import contextmanager
import pytest
from playwright.sync_api import Page
from playwright.sync_api import expect


LOCAL = False

PORT = '8501' if LOCAL else '8502'
BASE_URL = f'localhost:{PORT}'


def _wait_until_loaded(page: Page, timeout: int = 30):
    """Wait until 'Running...' animation has dissappeared."""
    selector = page.get_by_text('Running...')
    for _ in range(timeout):
        time.sleep(1)
        if not selector.is_visible():
            break


@pytest.fixture(scope='module', autouse=True)
def before_module():
    """Run dashboard in module scope."""
    with run_streamlit():
        yield


@contextmanager
def run_streamlit():
    """Run the dashboard."""
    import subprocess

    if not LOCAL:
        p = subprocess.Popen([
            'dianna-dashboard',
            '--server.port',
            PORT,
            '--server.headless',
            'true',
        ])
        time.sleep(5)

    yield

    if not LOCAL:
        p.kill()


def test_page_load(page: Page, assert_snapshot):
    """Test performance of landing page."""
    page.goto(BASE_URL)

    _wait_until_loaded(page)

    expect(page).to_have_title("Dianna's dashboard")

    assert_snapshot(page.screenshot())


def test_text_page(page: Page, assert_snapshot):
    """Test performance of text page."""
    page.goto(f'{BASE_URL}/Text')

    expect(page).to_have_title('Text · Streamlit')

    page.get_by_label('Load example data').click()
    page.get_by_label('RISE').click()

    _wait_until_loaded(page)

    assert_snapshot(page.screenshot())


def test_image_page(page: Page, assert_snapshot):
    """Test performance of image page."""
    page.goto(f'{BASE_URL}/Images')

    expect(page).to_have_title('Images · Streamlit')

    page.get_by_label('Load example data').click()
    page.get_by_label('RISE').click()

    _wait_until_loaded(page, timeout=45)

    assert_snapshot(page.screenshot())
