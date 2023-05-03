"""Module to test the dashboard.

This test module uses (playwright)[https://playwright.dev/python/]
to test the user workflow.

Installation:

    pip install pytest-playwright
    playwright install

Code generation (https://playwright.dev/python/docs/codegen):

    playwright codegen http://localhost:8501

Set `LOCAL=True` to connect to local instance for debugging
"""

import time
from contextlib import contextmanager
import pytest
from playwright.sync_api import Page
from playwright.sync_api import expect

LOCAL = False

PORT = '8501' if LOCAL else '8502'
BASE_URL = f'localhost:{PORT}'

pytestmark = pytest.mark.dashboard


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

    selector = page.get_by_text('Running...')
    selector.wait_for(state='detached')

    expect(page).to_have_title("Dianna's dashboard")
    for selector in (
            page.get_by_role('img', name='0'),
            page.get_by_text('Pages'),
            page.get_by_text('More information'),
    ):
        expect(selector).to_be_visible()


def test_text_page(page: Page, assert_snapshot):
    """Test performance of text page."""
    page.goto(f'{BASE_URL}/Text')

    page.get_by_text('Running...').wait_for(state='detached')

    expect(page).to_have_title('Text · Streamlit')

    selector = page.get_by_text(
        'Add your input data in the left panel to continue')

    expect(selector).to_be_visible()

    page.locator('label').filter(
        has_text='Load example data').locator('span').click()

    expect(page.get_by_text('Select a method to continue')).to_be_visible()

    page.locator('label').filter(has_text='RISE').locator('span').click()

    page.get_by_text('Running...').wait_for(state='detached', timeout=45_000)

    for selector in (
            page.get_by_role('heading', name='RISE').get_by_text('RISE'),
            # first text
            page.get_by_role('heading',
                             name='positive').get_by_text('positive'),
            page.get_by_text(
                'The movie started out great but the ending was dissappointing'
            ).first,
            # second text
            page.get_by_role('heading',
                             name='negative').get_by_text('negative'),
            page.get_by_text(
                'The movie started out great but the ending was dissappointing'
            ).nth(1),
    ):
        expect(selector).to_be_visible()


def test_image_page(page: Page, assert_snapshot):
    """Test performance of image page."""
    page.goto(f'{BASE_URL}/Images')

    page.get_by_text('Running...').wait_for(state='detached')

    expect(page).to_have_title('Images · Streamlit')

    expect(
        page.get_by_text('Add your input data in the left panel to continue')
    ).to_be_visible()

    page.locator('label').filter(
        has_text='Load example data').locator('span').click()

    expect(page.get_by_text('Select a method to continue')).to_be_visible()

    page.locator('label').filter(has_text='RISE').locator('span').click()

    page.get_by_text('Running...').wait_for(state='detached', timeout=45_000)

    for selector in (
            page.get_by_role('heading', name='RISE').get_by_text('RISE'),
            # first image
            page.get_by_role('heading', name='0').get_by_text('0'),
            page.get_by_role('img', name='0').first,
            # second image
            page.get_by_role('heading', name='1').get_by_text('1'),
            page.get_by_role('img', name='0').nth(1),
    ):
        expect(selector).to_be_visible()
