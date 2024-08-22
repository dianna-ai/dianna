"""Module to test the dashboard.

This test module uses (playwright)[https://playwright.dev/python/]
to test the user workflow.

Installation:

    pip install pytest-playwright
    playwright install

Make sure that the server is running by:
```bash
cd dianna/dashboard
streamlit run Home.py
```
Then, set variable `LOCAL=True` (see below) to connect to local instance for
debugging. Then, you can run the tests with:

```bash
pytest -v -m dashboard --dashboard
```
See more documentation about dashboard in: dianna/dashboard/readme.md

For Code generation (https://playwright.dev/python/docs/codegen):

    playwright codegen http://localhost:8501
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


def test_page_load(page: Page):
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


def test_text_page(page: Page):
    """Test performance of text page."""
    page.goto(f'{BASE_URL}/Text')

    page.get_by_text('Running...').wait_for(state='detached')

    expect(page).to_have_title('Text')

    # Movie sentiment example
    page.locator("label").filter(has_text="Use an example").locator("div").nth(1).click()
    page.get_by_text("Movie sentiment").click()
    expect(page.get_by_text("Select a method to continue")).to_be_visible()

    page.locator('label').filter(has_text='RISE').locator('span').click()
    page.locator('label').filter(has_text='LIME').locator('span').click()

    page.get_by_text('Running...').wait_for(state='detached', timeout=100_000)

    for selector in (
            page.get_by_role('heading', name='RISE').get_by_text('RISE'),
            page.get_by_role('heading', name='LIME').get_by_text('LIME'),
            # Images for positive (RISE/LIME)
            page.get_by_role('heading',
                             name='positive').get_by_text('positive'),
            page.get_by_role('img', name='0').first,
            page.get_by_role('img', name='0').nth(1),

            # Images for negative (RISE/LIME)
            page.get_by_role('heading',
                             name='negative').get_by_text('negative'),
            page.get_by_role('img', name='0').nth(2),
            page.get_by_role('img', name='0').nth(3),
    ):
        expect(selector).to_be_visible()

    # Own data option
    page.locator("label").filter(has_text="Use your own data").locator("div").nth(1).click()
    selector = page.get_by_text(
        'Add your input data in the left panel to continue')

    expect(selector).to_be_visible(timeout=30_000)

    # Check input panel
    page.get_by_label("Input string").click()
    expect(page.get_by_label("Select model").get_by_test_id("baseButton-secondary")).to_be_visible()
    page.get_by_label("Select labels").get_by_test_id("baseButton-secondary").click()


def test_image_page(page: Page):
    """Test performance of image page."""
    page.goto(f'{BASE_URL}/Images')

    page.get_by_text('Running...').wait_for(state='detached')

    expect(page).to_have_title('Images')

    expect(
        page.get_by_text('Select which input type to')
    ).to_be_visible(timeout=100_000)

    # Digits example
    page.locator("label").filter(has_text="Use an example").locator("div").nth(1).click()
    page.get_by_text("Hand-written digit recognition").click()

    expect(page.get_by_text('Select a method to continue')).to_be_visible()

    page.locator('label').filter(has_text='RISE').locator('span').click()
    page.locator('label').filter(has_text='KernelSHAP').locator('span').click()
    page.locator('label').filter(has_text='LIME').locator('span').click()

    page.get_by_text('Running...').wait_for(state='detached', timeout=45_000)

    for selector in (
            page.get_by_role('heading', name='RISE').get_by_text('RISE'),
            page.get_by_role('heading', name='KernelSHAP').get_by_text('KernelSHAP'),
            page.get_by_role('heading', name='LIME').get_by_text('LIME'),
            # first image
            page.get_by_role('heading', name='0').get_by_text('0'),
            page.get_by_role('img', name='0').first,
            page.get_by_role('img', name='0').nth(1),
            page.get_by_role('img', name='0').nth(2),
            # second image
            page.get_by_role('heading', name='1').get_by_text('1'),
            page.get_by_role('img', name='0').nth(3),
            page.get_by_role('img', name='0').nth(4),
            page.get_by_role('img', name='0').nth(5),
    ):
        expect(selector).to_be_visible(timeout=100_000)

    # Own data
    page.locator("label").filter(has_text="Use your own data").locator("div").nth(1).click()
    expect(page.get_by_label("Select image").get_by_test_id("baseButton-secondary")).to_be_visible()
    page.get_by_label("Select model").get_by_test_id("baseButton-secondary").click()
    page.get_by_label("Select labels").get_by_test_id("baseButton-secondary").click()


def test_timeseries_page(page: Page):
    """Test performance of timeseries page."""
    page.goto(f'{BASE_URL}/Time_series')

    page.get_by_text('Running...').wait_for(state='detached')

    expect(page).to_have_title('Time_series')

    expect(page.get_by_text("Select which input type to")).to_be_visible()

    page.locator("label").filter(has_text="Use an example").locator("div").nth(1).click()
    expect(page.get_by_text("Select an example in the left")).to_be_visible()
    expect(page.get_by_text("Weather")).to_be_visible()
    expect(page.get_by_text("FRB")).to_be_visible()

    # Test weather example
    page.locator("label").filter(has_text="Weather").locator("div").nth(1).click()
    expect(page.get_by_text("Select a method to continue")).to_be_visible()

    page.locator('label').filter(has_text='LIME').locator('span').click()
    page.locator('label').filter(has_text='RISE').locator('span').click()

    page.get_by_text('Running...').wait_for(state='detached', timeout=100_000)

    for selector in (
            page.get_by_role('heading', name='LIME').get_by_text('LIME'),
            page.get_by_role('heading', name='RISE').get_by_text('RISE'),
            # First image
            page.get_by_role('heading', name='winter').get_by_text('winter'),
            page.get_by_role('img', name='0').first,
            page.get_by_role('img', name='0').nth(1),
            # Second image
            page.get_by_role('heading', name='summer').get_by_text('summer'),
            page.get_by_role('img', name='0').nth(2),
            page.get_by_role('img', name='0').nth(3),
    ):
        expect(selector).to_be_visible()

    # Test FRB example
    page.locator("label").filter(has_text="FRB").locator("div").nth(1).click()
    expect(page.get_by_text("Select a method to continue")).to_be_visible()

    page.locator('label').filter(has_text='RISE').locator('span').click()

    page.get_by_text('Running...').wait_for(state='detached', timeout=100_000)

    for selector in (
            page.get_by_role('heading', name='RISE').get_by_text('RISE'),
            # First image
            page.get_by_role('heading', name='FRB').get_by_text('FRB'),
            page.get_by_role('img', name='0').first,
            page.get_by_role('img', name='0').nth(1),
    ):
        expect(selector).to_be_visible()

    # Test using your own data
    page.locator("label").filter(
        has_text="Use your own data").locator("div").nth(1).click()
    page.get_by_label("Select input data").get_by_test_id(
        "baseButton-secondary").click()
    page.get_by_label("Select model").get_by_test_id(
        "baseButton-secondary").click()
    page.get_by_label("Select labels").get_by_test_id(
        "baseButton-secondary").click()


def test_tabular_page(page: Page):
    """Test performance of tabular page"""
    page.goto(f'{BASE_URL}/Tabular')

    page.get_by_text('Running...').wait_for(state='detached')

    expect(page).to_have_title('Tabular')

    expect(page.get_by_text("Select which input type to")).to_be_visible()

    page.locator("label").filter(has_text="Use an example").locator("div").nth(1).click()

    # Test using your own data
    page.locator("label").filter(
        has_text="Use your own data").locator("div").nth(1).click()
    page.get_by_label("Select input data").get_by_test_id(
        "baseButton-secondary").click()
    page.get_by_label("Select model").get_by_test_id(
        "baseButton-secondary").click()
    page.get_by_label("Select training data").get_by_test_id(
        "baseButton-secondary").click()
    page.get_by_label("Select labels in case of classification model").get_by_test_id(
        "baseButton-secondary").click()
