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


def test_tabular_page(page: Page):
    """Test performance of tabular page."""
    page.goto(f'{BASE_URL}/Tabular')

    page.get_by_text('Running...').wait_for(state='detached')

    expect(page).to_have_title('Tabular')

    expect(page.get_by_text("Select which input type to")).to_be_visible(timeout=100_000)

    time.sleep(5)

    # Test using your own data
    page.locator("label").filter(
        has_text="Use your own data").locator("div").nth(1).click()

    #page.get_by_label("Select tabular data").get_by_test_id("baseButton-secondary").click(timeout=200_000)
    #page.get_by_label("Select model").get_by_test_id("baseButton-secondary").click(timeout=200_000)
    #page.get_by_label("Select training data").get_by_test_id("baseButton-secondary").click(timeout=200_000)
    #page.get_by_label("Select labels in case of").get_by_test_id("baseButton-secondary").click(timeout=200_000)
    expect(page.get_by_label("Select tabular data").get_by_test_id("baseButton-secondary")).to_be_visible()
    expect(page.get_by_label("Select model").get_by_test_id("baseButton-secondary")).to_be_visible()
    expect(page.get_by_label("Select training data").get_by_test_id("baseButton-secondary")).to_be_visible()
    expect(page.get_by_label("Select labels in case of").get_by_test_id("baseButton-secondary")).to_be_visible()


def test_tabular_sunshine(page: Page):
    """Test tabular sunshine example."""
    page.goto(f'{BASE_URL}/Tabular')

    page.get_by_text('Running...').wait_for(state='detached')

    expect(page).to_have_title('Tabular')

    expect(page.get_by_text("Select which input type to")).to_be_visible(timeout=100_000)

    page.locator("label").filter(has_text="Use an example").locator("div").nth(1).click()
    expect(page.get_by_text("Select an example in the left")).to_be_visible()
    expect(page.get_by_text("Sunshine hours prediction")).to_be_visible()
    expect(page.get_by_text("Penguin identification")).to_be_visible()

    # Test sunshine example
    page.locator("label").filter(has_text="Use an example").locator("div").nth(1).click()
    page.locator("label").filter(has_text="Sunshine hours prediction").locator("div").nth(1).click()
    expect(page.get_by_text("Select a method to continue")).to_be_visible(timeout=100_000)

    time.sleep(5)

    page.locator("label").filter(has_text="RISE").locator("span").click()
    page.locator("label").filter(has_text="LIME").locator("span").click()
    page.locator("label").filter(has_text="KernelSHAP").locator("span").click()
    page.locator("summary").filter(has_text="Click to modify RISE").get_by_test_id("stExpanderToggleIcon").click()

    time.sleep(3)

    expect(page.get_by_text("Select the input data by")).to_be_visible(timeout=100_000)
    page.frame_locator("iframe[title=\"st_aggrid\\.agGrid\"]").get_by_role(
        "gridcell", name="10", exact=True).click()
    page.get_by_text('Running...').wait_for(state='detached', timeout=200_000)

    time.sleep(3)

    expect(page.get_by_text("3.07")).to_be_visible(timeout=200_000)

    for selector in (
            page.get_by_role('heading', name='RISE').get_by_text('RISE'),
            page.get_by_role('heading', name='KernelSHAP').get_by_text('KernelSHAP'),
            page.get_by_role('heading', name='LIME').get_by_text('LIME'),
            page.get_by_role('img', name='0').first,
            page.get_by_role('img', name='0').nth(1),
            page.get_by_role('img', name='0').nth(2),
    ):
        expect(selector).to_be_visible(timeout=100_000)


def test_tabular_penguin(page: Page):
    """Test performance of tabular penguin example."""
    page.goto(f'{BASE_URL}/Tabular')
    page.get_by_text('Running...').wait_for(state='detached')

    expect(page).to_have_title('Tabular')
    expect(page.get_by_text("Select which input type to")).to_be_visible(timeout=100_000)

    page.locator("label").filter(has_text="Use an example").locator("div").nth(1).click()
    expect(page.get_by_text("Select an example in the left")).to_be_visible()
    expect(page.get_by_text("Sunshine hours prediction")).to_be_visible()
    expect(page.get_by_text("Penguin identification")).to_be_visible()

    # Test sunshine example
    page.locator("label").filter(has_text="Use an example").locator("div").nth(1).click()
    page.locator("label").filter(has_text="Penguin identification").locator("div").nth(1).click()
    expect(page.get_by_text("Select a method to continue")).to_be_visible(timeout=100_000)

    time.sleep(5)

    page.locator("label").filter(has_text="RISE").locator("span").click(timeout=300_000)
    page.locator("label").filter(has_text="LIME").locator("span").click(timeout=300_000)
    page.locator("label").filter(has_text="KernelSHAP").locator("span").click(timeout=300_000)

    expect(page.get_by_text("Select the input data by")).to_be_visible(timeout=300_000)
    page.frame_locator("iframe[title=\"st_aggrid\\.agGrid\"]").get_by_role(
        "gridcell", name="10", exact=True).click()
    page.get_by_text('Running...').wait_for(state='detached', timeout=300_000)

    time.sleep(5)

    for selector in (
        page.get_by_text('Predicted class:'),
        page.get_by_test_id('stMetricValue').get_by_text('Gentoo'),
        page.get_by_role('heading', name='RISE').get_by_text('RISE'),
        page.get_by_role('heading', name='KernelSHAP').get_by_text('KernelSHAP'),
        page.get_by_role('heading', name='LIME').get_by_text('LIME'),
        page.get_by_role('img', name='0').first,
        page.get_by_role('img', name='0').nth(1),
        page.get_by_role('img', name='0').nth(2),
    ):
        expect(selector).to_be_visible(timeout=200_000)
