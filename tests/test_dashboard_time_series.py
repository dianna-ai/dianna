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


def test_timeseries_page(page: Page):
    page.tracing.start(screenshots=True, snapshots=True, sources=True)
    """Test performance of timeseries page."""
    page.goto(f'{BASE_URL}/Time_series')

    page.get_by_text('Running...').wait_for(state='detached')

    expect(page).to_have_title('Time_series')

    expect(page.get_by_text("Select which input type to")).to_be_visible(timeout=100_000)

    page.locator("label").filter(has_text="Use an example").locator("div").nth(1).click()
    expect(page.get_by_text("Select an example in the left")).to_be_visible(timeout=200_000)
    expect(page.get_by_text("Weather")).to_be_visible()
    expect(page.get_by_text("FRB")).to_be_visible()

    # Test weather example
    page.locator("label").filter(has_text="Use an example").locator("div").nth(1).click()
    page.locator("label").filter(has_text="Weather").locator("div").nth(1).click()
    expect(page.get_by_text("Select a method to continue")).to_be_visible(timeout=100_000)

    time.sleep(5)
    page.screenshot(path="screenshotweather-methods.png")

    page.locator('label').filter(has_text='LIME').locator('span').click(timeout=200_000)
    page.locator('label').filter(has_text='RISE').locator('span').click(timeout=200_000)
    time.sleep(1)
    page.screenshot(path="screenshotweather-button.png")
    page.get_by_test_id("stNumberInput-StepUp").click(timeout=200_000)
    page.get_by_text('Running...').wait_for(state='detached', timeout=100_000)

    time.sleep(5)

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
    page.locator("label").filter(has_text="Use an example").locator("div").nth(1).click()
    page.locator("label").filter(has_text="FRB").locator("div").nth(1).click()
    expect(page.get_by_text("Select a method to continue")).to_be_visible(timeout=100_000)

    time.sleep(3)

    page.locator('label').filter(has_text='RISE').locator('span').click()

    time.sleep(3)
    page.screenshot(path="screenshotfrbbutton.png")
    page.get_by_test_id("stNumberInput-StepUp").click(timeout=200_000)
    page.get_by_text('Running...').wait_for(state='detached', timeout=100_000)

    time.sleep(5)

    for selector in (
            page.get_by_role('heading', name='RISE').get_by_text('RISE'),
            # First image
            page.get_by_role('heading', name='FRB').get_by_text('FRB'),
            page.get_by_role('img', name='0').first,
            page.get_by_role('img', name='0').nth(1),
            # Second image
            page.get_by_role('heading', name='summer').get_by_text('noise'),
            page.get_by_role('img', name='0').nth(2),
            page.get_by_role('img', name='0').nth(3),
    ):
        expect(selector).to_be_visible(timeout=200_000)

    # Test using your own data
    page.locator("label").filter(
        has_text="Use your own data").locator("div").nth(1).click()
    expect(page.get_by_label("Select input data").get_by_test_id("baseButton-secondary")).to_be_visible()
    expect(page.get_by_label("Select model").get_by_test_id("baseButton-secondary")).to_be_visible()
    expect(page.get_by_label("Select labels").get_by_test_id("baseButton-secondary")).to_be_visible()
    page.tracing.stop(path="tracetimeseries.zip")

