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


def test_image_page(page: Page):
    """Test performance of image page."""
    page.set_viewport_size({"width": 1920, "height": 1080})

    page.goto(f'{BASE_URL}/Images')

    page.get_by_text('Running...').wait_for(state='detached')

    expect(page).to_have_title('Images')

    expect(
        page.get_by_text('Select which input type to')
    ).to_be_visible(timeout=100_000)

    # Digits example
    page.locator("label").filter(has_text="Use an example").locator("div").nth(1).click()
    page.get_by_text("Hand-written digit recognition").click()

    expect(page.get_by_text('Select a method to continue')).to_be_visible(timeout=100_000)

    time.sleep(2)

    page.locator('label').filter(has_text='RISE').locator('span').click()
    page.locator('label').filter(has_text='KernelSHAP').locator('span').click()
    page.locator('label').filter(has_text='LIME').locator('span').click()

    page.get_by_label("Number of top classes to show").fill("2")
    page.get_by_label("Number of top classes to show").press("Enter")
    page.get_by_text('Running...').wait_for(state='detached', timeout=100_000)

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
        expect(selector).to_be_visible(timeout=200_000)

    # Own data
    page.locator("label").filter(has_text="Use your own data").locator("div").nth(1).click()

    page.get_by_label("Select image").click()
    page.get_by_label("Select model").click()
    page.get_by_label("Select labels").click()
