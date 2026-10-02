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
from tests.dashboard_helpers import wait_streamlit_ready

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

    wait_streamlit_ready(page)

    expect(page).to_have_title('Images')

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

    # Wait for all result images to appear: 2 classes × 3 methods = 6 images.
    # Exclude the static colormap image from the count.
    # This is the definitive signal that both class rows are fully computed.
    # Avoid relying on 'Running...' which may briefly disappear between
    # Streamlit re-runs, causing a premature return before class 1 is rendered.
    imgs = page.locator('img:not([alt="Colormap"])')
    expect(imgs).to_have_count(6, timeout=600_000)

    for selector in (
            page.get_by_role('heading', name='RISE').get_by_text('RISE'),
            page.get_by_role('heading', name='KernelSHAP').get_by_text('KernelSHAP'),
            page.get_by_role('heading', name='LIME').get_by_text('LIME'),
            # first class label
            page.get_by_text('Class: 0'),
            # second class label
            page.get_by_text('Class: 1'),
    ):
        expect(selector).to_be_visible(timeout=30_000)

    # Own data
    page.locator("label").filter(has_text="Use your own data").locator("div").nth(1).click()

    page.get_by_label("Select image").click()
    page.get_by_label("Select model").click()
    page.get_by_label("Select labels").click()


def test_image_page_draw_digit(page: Page):
    """Test drawing a digit on the canvas of the MNIST example."""
    page.set_viewport_size({"width": 1920, "height": 1080})

    page.goto(f'{BASE_URL}/Images')

    wait_streamlit_ready(page)

    page.locator("label").filter(has_text="Use an example").locator("div").nth(1).click()
    page.get_by_text("Hand-written digit recognition").click()
    page.get_by_text("Draw your own digit").click()

    expect(page.get_by_text('Draw a 0 or 1 in the left panel to continue')).to_be_visible(timeout=30_000)

    # draw a vertical stroke (a "1"); stay clear of the toolbar hovering over the top of the canvas
    box = page.locator('canvas.upper-canvas').bounding_box()
    x = box['x'] + box['width'] / 2
    page.mouse.move(x, box['y'] + 0.25 * box['height'])
    page.mouse.down()
    page.mouse.move(x, box['y'] + 0.85 * box['height'], steps=20)
    page.mouse.up()

    expect(page.get_by_text('Select a method to continue')).to_be_visible(timeout=100_000)
    page.get_by_text('RISE', exact=True).click()

    expect(page.get_by_text('Class: 1')).to_be_visible(timeout=100_000)
    expect(page.locator('img:not([alt="Colormap"])')).to_have_count(1, timeout=300_000)
