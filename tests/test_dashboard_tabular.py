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
Then, set `LOCAL = True` in tests/dashboard_helpers.py to connect to that instance for
debugging. Then, you can run the tests with:

```bash
pytest -v -m dashboard --dashboard
```
Add `--web` to test the browser version in web/_site instead (build it with `python web/build.py`):
```bash
pytest -v -m dashboard --dashboard --web
```
See more documentation about dashboard in: dianna/dashboard/readme.md

For Code generation (https://playwright.dev/python/docs/codegen):

    playwright codegen http://localhost:8501
"""

import pytest
from playwright.sync_api import Page
from playwright.sync_api import expect
from tests.dashboard_helpers import open_page
from tests.dashboard_helpers import wait_streamlit_ready
from tests.dashboard_helpers import wait_until_idle

pytestmark = [pytest.mark.dashboard, pytest.mark.usefixtures('dashboard')]


def test_tabular_page(page: Page):
    """Test performance of tabular page."""
    page.set_viewport_size({"width": 1920, "height": 1080})

    open_page(page, '/Tabular')

    wait_streamlit_ready(page)

    expect(page).to_have_title('Tabular')

    # Test using your own data
    page.locator("label").filter(
        has_text="Use your own data").locator("div").nth(1).click()

    page.get_by_label("Select tabular data").click()
    page.get_by_label("Select model").click()
    page.get_by_label("Select training data").click()
    page.get_by_label("Select labels in case of").click()


def test_tabular_sunshine(page: Page):
    """Test tabular sunshine example."""
    page.set_viewport_size({"width": 1920, "height": 1080})

    open_page(page, '/Tabular')

    wait_streamlit_ready(page)

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

    wait_until_idle(page)

    page.locator("label").filter(has_text="RISE").click()
    page.locator("label").filter(has_text="LIME").click()
    page.locator("label").filter(has_text="KernelSHAP").click()
    expander = page.locator("summary").filter(has_text="Click to modify RISE")
    expander.wait_for(state="visible", timeout=200_000)
    expander.click()

    expect(page.get_by_text("Select the input data by")).to_be_visible(timeout=100_000)
    page.locator("iframe[title*='agGrid']").scroll_into_view_if_needed()
    frame = page.frame_locator("iframe[title*='agGrid']")
    first_row = frame.locator(".ag-row").first
    expect(first_row).to_be_visible(timeout=300_000)
    first_row.click()
    page.get_by_text('Running...').wait_for(state='detached', timeout=200_000)

    expect(page.get_by_test_id('stMetricValue')).to_be_visible(timeout=200_000)

    for selector in (
            page.get_by_role('heading', name='RISE').get_by_text('RISE'),
            page.get_by_role('heading', name='KernelSHAP').get_by_text('KernelSHAP'),
            page.get_by_role('heading', name='LIME').get_by_text('LIME'),
            page.get_by_role('img').first,
            page.get_by_role('img').nth(1),
            page.get_by_role('img').nth(2),
    ):
        expect(selector).to_be_visible(timeout=100_000)


def test_tabular_penguin(page: Page):
    """Test performance of tabular penguin example."""
    page.set_viewport_size({"width": 1920, "height": 1080})

    open_page(page, '/Tabular')

    wait_streamlit_ready(page)

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

    wait_until_idle(page)

    page.locator("label").filter(has_text="RISE").click(timeout=300_000)
    page.locator("label").filter(has_text="LIME").click(timeout=300_000)
    page.locator("label").filter(has_text="KernelSHAP").click(timeout=300_000)

    expect(page.get_by_text("Select the input data by")).to_be_visible(timeout=300_000)
    page.locator("iframe[title*='agGrid']").scroll_into_view_if_needed()
    frame = page.frame_locator("iframe[title*='agGrid']")
    first_row = frame.locator(".ag-row").first
    expect(first_row).to_be_visible(timeout=300_000)
    first_row.click()
    page.get_by_text('Running...').wait_for(state='detached', timeout=300_000)

    for selector in (
        page.get_by_test_id('stMetricValue'),
        page.get_by_role('heading', name='RISE').get_by_text('RISE'),
        page.get_by_role('heading', name='KernelSHAP').get_by_text('KernelSHAP'),
        page.get_by_role('heading', name='LIME').get_by_text('LIME'),
        page.get_by_role('img').first,
        page.get_by_role('img').nth(1),
        page.get_by_role('img').nth(2),
    ):
        expect(selector).to_be_visible(timeout=200_000)
