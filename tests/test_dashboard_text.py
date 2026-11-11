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


def test_text_page(page: Page):
    """Test performance of text page."""
    page.set_viewport_size({"width": 1920, "height": 1080})

    open_page(page, '/Text')
    wait_streamlit_ready(page)
    expect(page).to_have_title('Text')
    # Movie sentiment example
    page.locator("label").filter(
        has_text="Use an example").locator("div").nth(1).click()
    page.get_by_text("Movie sentiment").click()
    expect(page.get_by_text("Select a method to continue")).to_be_visible(
        timeout=50_000)

    wait_until_idle(page)
    page.locator('label').filter(has_text='RISE').click()
    page.locator('label').filter(has_text='LIME').click()

    page.get_by_label("Number of top classes to show").fill("2")
    page.get_by_label("Number of top classes to show").press("Enter")
    page.get_by_text('Running...').wait_for(state='detached', timeout=200_000)

    for selector in (
            page.get_by_role('heading', name='RISE').get_by_text('RISE'),
            page.get_by_role('heading', name='LIME').get_by_text('LIME'),
            # Images for positive (RISE/LIME)
            page.get_by_role('heading',
                             name='positive').get_by_text('positive'),
            page.get_by_role('img').first,
            page.get_by_role('img').nth(1),
            # Images for negative (RISE/LIME)
            page.get_by_role('heading',
                             name='negative').get_by_text('negative'),
            page.get_by_role('img').nth(2),
            page.get_by_role('img').nth(3),
    ):
        expect(selector).to_be_visible(timeout=100_000)

    # Own data option
    page.locator("label").filter(
        has_text="Use your own data").locator("div").nth(1).click()
    selector = page.get_by_text(
        'Add your input data in the left panel to continue')
    expect(selector).to_be_visible(timeout=30_000)
    # Check input panel
    expect(page.get_by_label("Input string")).to_be_visible(timeout=200_000)
    page.get_by_label("Select model").click()
    page.get_by_label("Select labels").click()
