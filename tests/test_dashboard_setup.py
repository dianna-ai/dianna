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

pytestmark = [pytest.mark.dashboard, pytest.mark.usefixtures('dashboard')]


def test_page_load(page: Page):
    """Test performance of landing page."""
    open_page(page)

    selector = page.get_by_text('Running...')
    selector.wait_for(state='detached')

    # long timeout: the browser version (--web) first has to load Python
    expect(page.get_by_role("heading", name="Dashboard")).to_be_visible(timeout=200_000)
    expect(page).to_have_title("Dianna's dashboard")
    expect(page.get_by_role("img").first).to_be_visible()
