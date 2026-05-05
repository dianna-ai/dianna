"""Helpers for dashboard Playwright tests."""
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import expect


def wait_streamlit_ready(page: Page, timeout: int = 200_000) -> None:
    """Wait until the Streamlit app has finished loading and is ready.

    Waits for networkidle, for any "Running..." indicator to disappear,
    and for the main input-type prompt to become visible.
    """
    page.wait_for_load_state("networkidle", timeout=timeout)

    running = page.get_by_text("Running...").last
    try:
        running.wait_for(state="hidden", timeout=timeout)
    except PlaywrightTimeoutError:
        pass

    expect(page.get_by_text("Select which input type to")).to_be_visible(
        timeout=timeout)
