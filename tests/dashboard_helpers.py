"""Helpers for dashboard Playwright tests."""
import functools
import http.server
import os
import subprocess
import threading
import time
import urllib.request
from contextlib import contextmanager
from playwright.sync_api import Page
from playwright.sync_api import expect

LOCAL = False  # True: test a dashboard you started yourself (`streamlit run Home.py`, port 8501)

PORT = '8501' if LOCAL else '8502'
BASE_URL = f'localhost:{PORT}'
WEB_SITE = os.path.join(os.path.dirname(__file__), '..', 'web', '_site')
WEB = False  # set by run_dashboard: testing the browser version


class _WebHandler(http.server.SimpleHTTPRequestHandler):
    """Serve web/_site, with index.html for page paths like /Images (stlite routes them, like Streamlit does)."""

    def send_head(self):
        if not os.path.exists(self.translate_path(self.path)):
            self.path = '/index.html'
        return super().send_head()

    def log_message(self, *args):
        pass


@contextmanager
def run_dashboard(web: bool):
    """Run the Streamlit dashboard, or serve its browser version (web/_site) if `web`."""
    global WEB
    WEB = web
    if LOCAL:
        yield
    elif web:
        if not os.path.exists(os.path.join(WEB_SITE, 'app.zip')):
            raise FileNotFoundError('Build the browser version first: python web/build.py')
        server = http.server.ThreadingHTTPServer(('localhost', int(PORT)),
                                                 functools.partial(_WebHandler, directory=WEB_SITE))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield
        server.shutdown()
    else:
        p = subprocess.Popen(['dianna-dashboard', '--server.port', PORT, '--server.headless', 'true'])
        _wait_for_server(p, f'http://{BASE_URL}/_stcore/health')
        yield
        p.kill()


def _wait_for_server(process, url, timeout=60):
    """Wait until the server `process` answers on `url`."""
    deadline = time.time() + timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError(f'The dashboard exited (code {process.returncode}); is port {PORT} in use?')
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.2)


def open_page(page: Page, path: str = '') -> None:
    """Open a dashboard page (e.g. '/Images') in a fresh session.

    The browser version is only loaded once (~25 s), then switches pages with its remount() (~1 s).
    """
    if WEB and page.url.startswith(f'http://{BASE_URL}/'):
        page.evaluate('path => remount(path)', path or '/')
    else:
        page.goto(f'{BASE_URL}{path}')


def wait_streamlit_ready(page: Page, timeout: int = 90_000) -> None:
    """Wait until the app has loaded: it asks for the input type and is done running the script."""
    expect(page.get_by_text("Select which input type to")).to_be_visible(timeout=timeout)
    wait_until_idle(page, timeout=timeout)


def wait_until_idle(page: Page, timeout: int = 300_000) -> None:
    """Wait until Streamlit has finished running the script, so that the next input is not lost to a rerun."""
    expect(page.get_by_test_id('stStatusWidget')).to_be_hidden(timeout=timeout)
