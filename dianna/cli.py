import sys

if sys.version_info < (3, 10):
    from importlib_resources import files
else:
    from importlib.resources import files


def dashboard():
    """Start streamlit dashboard."""
    from streamlit.web import cli as stcli

    args = sys.argv[1:]

    dash = files('dianna.dashboard') / 'Home.py'

    # https://docs.streamlit.io/library/advanced-features/configuration
    sys.argv = [
        *('streamlit', 'run', str(dash)),
        *('--theme.base', 'light'),
        *('--theme.primaryColor', '7030a0'),
        *('--theme.secondaryBackgroundColor', 'e4f3f9'),
        *('--browser.gatherUsageStats', 'false'),
        *args,
    ]

    sys.exit(stcli.main())
