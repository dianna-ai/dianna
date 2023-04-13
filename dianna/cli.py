import os
import sys


if sys.version_info < (3, 10):
    from importlib_resources import files
else:
    from importlib.resources import files


def dashboard():
    """Start streamlit dashboard."""
    from streamlit.web import cli as stcli

    dashboard_dir = files('dianna').parent / 'dashboard'
    os.chdir(dashboard_dir)

    # https://docs.streamlit.io/library/advanced-features/configuration
    sys.argv = [
        'streamlit',
        'run',
        'Home.py',
        '--theme.base',
        'light',
        '--theme.primaryColor',
        '7030a0',
        '--theme.secondaryBackgroundColor',
        'e4f3f9',
        '--browser.gatherUsageStats',
        'false',
    ]

    sys.exit(stcli.main())
