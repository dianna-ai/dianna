"""Assemble the static site in web/_site: the Streamlit dashboard (dianna/dashboard) itself, run in the browser.

The browser runs it with stlite (Streamlit on Pyodide).

Run with `python web/build.py`, from a Python with the dashboard extras (`pip install .[dashboard]`).
Preview with: python -m http.server -d web/_site
"""
import glob
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path
import onnx
import seaborn
from dianna.utils.downloader import download

os.chdir(Path(__file__).resolve().parent.parent)
site = 'web/_site'
app = f'{site}/app'  # becomes app.zip, unpacked in the browser's home directory
groups = tomllib.load(open('pyproject.toml', 'rb'))['dependency-groups']


def fetch(name, version, packagetype):
    """Download a release file of a package from PyPI, checking its hash; None if there is no such file."""
    release = json.load(
        urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json'))
    file = next(
        (u for u in release['urls'] if u['packagetype'] == packagetype), None)
    if file is None:
        return None
    data = urllib.request.urlopen(file['url']).read()
    if hashlib.sha256(data).hexdigest() != file['digests']['sha256']:
        raise RuntimeError(f'hash mismatch for {file["url"]}')
    return data


shutil.rmtree(site, ignore_errors=True)
os.makedirs(app)
shutil.copy('web/index.html', site)

# The dianna package as tracked in git, without the model and data files the dashboard does not use
for path in subprocess.run(['git', 'ls-files', 'dianna'],
                           capture_output=True,
                           text=True,
                           check=True).stdout.split():
    if not re.fullmatch(r'dianna/models/.*\.onnx|.*/binary-mnist\.npz', path):
        os.makedirs(os.path.dirname(f'{app}/{path}'), exist_ok=True)
        shutil.copy(path, f'{app}/{path}')
os.makedirs(f'{app}/dianna/models', exist_ok=True
            )  # files('dianna.models') needs the (namespace package) directory
shutil.copytree('web/shims',
                app,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__'))
os.makedirs(f'{app}/.streamlit')
shutil.copy('dianna/dashboard/.streamlit/config.toml', f'{app}/.streamlit/')

# The `web-vendored` dependency group: packages micropip cannot install. Unpack their Python sources, from a wheel
# (dropping compiled extensions) or else from the sdist.
for requirement in groups['web-vendored']:
    name, version = requirement.split('==')
    package_dir = f'{name}/'
    if data := fetch(name, version, 'bdist_wheel'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(app, [
                n for n in z.namelist()
                if n.startswith(package_dir) and not n.endswith(('.so',
                                                                 '.pyd'))
            ])
    else:
        with tarfile.open(
                fileobj=io.BytesIO(fetch(name, version, 'sdist'))) as t:
            prefix = f'{t.getmembers()[0].name.split("/")[0]}/'
            members = [
                m for m in t.getmembers()
                if m.name.startswith(prefix + package_dir)
            ]
            for m in members:
                m.name = m.name[len(prefix):]
            t.extractall(app, members, filter='data')
shutil.move(f'{app}/shap_cutils.py', f'{app}/shap/_cutils.py')
shutil.copy(onnx.onnx_ml_pb2.__file__, f'{app}/onnx/')

# Pre-fill the download caches, so the dashboard finds its example files without fetching them (Zenodo and GitHub
# do not allow cross-origin downloads). The files are whatever the dashboard pages download().
os.makedirs(f'{app}/.cache/dianna')
for page in glob.glob('dianna/dashboard/**/*.py', recursive=True):
    for name, kind in re.findall(r"download\(\s*'([^']+)',\s*'(\w+)'",
                                 open(page).read()):
        shutil.copy(download(name, kind), f'{app}/.cache/dianna/')
seaborn.load_dataset('penguins')
shutil.copytree(seaborn.utils.get_data_home(), f'{app}/seaborn-data')

shutil.make_archive(app, 'zip', app)
shutil.rmtree(app)

# What index.html installs in Pyodide: the `web` dependency group
with open(f'{site}/requirements.json', 'w') as f:
    json.dump(groups['web'], f)

# stlite shows a custom component by copying its index.html into an iframe (srcdoc) and then inlining its scripts and
# stylesheets. Until then, the browser resolves the component's relative URLs against the page, so serve the
# components' build files (all but index.html) next to it. The file names are content hashes, so they must come from
# exactly the versions that are installed. (What the inlined stylesheets reference is handled in shims/web_setup.py.)
for requirement in groups['web']:
    match = re.fullmatch(r'(streamlit-[\w-]+)==([\w.]+)', requirement)
    if not match:
        continue
    with zipfile.ZipFile(io.BytesIO(fetch(*match.groups(),
                                          'bdist_wheel'))) as z:
        index = next(n for n in z.namelist()
                     if n.endswith('/index.html') and '/frontend/' in n)
        build_dir = index[:-len('index.html')]
        for n in z.namelist():
            if n.startswith(build_dir) and n != index and not n.endswith(
                ('/', '.map')):
                os.makedirs(os.path.dirname(f'{site}/{n[len(build_dir):]}'),
                            exist_ok=True)
                with open(f'{site}/{n[len(build_dir):]}', 'wb') as f:
                    f.write(z.read(n))
