# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Tutorials ---------------------------------------------------------------

# Generate an nblink file in docs/tutorials for every tutorial notebook, so
# adding a notebook to tutorials/ is enough to have it show up in the docs.
# Notebooks listed in _tutorial_order come first, in that order; any others
# follow alphabetically. The number prefix sets the order in the toctree.
import json  # noqa: E402
from pathlib import Path  # noqa: E402

_tutorial_order = [
    'overview.ipynb',
    'explainers/LIME/lime_images.ipynb',
    'explainers/LIME/lime_tabular_penguin.ipynb',
    'explainers/LIME/lime_timeseries_weather.ipynb',
    'explainers/KernelSHAP/kernelshap_mnist.ipynb',
    'explainers/KernelSHAP/kernelshap_geometric_shapes.ipynb',
    'explainers/KernelSHAP/kernelshap_tabular_weather.ipynb',
    'explainers/RISE/rise_text.ipynb',
    'explainers/RISE/rise_imagenet.ipynb',
    'explainers/RISE/rise_timeseries_frb.ipynb',
    'explainers/LIME/lime_text_eulaw.ipynb',
    'explainers/RISE/rise_tabular_penguin.ipynb',
]
# Notebooks that are left out of the docs.
_tutorial_exclude = [
    'conversion_onnx/keras2onnx.ipynb',
    'conversion_onnx/pytorch2onnx.ipynb',
    'conversion_onnx/skl2onnx.ipynb',
    'conversion_onnx/tensorflow2onnx.ipynb',
    'explainers/KernelSHAP/kernelshap_tabular_land_atmosphere.ipynb',
    'explainers/KernelSHAP/kernelshap_tabular_penguin.ipynb',
    'explainers/LIME/lime_tabular_weather.ipynb',
    'explainers/LIME/lime_text.ipynb',
    'explainers/LIME/lime_timeseries_coffee.ipynb',
    'explainers/RISE/rise_mnist.ipynb',
    'explainers/RISE/rise_timeseries_weather.ipynb',
]
_docs = Path(__file__).parent
_tutorials = _docs.parent / 'tutorials'
_notebooks = sorted(nb.relative_to(_tutorials).as_posix() for nb in _tutorials.rglob('*.ipynb')
                    if '.ipynb_checkpoints' not in nb.parts)
_notebooks = [nb for nb in _notebooks if nb not in _tutorial_exclude]
_notebooks = [nb for nb in _tutorial_order if nb in _notebooks] + [nb for nb in _notebooks if nb not in _tutorial_order]
(_docs / 'tutorials').mkdir(exist_ok=True)
for _nblink in (_docs / 'tutorials').glob('*.nblink'):
    _nblink.unlink()  # drop links to renamed/removed notebooks
for _i, _nb in enumerate(_notebooks):
    (_docs / 'tutorials' / f'{_i:02d}-{Path(_nb).stem}.nblink').write_text(
        json.dumps({'path': f'../../tutorials/{_nb}'}, indent=4))

# -- Project information -----------------------------------------------------

project = u'dianna'
copyright = u'2022, Netherlands eScience Center'
author = u'DIANNA Team'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '1.8.1'
# The full version, including alpha/beta/rc tags.
release = version

# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.doctest',
    'sphinx.ext.intersphinx', 'sphinx.ext.mathjax', 'sphinx.ext.napoleon',
    'sphinx.ext.todo', 'sphinx.ext.viewcode', 'autoapi.extension', 'nbsphinx',
    'nbsphinx_link', 'myst_parser'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Use autoapi.extension to run sphinx-apidoc -------

autoapi_dirs = ['../dianna']

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = 'DIANNA_Logo_blueBG.png'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# -- Options for Intersphinx

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    # Commonly used libraries, uncomment when used in package
    # 'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    # 'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
    # 'scikit-learn': ('https://scikit-learn.org/stable/', None),
    # 'matplotlib': ('https://matplotlib.org/stable/', None),
    # 'pandas': ('http://pandas.pydata.org/docs/', None),
}
