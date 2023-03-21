************************************************
Developer documentation for the DIANNA project
************************************************

This chapter lists the tools and practices we typically use during development.
Most of our choices are motivated in the `NL eScience Center Guide <https://guide.esciencecenter.nl>`__, specifically the `Python chapter <https://guide.esciencecenter.nl/#/best_practices/language_guides/python>`__ and our `software guide checklist <https://guide.esciencecenter.nl/#/best_practices/checklist>`__.
If you're considering contributing to DIANNA, please have a look and be sure to let us know (e.g. via an issue) if you have any questions.


Development install
-------------------

.. code:: shell

   # Create a virtual environment, e.g. with
   python3 -m venv env

   # activate virtual environment
   source env/bin/activate

   # make sure to have a recent version of pip and setuptools
   python3 -m pip install --upgrade pip setuptools

   # (from the project root directory)
   # install dianna as an editable package and install development dependencies
   python3 -m pip install --no-cache-dir --editable .[dev]

It's also possible to use ``conda`` for maintaining virtual environments; in that case you can still install DIANNA and dependencies with ``pip``.

Dependencies and Package management
-----------------------------------

DIANNA aims to support all Python 3 minor versions that are still
actively maintained, currently:

.. image:: https://img.shields.io/pypi/pyversions/dianna
   :target: https://pypi.python.org/project/dianna/

Add or remove Python versions based on availability of dependencies in
all versions. See `the
guide <https://guide.esciencecenter.nl/#/best_practices/language_guides/python>`__
for more information about Python versions.

When adding new dependencies, make sure to do so as follows:

-  Runtime dependencies should be added to ``setup.cfg`` in the
   ``install_requires`` list under ``[options]``.
-  Development dependencies should be added to ``setup.cfg`` in one of
   the lists under ``[options.extras_require]``.

Testing and code coverage
-------------------------

-  Tests should be put in the ``tests`` folder.
-  Take a look at the existing tests and add your own meaningful tests
   (file: ``test_my_module.py``) when you add a feature.
-  The testing framework used is `PyTest <https://pytest.org>`__
-  The project uses `GitHub action
   workflows <https://docs.github.com/en/actions>`__ to automatically
   run tests on GitHub infrastructure against multiple Python versions

   -  Workflows can be found in
      `.github/workflows <https:://github.com/dianna-ai/dianna/.github/workflows/>`__

-  `Relevant section in the
   guide <https://guide.esciencecenter.nl/#/best_practices/language_guides/python?id=testing>`__

Running the tests
~~~~~~~~~~~~~~~~~

There are two ways to run tests.

The first way requires an activated virtual environment with the
development tools installed run the following from the root directory of this repository:

.. code:: shell

   pytest -v

The second is to use ``tox``, which must be installed separately (e.g. with ``pip install tox``), but then builds the necessary virtual environments itself by simply running:

.. code:: shell

   tox

Testing with ``tox`` allows for keeping the testing environment separate from your development environment.
The development environment will typically accumulate (old) packages during development that interfere with testing; this problem is avoided by testing with ``tox``.

Running linters locally
-----------------------

For linting and import sorting we use `ruff <https://github.com/charliermarsh/ruff>`__, and to autoformat
the code we use `yapf <https://github.com/google/yapf>`__
Running the linters requires `pre-commit <https://pre-commit.com/>`__.

.. code:: shell

   # staged files only
   pre-commit

   # all files
   pre-commit run --all-files

You can enable automatic linting and code formatting with ``pre-commit``
on each commit by enabling the git hook, like so:

.. code:: shell

   pre-commit install

We also check linting errors in a GitHub Actions CI workflow.

Documentation
-------------

-  Documentation should be put in the ``docs/`` directory in the repository.
-  We use Restructured Text (reST) and Google style docstrings.

   -  `Restructured Text (reST)
      primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__
   -  `Google style docstring
      examples <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`__.

-  The documentation is set up with the ReadTheDocs Sphinx theme.

   -  Check out its `configuration
      options <https://sphinx-rtd-theme.readthedocs.io/en/latest/>`__.

-  `AutoAPI <https://sphinx-autoapi.readthedocs.io/>`__ is used to
   generate documentation for the package Python objects.
-  ``.readthedocs.yaml`` is the ReadTheDocs configuration file. When
   ReadTheDocs is building the documentation this package and its
   development dependencies are installed so the API reference can be
   rendered.
-  `Relevant section in the
   guide <https://guide.esciencecenter.nl/#/best_practices/language_guides/python?id=writingdocumentation>`__

Generating documentation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   cd docs
   make html

The documentation will be in ``docs/_build/html``

If you do not have ``make`` use

.. code:: shell

   sphinx-build -b html docs docs/_build/html

To find undocumented Python objects you can run

.. code:: shell

   cd docs
   make coverage
   cat _build/coverage/python.txt

We also check for undocumented functionality in a GitHub Actions CI workflow.

To `test
snippets <https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html>`__
in documentation run

.. code:: shell

   cd docs
   make doctest

Versioning
----------

Bumping the version across all files is done with
`bumpversion <https://github.com/c4urself/bump2version>`__, e.g.

.. code:: shell

   bumpversion major
   bumpversion minor
   bumpversion patch

Making a release
----------------

This section describes how to make a release in 4 steps:

1. Verify that the information in ``CITATION.cff`` is correct.
2. Make sure the `version has been updated <#versioning>`__.
3. Run the unit tests with ``pytest -v`` or ``tox``.
4. *If applicable:* list non-Python files that should be included in the distribution in ``MANIFEST.in``.
5. Make a `release on GitHub <https://github.com/dianna-ai/dianna/releases/new>`__.
   This will trigger the release workflow, which will build and upload DIANNA as a package to PyPI.
   It will also trigger Zenodo into making a snapshot of the repository and sticking a DOI on it.
   In this project the habit is to use the release notes that can be auto-generated by Github.

Note that the build is uploaded to both pypi and test-pypi.
If you trigger the workflow manually, it's only uploaded to test-pypi, which can be useful for testing.
