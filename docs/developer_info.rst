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
   # install dianna as an editable package
   python3 -m pip install --no-cache-dir --editable .
   # install development dependencies
   python3 -m pip install --no-cache-dir --editable .[dev]

Afterwards check that the install directory is present in the ``PATH`` environment variable.
It's also possible to use `conda` for maintaining virtual environments; in that case you can still install DIANNA and dependencies with `pip`.

Running the tests
-----------------

There are two ways to run tests.

The first way requires an activated virtual environment with the
development tools installed:

.. code:: shell

   pytest -v

The second is to use `tox`, which must be installed separately (e.g. with `pip install tox`), but then builds the necessary virtual environments itself by simply running:

.. code:: shell

   tox

Testing with `tox` allows for keeping the testing environment separate from your development environment.
The development environment will typically accumulate (old) packages during development that interfere with testing; this problem is avoided by testing with `tox`.

Running linters locally
-----------------------

For linting we use
`prospector <https://pypi.org/project/prospector/>`__ and to sort
imports we use `isort <https://pycqa.github.io/isort/>`__. Running
the linters requires an activated virtual environment with the
development tools installed.

.. code:: shell

   # linter
   prospector

   # recursively check import style for the dianna module only
   isort --recursive --check-only dianna

   # recursively check import style for the dianna module only and show
   # any proposed changes as a diff
   isort --recursive --check-only --diff dianna

   # recursively fix import style for the dianna module only
   isort --recursive dianna

You can enable automatic linting with ``prospector`` and ``isort`` on
commit by enabling the git hook from ``.githooks/pre-commit``, like so:

.. code:: shell

   git config --local core.hooksPath .githooks

We also check linting errors in a GitHub Actions CI workflow.

Generating the API docs
-----------------------

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

This section describes how to make a release in 3 parts:

1. preparation
2. making a release on PyPI
3. making a release on GitHub

(1/3) Preparation
~~~~~~~~~~~~~~~~~

1. Verify that the information in ``CITATION.cff`` is correct.
2. Make sure the `version has been updated <#versioning>`__.
3. Run the unit tests with ``pytest -v`` or ``tox``.

(2/3) PyPI
~~~~~~~~~~

In a new terminal, without an activated virtual environment or an env
directory:

.. code:: shell

   # prepare a new directory
   cd $(mktemp -d --tmpdir dianna.XXXXXX)

   # fresh git clone ensures the release has the state of origin/main branch
   git clone https://github.com/dianna-ai/dianna .

   # prepare a clean virtual environment and activate it
   python3 -m venv env
   source env/bin/activate

   # make sure to have a recent version of pip and setuptools
   python3 -m pip install --upgrade pip setuptools

   # install runtime dependencies and publishing dependencies
   python3 -m pip install --no-cache-dir .
   python3 -m pip install --no-cache-dir .[publishing]

   # clean up any previously generated artefacts
   rm -rf dianna.egg-info
   rm -rf dist

   # create the source distribution and the wheel
   python3 setup.py sdist bdist_wheel

   # upload to test pypi instance (requires credentials)
   twine upload --repository-url https://test.pypi.org/legacy/ dist/*

Visit https://test.pypi.org/project/dianna and verify that your package
was uploaded successfully. Keep the terminal open, we’ll need it later.

In a new terminal, without an activated virtual environment or an env
directory:

.. code:: shell

   cd $(mktemp -d --tmpdir dianna-test.XXXXXX)

   # prepare a clean virtual environment and activate it
   python3 -m venv env
   source env/bin/activate

   # make sure to have a recent version of pip and setuptools
   pip install --upgrade pip setuptools

   # install from test pypi instance:
   python3 -m pip -v install --no-cache-dir \
   --index-url https://test.pypi.org/simple/ \
   --extra-index-url https://pypi.org/simple dianna

Check that the package works as it should when installed from pypitest.

Then upload to pypi.org with:

.. code:: shell

   # Back to the first terminal,
   # FINAL STEP: upload to PyPI (requires credentials)
   twine upload dist/*

(3/3) GitHub
~~~~~~~~~~~~

Don’t forget to also make a `release on
GitHub <https://github.com/dianna-ai/dianna/releases/new>`__. If your
repository uses the GitHub-Zenodo integration this will also trigger
Zenodo into making a snapshot of your repository and sticking a DOI on
it.
