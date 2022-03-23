.. _installation:

Installation
============

.. admonition:: Installation Options

	There are two options to install the library:

	1. Install from PyPI using the prebuilt wheel package (``pip install mab2rec``)
	2. Build from the source code

Requirements
------------

The library requires Python **3.7+**. The ``requirements.txt`` lists the necessary
packages.

Source Code
-----------

You can build a wheel package on your platform from scratch using the source code:

.. code-block:: python

	git clone https://github.com/fidelity/mab2rec.git
	cd mab2rec
	pip install setuptools wheel # if wheel is not installed
	python setup.py sdist bdist_wheel
	pip install dist/mab2rec-X.X.X-py3-none-any.whl

Test Your Setup
---------------

To confirm that cloning was successful, run the tests included in the project.

All tests should pass.

.. code-block:: python

	git clone https://github.com/fidelity/mab2rec.git
	cd mab2rec
	python -m unittest discover tests

Upgrade the Library
-------------------

To upgrade to the latest version of the library, run ``pip install --upgrade mab2rec``.

If you installed from the source code:

.. code-block:: python

	git pull origin master
	python setup.py sdist bdist_wheel
	pip install --upgrade --no-cache-dir dist/mab2rec-X.X.X-py3-none-any.whl
