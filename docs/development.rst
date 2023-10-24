.. _virtualenv: https://virtualenv.pypa.io
.. _pip: https://pip.pypa.io
.. _Pytest: http://pytest.org
.. _Napoleon: https://sphinxcontrib-napoleon.readthedocs.org
.. _Flake8: http://flake8.readthedocs.org
.. _Sphinx: http://www.sphinx-doc.org
.. _tox: http://tox.readthedocs.io
.. _livereload: https://livereload.readthedocs.io
.. _twine: https://twine.readthedocs.io

.. _development_intro:

===========
Development
===========

locallm is developed with:

* *Test Development Driven* (TDD) using `Pytest`_;
* Respecting flake and pip8 rules using `Flake8`_;
* `Sphinx`_ for documentation with enabled `Napoleon`_ extension (using
  *Google style*);
* `tox`_ to run tests on various environments;

Every requirements are available in package extra requirements.

.. _development_install:


System requirements
*******************

This will requires `Python`, `pip`_, `virtualenv`_, *GNU make* and some other common
system packages.

Lists below are the required basic development system packages and some other optional
ones.

.. Warning::
   Package names may differ depending your system.

* Git;
* Python (according version to the package setup);
* ``python-dev``;
* ``make``;

.. Hint::
   If your system does not have the right Python version as the default one, you should
   learn to use something like `pyenv <https://github.com/pyenv/pyenv>`_.

On Linux distribution
    You will install them from your common package manager like ``apt`` for Debian
    based distributions: ::

        apt install python-dev make

On macOS
    Recommended way is to use ``brew`` utility for system packages, some names
    can vary.

On Windows
    Windows is supported but some things may need some tricks on your own.


Deployment
**********

Once requirements are ready you can use the following commands: ::

    git clone https://github.com/emencia/locallm.git
    cd locallm
    make install


Unittests
*********

Unittests are made to works on `Pytest`_, a shortcut in Makefile is available
to start them on your current development install: ::

    make test


Tox
***

To ease development against multiple Python versions a tox configuration has
been added. You are strongly encouraged to use it to test your pull requests.

Just execute Tox: ::

    make tox

This will run tests for all configured Tox environments, it may takes some time so you
may use it only before releasing as a final check.


Documentation
*************

You can easily build the documentation from one Makefile action: ::

    make docs

There is Makefile action ``livedocs`` to serve documentation and automatically
rebuild it when you change documentation files: ::

    make livedocs

Then go on ``http://localhost:8002/`` or your server machine IP with port 8002.

.. Note::
    You need to build the documentation at least once before using  ``livedocs``.


Releasing
*********

Before releasing, you must ensure about quality, use the command below to run every
quality check tasks: ::

    make quality

If quality is correct and after you have correctly pushed all your commits
you can proceed to release: ::

    make release

This will build the package release and send it to Pypi with `twine`_.
You will have to
`configure your Pypi account <https://twine.readthedocs.io/en/latest/#configuration>`_
on your machine to avoid to input it each time.


Contribution
************

* Every new feature or changed behavior must pass all quality tasks and must be
  documented (at least docstrings);
* Every feature or behavior must be compatible for all supported environment;
