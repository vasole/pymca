
Installation steps
==================

*PyMca* supports most operating systems and different version of the Python
programming language.

Stand-alone Executable
----------------------

Stand-alone applications (aka. frozen binaries) are supplied for Windows and MacOS. They do not require any additional dependency and can be downloaded from `here <https://sourceforge.net/projects/pymca/files/pymca/>`_. Just download the installer for your platform.


Python module
-------------

The best use of PyMca can be achieved installing PyMca as a python package inside an existing Python installation. For Windows and MacOS there are pre-compiled modules available in order to simplify the tast.

You can also install PyMca from its source code. While `numpy <http://www.numpy.org/>`_ and `fisx <https://github.com/vasole/fisx>`_ are the only mandatory dependencies for command line usage,
graphical widgets require Qt and `matplotlib <http://matplotlib.org/>`_ and management of HDF5 data files requires
`h5py <http://docs.h5py.org/en/latest/build.html>`_.

This table summarized the the support matrix of PyMca:

+------------+--------------+---------------------+
| System     | Python vers. | Qt and its bindings |
+------------+--------------+---------------------+
| `Windows`_ | 2.7, 3.5-3.6 | PyQt4.8+, PyQt5.3+  |
+------------+--------------+---------------------+
| `MacOS`_   | 2.7, 3.5-3.6 | PyQt4.8+, PyQt5.3+  |
+------------+--------------+---------------------+
| `Linux`_   | 2.7, 3.4-3.6 | PyQt4.8+, PyQt5.3+  |
+------------+--------------+---------------------+

For all platforms, you can install *PyMca5* from the source, see `Installing from source`_.


Dependencies
------------

Tools for reading and writing HDF5 files depend on:

* `h5py <http://docs.h5py.org/en/latest/build.html>`_

The GUI widgets depend on the following extra packages:

* A Qt binding: either `PyQt5, PyQt4 <https://riverbankcomputing.com/software/pyqt/intro>`_,
  `PySide <https://pypi.python.org/pypi/PySide/>`_, or `PySide2 <https://wiki.qt.io/PySide2>`_
* `matplotlib <http://matplotlib.org/>`_

The following packages are optional dependencies:

* `silx <https://github.com/silx-kit/silx>`_ for enhanced widgets 
* `qt_console <https://pypi.python.org/pypi/qtconsole>`_ for the interactive console widget.
* `PyOpenGL <http://pyopengl.sourceforge.net/>`_ for 3D and scatter plot visualization

It is expected that h5py and silx become required dependencies within short because:

- h5py will become the preferred input/output file format of PyMca
- silx provides a better widget library than the one currently supplied by PyMca
  
The complete list of dependencies with the minimal version is described in the
`requirements.txt <https://github.com/vasole/pymca/requirements.txt>`_
at the top level of the source package.

Build dependencies
++++++++++++++++++

In addition to run-time dependencies, building *PyMca* requires a C/C++ compiler,
`numpy <http://www.numpy.org/>`_ and `cython <http://cython.org>`_ (optional).

On Windows it is recommended to use Python 3.5 or later, because of using a more recent compiler.

This project uses Cython (version > 0.21) to generate C files.
Cython is now mandatory to build *PyMca* from the development branch and is only
needed when compiling binary modules.

Linux
-----

There are no frozen binaries or wheels available for linux. Nevertheless, there are strong chances that *PyMca*  is available as a native package for your distribution. 

If you need to build *PyMca* from its source code, and NumPy and fisx are not installed on your system, you need to install them first, preferably with the package manager of your system. If you cannot use the package manager of your system (which requires the root access), please refer to the Virtual Environment procedure explained in the `silx documentation <http://www.silx.org/doc/silx/latest/install.html>`_

Please refer to `Installing from source`_

.. note::

    The Debian packages `python-pymca5` and `python3-pymca5` will not install executables 
    (`pymca`, `pymcaroitool` ...). Please install the pymca package.

Windows
-------

The simple way of installing *PyMca*  on Windows is to type the following
commands in a command prompt:

.. code-block:: bash

    pip install PyMca5

.. note::
    
    This installs *PyMca* without the optional dependencies.
    Instructions on how to install dependencies are given in the
    `Installing dependencies`_ section.
    
This assumes you have Python and pip installed and configured. If you don't,
read the following sections.


Installing Python
+++++++++++++++++

Please follow the instructions suplied by the silx project http://www.silx.org/doc/silx/latest/install.html

Using pip
+++++++++

Configure your PATH environment variable to include the pip installation
directory, the same way as described for Python.

The pip installation directory will likely be ``C:\Python35\Scripts\``.

Then you will be able to use all pip commands listed in following in a command
prompt.


Installing dependencies
+++++++++++++++++++++++

All dependencies may be simply installed with pip:

.. code-block:: bash 

    pip install -r https://raw.githubusercontent.com/vasole/pymca/master/requirements.txt


Installing *PyMca*
++++++++++++++++++

Provided numpy is installed, you can install *PyMca* with:

.. code-block:: bash 

    pip install pymca

or 

.. code-block:: bash 

    pip install PyMca5


MacOS
-----

While Apple ships Python 2.7 by default on their operating systems, we recommend
using Python 3.5 or newer to ease the installation of the Qt library.

The installation of PyMca can simply be performed by:

.. code-block:: bash 

    pip install -r https://raw.githubusercontent.com/vasole/pymca/master/requirements.txt

Then install *PyMca* with:

.. code-block:: bash 

    pip install pymca

or 

.. code-block:: bash 

    pip install PyMca5

This should work without issues, as binary wheels of *PyMca* are provided on PyPI.


Installing from source
----------------------

Building *PyMca* from the source requires NumPy and fisx installed that can be
installed using:

.. code-block:: bash 

    pip install numpy
    pip install fisx


Building from source
++++++++++++++++++++

The most straightforward way is to use pip to take the sources from PyPI:

.. code-block:: bash

    pip install PyMca5 --no-binary [--user]
    

Alternatively, the source package of *PyMca* releases can be downloaded from
`the pypi project page <https://pypi.python.org/pypi/PyMca5>`_.

After downloading the `PyMca5-x.y.z.tar.gz` archive, extract its content:

.. code-block:: bash 

    tar xzvf PyMca5-x.y.z.tar.gz
    cd PyMca5-x.y.z
    pip uninstall -y silx
    pip install . [--user]
    
Alternatively, you can get the latest source code from the master branch of the
`git repository <https://github.com/vasole/pymca/silx/archive/master.zip>`_: https://github.com/vasole/pymca

Known issues
............

There are specific issues related to MacOSX. If you get this error::

  UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 1335: ordinal not in range(128)

This is related to the two environment variable LC_ALL and LANG not defined (or wrongly defined to UTF-8).
To set the environment variable, type on the command line:

.. code-block:: bash 

    export LC_ALL=en_US.UTF-8
    export LANG=en_US.UTF-8

Advanced build options
++++++++++++++++++++++

In case you want more control over the build procedure, the build command is:

.. code-block:: bash 

    python setup.py build

There are few advanced options to ``setup.py build``:

* ``--no-cython``: Prevent Cython (even if installed) to re-generate the C source code.
  Use the one provided by the development team.

It is not recommended to run the test suite of *PyMca* only after installation:

.. code-block:: bash 

    python -m PyMca5.tests.TestAll

Package the built into a wheel and install it:

.. code-block:: bash 

    python setup.py bdist_wheel
    pip install dist/PyMca5*.whl 

To build the documentation, using  `Sphinx <http://www.sphinx-doc.org/>`_:

.. code-block:: bash 

    python setup.py build build_doc


Testing
+++++++

To run the tests of an installed version of *PyMca*, from the python interpreter, run:

.. code-block:: python
    
     import PyMca5.tests
     PyMca5.tests.testAll()

To run the test suite from the command line run:

.. code-block:: bash
    
     python -m PyMca5.tests.TestAll
