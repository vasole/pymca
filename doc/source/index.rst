.. PyMca5 documentation master file, created by
   sphinx-quickstart on Mon Dec  9 19:27:24 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyMca's documentation!
=================================

Contents:

.. toctree::
   :maxdepth: 2

   modules

PyMca
=====

PyMca is a collection of Python tools to assist on common data analysis problems. When first released (in 2004), its main motivation was X-Ray Fluorescence (XRF) Analysis, field for which is among the most complete solutions available. 

Synchotron radiation XRF is closely associated to microscopy. To properly achieve its objectives, PyMca had to incorporate more than just 1D visualization and XRF spectrum modelling. PyMca has evolved into a library and set of tools to provide close-to-the-source data visualization and diagnostic capabilities.

Features
--------

- State-of-the-art X-Ray Fluorescence Analysis (Quantification, Mapping, ...)
- Support of multiple data formats 
- 1D, 2D, 3D and 4D imaging capabilities
- Extendible via plugins.
- Large dataset imaging (XRF, Powder diffraction, XAS, FT-IR, Raman, ...)
- Multivariate analysis.
- Common data reduction operation (normalization, fitting, ...)

Installation
------------

It can be installed from source via the usual "python setup.py install" approach (see the README file associated to the source code for details).

Official releases and ready-to-use binaries can be downloaded from http://www.sourceforge.net/projects/pymca

Contribute
----------

- Issue Tracker: github.com/vasole/pymca/issues
- Source Code: github.com/vasole/pymca

Support
-------

If you are having issues, please let us know.

The associated mailing list is: pymca-users@lists.sourceforge.net
Subscription URL: http://sourceforge.net/p/pymca/mailman/pymca-users/

License
-------

PyMca itself is licensed under the MIT license.

Please note that if you use the provided graphical user interfaces (GUI) or other libraries not supplied with PyMca, you can be conditioned by their licenses. For instance, if you use PySide (LGPL license) as widget library, you can safely use PyMca even in close source projects. If you use PyQt (GPL license or commercial license) instead of PySide, you will not be able PyMca in closed source projects unless you own a commercial license of PyQt.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

