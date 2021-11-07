Tutorials and Exercises
=======================

.. toctree::
   :hidden:

   ./xrf/material-definition/index.rst
   ./xrf/strip-background/index.rst
   ./customization/index.rst
   ./hdf5/index.rst
   ./training/quantification/index.rst
   ./training/tertiary/index.rst
   ./training/matrix/index.rst
   ./training/xraydata/index.rst

Things learned by practice usually require a greater effort than just reading or listening and tend to be better retained. Therefore we have prepared some `Exercises`_ to complement the usual set of `Tutorials`_ teaching different aspects of *PyMca*.

Their combination should provide a good starting point to use the program.

Tutorials
---------

The `Getting Started tutorial <http://ftp.esrf.fr/pub/scisoft/pymca/PyMcaCHESS.pdf>`_
is a very old tutorial written by Darren Dale and initially tailored to `CHESS <http://www.chess.cornell.edu>`_
users but usefull to everybody starting to use *PyMca*.

`Calibration tutorial <http://ftp.esrf.fr/pub/scisoft/pymca/calibrationtutorial.htm>`_.
To be used if you still have some doubts about how to calibrate your spectra.

:doc:`./xrf/material-definition/index`. This tutorial will show you how to define your own materials.

:doc:`./xrf/strip-background/index`. Description of the parameters defining your favorite background.

`ROI Imaging tutorial <http://ftp.esrf.fr/pub/scisoft/pymca/roitooldemo.htm>`_ .
Introduction to the stack imaging capabilities of *PyMca*

`Kinetics tutorial <http://ftp.esrf.fr/pub/scisoft/pymca/kineticstutorial.htm>`_ .
Illustration of the use of the ROI Imaging tool for kinetics studies.

:doc:`./hdf5/index` *PyMca* can deal with
HDF5 files since version 4.4.0. You should take a look at the
`HDF Group web site <https://portal.hdfgroup.org/display/HDF5/HDF5>`_ to know more about HDF.
`NeXus <http://www.nexusformat.org>`_ files are only supported when using the HDF5 backend.

:doc:`./customization/index` Description about how to provide customized settings and add-ons to *PyMca*.

Exercises
---------

:doc:`./training/quantification/index`. The classical exercise to learn how to carry out XRF analysis with *PyMca*.

:doc:`./training/tertiary/index`. Press-button exercise to show how to deal with secondary and higher order excitations in X-ray fluorescence quantification problems.

:doc:`./training/matrix/index`. This exercise shows the user one way to tell the program how to automatically update the sample composition.

:doc:`./training/xraydata/index`. Exercise to teach the user how to modify the theoretical data used by *PyMca*.
