#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Software Group"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
These modules allow to parse the Evaluated Photon Data Library files.

The modules to use are:

EADLParser
EPDL97Parser

The converted files used by PyMca can be obtained using the scripts:

    - GenerateEADLBindingEnergies.py
    - GenerateEADLShellConstants.py
    - GenerateEADLShellNonradiativeRates.py
    - GenerateEADLShellRadiativeRates.py
    - GenerateEPDL97CrossSections.py
    - GenerateEPDL97TotalCrossSections.py

Those scripts can be found in your EPDL97 installation directory:

.. code-block:: python

    import os
    from PyMca5 import EPDL97
    print(os.path.dirname(EPDL97.__file__))

"""
__version__ = '1.0'

# The parsing modules
# force the import here in order to see the available
# modules when doing from PyMca5 import EADL97
# followed by dir(EADL97) in an interactive session.
from . import EADLParser, EADLSubshells, EPDL97Parser
