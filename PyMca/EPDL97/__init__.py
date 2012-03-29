"""
This modules allow to parse the Evaluated Photon Data Library files.

The modules to use are:

EADLParser
EPDL97Parser

The converted files used by PyMca can be obtained using the scripts:

GenerateEADLBindingEnergies.py
GenerateEADLShellConstants.py
GenerateEADLShellNonradiativeRates.py
GenerateEADLShellRadiativeRates.py
GenerateEPDL97CrossSections.py
GenerateEPDL97TotalCrossSections.py

Those scripts can be found in your EPDL97 installation directory:

import os
from PyMca import EPDL97
print(os.path.dirname(EPDL97.__file__))

"""
import sys
import os
__author__ = "V.A. Sole - ESRF Software Group"
__version__ = '1.0'

# The parsing modules
from . import EADLParser, EADLSubshells, EPDL97Parser
