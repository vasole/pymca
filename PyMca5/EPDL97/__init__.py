"""
These modules allow to parse the Evaluated Photon Data Library files.

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
from PyMca5 import EPDL97
print(os.path.dirname(EPDL97.__file__))

"""
__author__ = "V.A. Sole - ESRF Software Group"
__version__ = '1.0'

# The parsing modules
# force the import here in order to see the available
# modules when doing from PyMca5 import EADL97
# followed by dir(EADL97) in an interactive session.
from . import EADLParser, EADLSubshells, EPDL97Parser
