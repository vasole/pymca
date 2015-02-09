#encoding:latin-1
__copyright__ = "Uwe Schmitt, uschmitt@gateway.de"
__license__ = "BSD"
__doc__="""

Routines for nonnegative matrix approximation (nnma)

"""
import sys
import os
try:
   from nnma import *
except ImportError:
   if sys.version > '2.6':
      sys.path.append(os.path.dirname(__file__))
      from nnma import *


