#encoding:latin-1
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
      

