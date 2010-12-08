#encoding:latin-1
__doc__="""

Routines for nonnegative matrix approximation (nnma)

"""
import sys
try:
   from nnma import *
except ImportError:
   if sys.version > '2.6':
        eval('from .nnma import *')

