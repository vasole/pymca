print("Please update your plugins")
print("Use from PyMca5 import Plugin1DBase")
try:
    from PyMca5.PyMcaCore.Plugin1DBase import *
except:
    from PyMca5.Plugin1DBase import *
