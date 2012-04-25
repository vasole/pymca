"""
Script for building the PyMca bundle.
For a distributable application Platypus is needed

Usage:
    python py2app_setup.py py2app
"""
from distutils.core import setup
import py2app
import os
import sys
os.system("/bin/rm -rf dist")
os.system("/bin/rm -rf build")
os.system("/bin/rm -rf *.pyc")

PyMcaInstallationDir = os.path.abspath("build")
PyMcaDir = os.path.join(PyMcaInstallationDir, "PyMca")
#make sure PyMca is freshly built
cmd = "python setup.py install --install-lib %s --install-scripts /tmp" % PyMcaInstallationDir
if os.system(cmd):
    print "Error building PyMca"
    sys.exit(1)

# awful workaround because py2app picks PyMca form the source directory 
os.chdir(PyMcaInstallationDir)
sys.path.insert(0, PyMcaInstallationDir)


pymcapath = PyMcaDir
application=os.path.join(pymcapath, "PyMcaMain.py")

#The options below are equivalent to running from the command line
#python py2app_setup.py py2app --packages=matplotlib,ctypes,h5py,Object3D
#probably matplotlib and PyOpenGL are properly detected by py2app
PACKAGES = ['h5py','OpenGL','ctypes','matplotlib','logging', 'PyMca']
try:
    import mdp
    PACKAGES.append('mdp')
except:
    pass

setup(
    app=[application],
    options={'py2app':{'packages':PACKAGES}}
)

# move to the proper place
os.system("mv -f ./dist ../dist")

#I should add here a command line call to Platypus ...
