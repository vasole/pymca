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

PyMcaInstallationDir = "build"
PyMcaDir =os.path.join(PyMcaInstallationDir, "PyMca")
#make sure PyMca is freshly built
cmd = "python setup.py install --install-lib %s --install-scripts /tmp" % PyMcaInstallationDir
if os.system(cmd):
    print "Error building PyMca"
    sys.exit(1)

pymcapath = PyMcaDir
application=os.path.join(pymcapath, "PyMcaMain.py")

#The options below are equivalent to running from the command line
#python py2app_setup.py py2app --packages=matplotlib,ctypes,h5py,Object3D
#probably matplotlib and PyOpenGL are properly detected by py2app
PACKAGES = ['h5py','OpenGL','ctypes','matplotlib','logging','Object3D','PyMcaPlugins']
try:
    import mdp
    PACKAGES.append('mdp')
except:
    pass
setup(
    app=[application],
    options={'py2app':{'packages':PACKAGES}}
)

PY_DIR="python%s"  % sys.version[0:3]
#add the data files
os.system("cp -Rf ./PyMca/attdata ./dist/PyMcaMain.app/Contents/Resources/lib/%s" % PY_DIR)
os.system("cp -Rf ./PyMca/HTML ./dist/PyMcaMain.app/Contents/Resources/lib/%s" % PY_DIR)
os.system("cp -f ./PyMca/Scofield1973.dict ./dist/PyMcaMain.app/Contents/Resources/lib/%s" % PY_DIR)
os.system("cp -f ./PyMca/*ShellRates*.dat ./dist/PyMcaMain.app/Contents/Resources/lib/%s" % PY_DIR)
os.system("cp -f ./PyMca/*ShellConstants*.dat ./dist/PyMcaMain.app/Contents/Resources/lib/%s" % PY_DIR)
os.system("cp -f ./PyMca/E*.dat ./dist/PyMcaMain.app/Contents/Resources/lib/%s" % PY_DIR)
os.system("cp -f ./PyMca/BindingEnergies.dat ./dist/PyMcaMain.app/Contents/Resources/lib/%s" % PY_DIR)
os.system("cp -f ./PyMca/McaTheory.cfg ./dist/PyMcaMain.app/Contents/Resources/lib/%s" % PY_DIR)
os.system("cp -f ./PyMca/changelog.txt ./dist/PyMcaMain.app/Contents/Resources")
os.system("cp -f ./PyMca/PyMcaSplashImage.png ./dist/PyMcaMain.app/Contents/Resources")

#I should add here a command line call to Platypus ...
