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

#force a clean build
os.system("/bin/rm -rf dist")
os.system("/bin/rm -rf build")
os.system("/bin/rm -rf *.pyc")


#obtain the current PyMca version from the source file
ffile = open(os.path.join('PyMca', 'PyMcaMain.py'), 'r').readlines()
for line in ffile:
    if line.startswith('__version__'):
        #remove spaces and split
        __version__ = "%s" % line.replace(' ','').split("=")[-1][:-1]
        #remove " or ' present
        __version__ = __version__[1:-1]
        break

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
try:
    import pyopencl
    PACKAGES.append('pyopencl')
except:
    pass
PY2APP_OPTIONS = {'packages':PACKAGES}
BUNDLE_ICON = '/Users/sole/Installation/PythonSnake.icns' 
if os.path.exists(BUNDLE_ICON):
    PY2APP_OPTIONS['iconfile'] = BUNDLE_ICON
else:
    BUNDLE_ICON = None
setup(
    app=[application],
    options={'py2app':PY2APP_OPTIONS}
)

# move to the proper place
os.system("mv -f ./dist ../dist")

#Command line call to Platypus ...
platypusFile = '/usr/local/bin/platypus'
if os.path.exists(platypusFile):
    import subprocess
    args = [platypusFile,
            '-R',
            '-a',
            'PyMca%s' % __version__,
            '-o',
            'Progress Bar',
            '-p',
            '/bin/bash',
            '-V',
            '%s' % __version__,
            '-I',
            'ESRF.sole.PyMca%s' % __version__,
            '-y', #force overwrite
            '-f'
            '/Users/sole/svnsourceforge/pymca/dist/PyMcaMain.app']
    if BUNDLE_ICON is not None:
        args.append('-i')
        args.append(BUNDLE_ICON)
    args.append('/Users/sole/svnsourceforge/pymca/PlatypusScript')
    process = subprocess.call(args)
