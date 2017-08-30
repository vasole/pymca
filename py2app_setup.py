#
# These Python modules have been developed by V.A. Sole, from the European
# Synchrotron Radiation Facility (ESRF) to build a frozen version of PyMca.
# Given the nature of this work, these module can be considered public domain.
# Therefore redistribution and use in source and binary forms, with or without
# modification, are permitted provided the following disclaimer is accepted:
#
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) AND THE ESRF ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHOR(S) AND/OR THE ESRF BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""
Script for building the PyMca bundle.
For a distributable application Platypus is needed

Usage:
    python py2app_setup.py py2app
"""
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import py2app
import os
import sys

#force a clean build
os.system("/bin/rm -rf dist")
os.system("/bin/rm -rf build")
os.system("/bin/rm -rf *.pyc")

BUNDLE_ICON = os.path.join(os.path.abspath('icons'), 'PyMca.icns')

#obtain the current PyMca version from the source file
ffile = open(os.path.join('PyMca5', '__init__.py'), 'r').readlines()
for line in ffile:
    if line.startswith('__version__'):
        #remove spaces and split
        __version__ = "%s" % line.replace(' ','').split("=")[-1][:-1]
        #remove " or ' present
        __version__ = __version__[1:-1]
        break

PyMcaInstallationDir = os.path.abspath("build")
PyMcaDir = os.path.join(PyMcaInstallationDir, "PyMca5")
#make sure PyMca is freshly built
cmd = "setup.py install --distutils --install-lib %s --install-scripts /tmp" % PyMcaInstallationDir
cmd = sys.executable + " " + cmd
if os.system(cmd):
    print("Error building PyMca")
    sys.exit(1)

# awful workaround because py2app picks PyMca form the source directory
os.chdir(PyMcaInstallationDir)
sys.path.insert(0, PyMcaInstallationDir)
pymcapath = PyMcaDir
application=os.path.join(pymcapath, 'PyMcaGui','pymca', "PyMcaMain.py")

#The options below are equivalent to running from the command line
#python py2app_setup.py py2app --packages=matplotlib,ctypes,h5py,Object3D
#probably matplotlib and PyOpenGL are properly detected by py2app
PACKAGES = ['fisx', 'OpenGL','ctypes','matplotlib', 'h5py','hdf5plugin','logging', 'PyMca5']
try:
    import PyQt4.Qt
except ImportError:
    print("Using PyQt5")
    PACKAGES.append("PyQt5")

try:
    import silx
    PACKAGES.append("silx")
except ImportError:
    print("silx not present")

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
if "PyQt5" in PACKAGES:
    PY2APP_OPTIONS['qt_plugins'] = ["cocoa"]

if os.path.exists(BUNDLE_ICON):
    PY2APP_OPTIONS['iconfile'] = BUNDLE_ICON
else:
    BUNDLE_ICON = None
PY2APP_OPTIONS["excludes"] = ["scipy"]

if sys.version.startswith("2"):
    PY2APP_OPTIONS["excludes"].append("PyQt4.uic.port_v3")
    PY2APP_OPTIONS["excludes"].append("PyQt5.uic.port_v3")

setup(
    app=[application],
    options={'py2app':PY2APP_OPTIONS}
)

# move to the proper place
os.system("mv -f ./dist ../dist")
os.chdir(os.path.dirname(PyMcaInstallationDir))

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
            '-f',
            os.path.join(os.getcwd(),'dist', 'PyMcaMain.app')]
    if BUNDLE_ICON is not None:
        args.append('-i')
        args.append(BUNDLE_ICON)
    args.append(os.path.join(os.getcwd(), 'PlatypusScript'))
    process = subprocess.call(args)
    py2app_bundle = os.path.join(os.getcwd(), 'PyMca%s.app' % __version__, "Contents", "Resources", "PyMcaMain.app")
    if not os.path.exists(py2app_bundle):
        print("Forcing copy")
        os.system("cp -R ./dist/PyMcaMain.app "+ os.path.dirname(py2app_bundle)) 
