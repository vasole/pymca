#
# These Python module have been developed by V.A. Sole, from the European
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

# A cx_freeze setup script to create PyMca executables
#
# Use "python cx_setup.py install"
#
# It expects a properly configured compiler.
#
# Under windows you may need to set MINGW = True (untested) if you are
# not using VS2003 (python 2.5) or VS2008 (python > 2.5)
#
# If everything works well one should find a directory in the build
# directory that contains the files needed to run the PyMca without Python
#
from cx_Freeze import setup, Executable, version
if not version.startswith('4'):
    raise RuntimeError("At this point only cx_Freeze 4.x.x supported")
import sys
if sys.version_info >= (3,):
    raise RuntimeError("At this point only Python 2 supported")
import os
import glob
import cx_Freeze.hooks as _hooks

MINGW = False
DEBUG = False

def load_PyQt4_Qt(finder, module):
    """the PyQt4.Qt module is an extension module which imports a number of
       other modules and injects their namespace into its own. It seems a
       foolish way of doing things but perhaps there is some hidden advantage
       to this technique over pure Python; ignore the absence of some of
       the modules since not every installation includes all of them."""
    try:
        #This modules does not seem to be always present
        finder.IncludeModule("PyQt4._qt")
    except ImportError:
        pass
    finder.IncludeModule("PyQt4.QtCore")
    finder.IncludeModule("PyQt4.QtGui")
    finder.IncludeModule("sip")
    for name in ("PyQt4.QtSvg", "PyQt4.Qsci", "PyQt4.QtAssistant",
            "PyQt4.QtNetwork", "PyQt4.QtOpenGL", "PyQt4.QtScript",
            "PyQt4.QtSql", "PyQt4.QtSvg", "PyQt4.QtTest", "PyQt4.QtXml"):
        try:
            finder.IncludeModule(name)
        except ImportError:
            pass

def dummy(*var, **kw):
    pass

_hooks.load_PyQt4_Qt = load_PyQt4_Qt
_hooks.copy_qt_plugins = dummy


PyMcaInstallationDir = "build"
if sys.platform != "windows":
    PyMcaDir = os.path.join(PyMcaInstallationDir, "PyMca5").replace(" ","_")
else:
    PyMcaDir =os.path.join(PyMcaInstallationDir, "PyMca5")

#make sure PyMca is freshly built
if sys.platform == 'win32':
    if MINGW:
        # MinGW compiler needs two steps
        cmd = "python setup.py build -c mingw32 --distutils"
        if os.system(cmd):
            print("Error building PyMca")
            sys.exit(1)

cmd = "python setup.py install --install-lib %s --install-data %s --distutils" % \
              (PyMcaInstallationDir, PyMcaInstallationDir)
if os.system(cmd):
    print("Error building PyMca")
    sys.exit(1)


# PyMca expected to be properly installed
include_files = []

# this is critical for Qt to find image format plugins
include_files.append(("qtconffile", "qt.conf"))

# Add the qt plugins directory
import PyQt4.Qt as qt
app = qt.QApplication([])
pluginsDir = str(qt.QLibraryInfo.location(qt.QLibraryInfo.PluginsPath))
for pluginSet in glob.glob(os.path.join(pluginsDir,'*')):
    plugin = os.path.basename(pluginSet)
    if plugin in ["imageformats"]:
        if sys.platform == 'win32':
            ext = "*dll"
        else:
            if sys.platform.startswith("darwin"):
                print("WARNING: Not ready for this platform")
            #for darwin platform I use py2app
            #this only applies to linux
            ext = "*so"
        destination = os.path.join("plugins", plugin)
        fList = glob.glob(os.path.join(pluginSet,ext))
        for f in fList:
            include_files.append((f,
                                 os.path.join(destination,os.path.basename(f))))

#I should use somehow absolute import ...
sys.path = [PyMcaInstallationDir] + sys.path[1:]
import ctypes
import OpenGL
from PyMca5 import Object3D
OBJECT3D = True

try:
    import scipy
    SCIPY = True
    if DEBUG:
        print("ADDING SciPy DOUBLES THE SIZE OF THE DOWNLOADABLE PACKAGE...")
except:
    SCIPY = False

# For the time being I leave SciPy out
SCIPY = False

import matplotlib
MATPLOTLIB = True

try:
    import pyopencl
    OPENCL = True
except:
    OPENCL = False

if sys.platform.lower().startswith("linux"):
    # no sense to freeze
    OPENCL = False

try:
    import mdp
    MDP = True
except ImportError:
    MDP = False

H5PY_SPECIAL = False
import h5py
if h5py.version.version < '1.2.0':
    includes = ['h5py._extras']
elif h5py.version.version < '1.3.0':
    includes = ['h5py._stub', 'h5py._sync', 'h5py.utils']
elif h5py.version.version < '2.0.0':
    includes = ['h5py._extras', 'h5py._stub', 'h5py.utils',
                'h5py._conv', 'h5py._proxy']
else:
    H5PY_SPECIAL = True
    includes = []

import fisx
FISX = True

#some standard encodings
includes.append('encodings.ascii')
includes.append('encodings.utf_8')
includes.append('encodings.latin_1')
import PyMca5
import hdf5plugin
import silx
import pkg_resources
SILX = True

special_modules = [os.path.dirname(PyMca5.__file__),
                   os.path.dirname(matplotlib.__file__),
                   os.path.dirname(ctypes.__file__),
                   os.path.dirname(fisx.__file__),
                   os.path.dirname(hdf5plugin.__file__),
                   os.path.dirname(silx.__file__),
                   os.path.dirname(pkg_resources.__file__)]

try:
    import tomogui
    special_modules.append(os.path.dirname(tomogui.__file__))
    import freeart
    special_modules.append(os.path.dirname(freeart.__file__))
except:
    pass


excludes = ["Tkinter", "tkinter",
            'tcl','_tkagg', 'Tkconstants',
            "scipy", "Numeric", "numarray", "PyQt5"]

try:
    import IPython
    if IPython.__version__.startswith('2'):
        # this works with IPython 2.4.1
        special_modules.append(os.path.dirname(IPython.__file__))
        includes.append("colorsys")
        import pygments
        special_modules.append(os.path.dirname(pygments.__file__))
        import zmq
        special_modules.append(os.path.dirname(zmq.__file__))
        import pygments
        #includes.append("IPython")
        #includes.append("IPython.qt")
        #includes.append("IPython.qt.console")
        #includes.append("IPython.qt.console.rich_ipython_widget")
        #includes.append("IPython.qt.inprocess")
        #includes.append("IPython.lib")
except ImportError:
    print("Console plugin not available")
    pass

if sys.version < '3.0':
    #https://bitbucket.org/anthony_tuininga/cx_freeze/issues/127/collectionssys-error
    excludes.append("collections.abc")

if H5PY_SPECIAL:
    special_modules.append(os.path.dirname(h5py.__file__))
if OPENCL:
    special_modules.append(os.path.dirname(pyopencl.__file__))
    import mako
    import cffi
    import pytools
    special_modules.append(os.path.dirname(mako.__file__))
    special_modules.append(os.path.dirname(cffi.__file__))
    special_modules.append(os.path.dirname(pytools.__file__))
    #includes.append("pytools")
    includes.append("decorator")
else:
    excludes.append("pyopencl")

if MDP:
    #mdp versions above 2.5 need special treatment
    if mdp.__version__  > '2.5':
        special_modules.append(os.path.dirname(mdp.__file__))
if SCIPY:
    special_modules.append(os.path.dirname(scipy.__file__))

if OBJECT3D:
    includes.append("logging")
    excludes.append("OpenGL")
    special_modules.append(os.path.dirname(OpenGL.__file__))

for f in special_modules:
    include_files.append((f, os.path.basename(f)))

for f in ['qt', 'qttable', 'qtcanvas', 'Qwt5']:
    excludes.append(f)

buildOptions = dict(
        compressed = True,
        include_files = include_files,
        excludes = excludes,
        includes = includes,
        #includes = ["scipy.interpolate", "scipy.linalg"]
        #optimize=2,
        #packages = packages,
        #includes = ["Object3D"],
        #path = [PyMcaDir] + sys.path
        )

install_dir = PyMcaDir + " " + PyMca5.version()
if not sys.platform.startswith('win'):
    install_dir = install_dir.replace(" ","")
if os.path.exists(install_dir):
    try:
        def dir_cleaner(directory):
            for f in glob.glob(os.path.join(directory,'*')):
                if os.path.isfile(f):
                    try:
                        os.remove(f)
                    except:
                        print("file <%s> not deleted" % f)
                if os.path.isdir(f):
                    dir_cleaner(f)
            try:
                os.rmdir(directory)
            except:
                print("directory ", directory, "not deleted")
        dir_cleaner(install_dir)
    except:
        print("Unexpected error:", sys.exc_info())
        pass

if os.path.exists('bin'):
    for f in glob.glob(os.path.join('bin','*')):
        os.remove(f)
    os.rmdir('bin')
installOptions = dict(
    install_dir= install_dir,
)


exec_list = {"PyMcaMain": os.path.join(PyMcaDir, "PyMcaGui", \
                                       "pymca", "PyMcaMain.py"),
             "PyMcaBatch": os.path.join(PyMcaDir, "PyMcaGui", \
                                        "pymca", "PyMcaBatch.py"),
             "QStackWidget":os.path.join(PyMcaDir, "PyMcaGui", \
                                         "pymca", "QStackWidget.py"),
             "PeakIdentifier":os.path.join(PyMcaDir, "PyMcaGui", \
                                        "physics", "xrf", "PeakIdentifier.py"),
             "EdfFileSimpleViewer": os.path.join(PyMcaDir, "PyMcaGui", \
                                                 "pymca", "EdfFileSimpleViewer.py"),
             "PyMcaPostBatch": os.path.join(PyMcaDir, "PyMcaGui", \
                                            "pymca", "PyMcaPostBatch.py"),
             "Mca2Edf": os.path.join(PyMcaDir, "PyMcaGui", \
                                     "pymca", "Mca2Edf.py"),
             "ElementsInfo":os.path.join(PyMcaDir, "PyMcaGui", \
                                         "physics", "xrf", "ElementsInfo.py"),
             }

for f in list(exec_list.keys()):
    executable = os.path.join(install_dir, f)
    if os.path.exists(executable):
        os.remove(executable)


executables = []
for key in exec_list:
    icon = None
    # this allows to map a different icon to each executable
    if sys.platform.startswith('win'):
        if key in ["PyMcaMain", "QStackWidget"]:
            icon = os.path.join(os.path.dirname(__file__), "icons", "PyMca.ico")
    python_module = exec_list[key]
    executables.append(Executable(python_module,
                                  icon=icon))

setup(
        name = "PyMca5",
        version = PyMca5.version(),
        description = "PyMca %s" % PyMca5.version(),
        options = dict(build_exe = buildOptions,
                       install_exe = installOptions
                       ),
        executables = executables)

if SILX:
    # silx gui._qt module needs to be patched to get rid of uic
    initFile = os.path.join(install_dir, "silx", "gui", "qt", "_qt.py")
    print("###################################################################")
    print("Patching silx file")
    print(initFile)
    print("###################################################################")
    f = open(initFile, "r")
    content = f.readlines()
    f.close()
    f = open(initFile, "w")
    for line in content:
        if ("PyQt4.uic" in line) or ("PyQt5.uic" in line):
            continue
        f.write(line)
    f.close()

if OPENCL:
    # pyopencl __init__.py needs to be patched
    initFile = os.path.join(install_dir, "pyopencl", "__init__.py")
    print("###################################################################")
    print("Patching pyopencl file")
    print(initFile)
    print("###################################################################")
    f = open(initFile, "r")
    content = f.readlines()
    f.close()
    i = 0
    i0 = 0
    for line in content:
        if "def _find_pyopencl_include_path():" in line:
            i0 = i - 1
        elif (i0 != 0) and ("}}}" in line):
            i1 = i
            break
        i += 1
    f = open(initFile, "w")
    for i in range(0, i0):
        f.write(content[i])
    txt ='\n'
    txt +='def _find_pyopencl_include_path():\n'
    txt +='     from os.path import dirname, join, realpath\n'
    txt +="     return '\"%s\"' % join(realpath(dirname(__file__)), \"cl\")"
    txt +="\n"
    txt +="\n"
    f.write(txt)
    for line in content[i1:]:
        f.write(line)
    f.close()

if not sys.platform.startswith('win'):
    #rename the executables to .exe for easier handling by the start scripts
    for f in exec_list:
        executable = os.path.join(install_dir, f)
        if os.path.exists(executable):
            os.rename(executable, executable+".exe")
        #create the start script
        text  = "#!/bin/bash\n"
        text += 'if test -e "./%s.exe"; then\n' % f
        text += '    export LD_LIBRARY_PATH=./:${LD_LIBRARY_PATH}\n'
        text += '    exec ./%s.exe $*\n' % f
        text += 'else\n'
        text += '    if test -z "${PYMCAHOME}" ; then\n'
        text += '        thisdir=`dirname $0` \n'
        text += '        export PYMCAHOME=${thisdir}\n'
        text += '    fi\n'
        text += '    export LD_LIBRARY_PATH=${PYMCAHOME}:${LD_LIBRARY_PATH}\n'
        text += '    exec ${PYMCAHOME}/%s.exe $*\n' % f
        text += 'fi\n'
        nfile = open(executable,'w')
        nfile.write(text)
        nfile.close()
        os.system("chmod 775 %s"  % executable)
        #generate the lowercase commands
        if f == "PyMcaMain":
            os.system("cp -f %s %s" % (executable, os.path.join(install_dir, 'pymca')))
        elif f == "QStackWidget":
            os.system("cp -f %s %s" % (executable, os.path.join(install_dir, 'pymcaroitool')))
        elif f == "EdfFileSimpleViewer":
            os.system("cp -f %s %s" % (executable, os.path.join(install_dir, 'edfviewer')))
        else:
            os.system("cp -f %s %s" % (executable,
                                       os.path.join(install_dir, f.lower())))
            if f == "PyMcaPostBatch":
                os.system("cp -f %s %s" % (executable, os.path.join(install_dir, 'rgbcorrelator')))


#cleanup
for f in glob.glob(os.path.join(os.path.dirname(__file__),"PyMca5", "*.pyc")):
    os.remove(f)

if not sys.platform.startswith('win'):
    #Unix binary ...
    for dirname in ['/lib','/usr/lib', '/usr/X11R6/lib/']:
        for fname0 in ["libreadline.so.4",
                       "libgthread-2.0.so.0",
                       "libglib-2.0.so.0",
                       "libpng12.so.0",
                       "libfreetype.so.6",
                       "libXrender.so.1",
                       "libXxf86vm.so.1",
                       "libfontconfig.so.1",
                       "libexpat.so.0",
                      ]:
            fname = os.path.join(dirname, fname0)
            if os.path.exists(fname):
                cmd =  "cp -f %s %s" % (fname, os.path.join(install_dir, fname0))
                os.system(cmd)
        #numpy is now compiled with libgfortran at the ESRF
        for fname in glob.glob(os.path.join(dirname, "libgfortra*")):
            if os.path.exists(fname):
                cmd = "cp -f %s %s" % (fname,
                                    os.path.join(install_dir, os.path.basename(fname)))
                os.system(cmd)

    #remove libX of the packaging system to use that of the target system
    #for fname in glob.glob(os.path.join(install_dir,"libX*")):
    #    os.remove(fname)

    #remove libfontconfig.so of the package in order to use the one in the target system
    #for fname in glob.glob(os.path.join(install_dir,"libfontconf*")):
    #    os.remove(fname)

    #end linux binary

# check if the library has been created
library = os.path.join(install_dir,"library.zip")
if not os.path.exists(library):
    print("PROBLEM")
    print("Cannot find zipped library: ")
    print(library)
    print("Please use python cx_setup.py install")
else:
    if DEBUG:
        print("NO PROBLEM")

# cleanup binary modules already added within packages
files = glob.glob(os.path.join(install_dir, "*"))
for fname in files:
    for module in ["PyMca5", "matplotlib", "fisx", "silx",
                   "h5py", "hdf5", "freeart"]:
        basename = os.path.basename(fname)
        if basename.startswith(module + "_"):
            os.remove(fname)
            print("DELETING %s" % fname)
        elif basename.startswith(module + "."):
            os.remove(fname)
            print("DELETING %s" % fname)

if os.path.exists('bin'):
    for f in glob.glob(os.path.join('bin','*')):
        os.remove(f)
        print("DELETING %s" % f)
    os.rmdir('bin')

if not SCIPY:
    for f in ["SplinePlugins.py"]:
        plugin = os.path.join(install_dir, "PyMcaPlugins", f)
        if os.path.exists(plugin):
            print("Deleting plugin %s" % plugin)
            os.remove(plugin)

nsis = os.path.join("\Program Files (x86)", "NSIS", "makensis.exe")
if sys.platform.startswith("win") and os.path.exists(nsis):
    program = "PyMca"
    version = PyMca5.version()
    frozenDir = os.path.join(".", "build", "PyMca5 %s" % version) 
    # check if we can perform the packaging
    outFile = "nsisscript.nsi"
    f = open("nsisscript.nsi.in", "rb")
    content = f.readlines()
    f.close()
    if os.path.exists(outFile):
        os.remove(outFile)
    pymcaexe = "pymca%s-win64.exe" % version
    if os.path.exists(pymcaexe):
        os.remove(pymcaexe)
    f = open(outFile, "wb")
    for line in content:
        if "__VERSION__" in line:
            line = line.replace("__VERSION__", version)
        if "__PROGRAM__" in line:
            line = line.replace("__PROGRAM__", program)
        if "__OUTFILE__" in line:
            line = line.replace("__OUTFILE__", pymcaexe)
        if "__SOURCE_DIRECTORY__" in line:
            line = line.replace("__SOURCE_DIRECTORY__", frozenDir)
        f.write(line)
    f.close()
    cmd = '"%s" %s' % (nsis, outFile)
    print(cmd)
    os.system(cmd)
sys.exit(0)

########################## WHAT FOLLOWS IS UNUSED CODE ################
#                                                                     #
# I was using it to add undetected modules to library.zip             #
#                                                                     #
# It might be needed in the future. Code kept here as "backup".       #
#                                                                     #
#######################################################################

#Add modules to library.zip
import zipfile
zf = zipfile.ZipFile(library, mode='a')
PY_COUNTER = 0
PYC_COUNTER = 0
SKIP_PYC = False
SKIP_PY = True
def addToZip(zf, path, zippath, full=False):
    global PY_COUNTER
    global PYC_COUNTER
    if os.path.basename(path).upper().endswith("HTML"):
        if DEBUG:
            print("NOT ADDING", path)
        if not full:
            return
    if path.upper().endswith(".PYC"):
        PYC_COUNTER += 1
        if SKIP_PYC:
            if not full:
                if DEBUG:
                    print("NOT ADDING", path)
                return
    if path.upper().endswith(".PY"):
        PY_COUNTER += 1
        if SKIP_PY:
            if not full:
                if DEBUG:
                    print("NOT ADDING", path)
                return
    if os.path.isfile(path):
        zf.write(path, zippath, zipfile.ZIP_DEFLATED)
    elif os.path.isdir(path):
        if not os.path.basename(path).upper().startswith("PLUGIN"):
            for nm in os.listdir(path):
                addToZip(zf,
                    os.path.join(path, nm), os.path.join(zippath, nm))
        else:
            if DEBUG:
                print("NOT ADDING", path)

addToZip(zf, PyMcaDir, os.path.basename(PyMcaDir), full=False)

#if PY_COUNTER > PYC_COUNTER:
#    print "WARNING: More .py files than  .pyc files. Check cx_setup.py"
if PY_COUNTER < PYC_COUNTER:
    print("WARNING: More .pyc files than  .py files. Check cx_setup.py")


