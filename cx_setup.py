# A cx_freeze setup script to create PyMca executables
#
# Use either "python cx_setup.py build" or "python cx_setup.py install"
#
# It expects a properly configured compiler.
#
# Under windows you may need to set MINGW = True (untested) if you are
# not using VS2003 (python 2.5) or VS2008 (python 2.6)
#
# If everything works well one should find a directory in the build
# directory that contains the files needed to run the PyMca without Python
from cx_Freeze import setup, Executable
import sys
import os
import glob
import cx_Freeze.hooks as _hooks

MINGW = False

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

_hooks.load_PyQt4_Qt = load_PyQt4_Qt

if 0:
    #This does not seem to solve the OpenGL problem
    def _load_OpenGL(finder, module):
        """
        OpenGL >= 3.0.0 is ctypes and setuptools based. Therefore, a plethora of
        modules are missed by the finder. So force inclusion of them all.
        """
        import OpenGL
        version = -1
        try:
            version = int(OpenGL.version.__version__.split(".")[0])
        except:
            pass
        if version >= 3:
            basedir, sep, basemod = module.path[0].rpartition(os.sep)
        for root, dirs, files in os.walk(module.path[0]):
            package = root.replace(basedir, "", 1).strip(sep).replace(sep, ".")
        if package != "OpenGL.tests": # ignore the OpenGL.tests package
            finder.IncludePackage(package)
    _hooks.load_OpenGL = _load_OpenGL


PyMcaInstallationDir = "build"
PyMcaDir =os.path.join(PyMcaInstallationDir, "PyMca")
#make sure PyMca is freshly built
if sys.platform == 'win32':
    if MINGW:
        # MinGW compiler needs two steps
        cmd = "python setup.py build -c mingw32"
        if os.system(cmd):
            print "Error building PyMca"
            sys.exit(1)

cmd = "python setup.py install --install-lib %s" % PyMcaInstallationDir
if os.system(cmd):
    print "Error building PyMca"
    sys.exit(1)

include_files = []
for f in ["attdata", "HTML", "Scofield1973.dict", "changelog.txt", "McaTheory.cfg"]:
    include_files.append((os.path.join(PyMcaDir, f), f))

flist = glob.glob(os.path.join(PyMcaDir, "*.dat"))
for f in flist:
    include_files.append((f, os.path.basename(f)))

include_files.append(("qtconffile", "qt.conf"))
include_files.append((os.path.join("PyMca", "Plugins1D"), "Plugins1D"))

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
            #for darwin platfrom I use py2app
            #this only applies to linux
            ext = "*so"
        destination = os.path.join("plugins", plugin)
        fList = glob.glob(os.path.join(pluginSet,ext))
        for f in fList:
            include_files.append((f,
                                 os.path.join(destination,os.path.basename(f))))

try:
    import ctypes
    import OpenGL
    import Object3D
    OBJECT3D = True
except:
    OBJECT3D = False

#I should use somehow absolute import ...
sys.path = [PyMcaDir] + sys.path
import PyMcaMain
import Plugins1D

if OBJECT3D:
    excludes = ["OpenGL", "Tkinter", "Object3D", "Plugins1D"]
    for f in [os.path.dirname(ctypes.__file__),
              os.path.dirname(OpenGL.__file__),
              os.path.dirname(Object3D.__file__)]:
        include_files.append((f,os.path.basename(f)))
else:
    excludes = ["Tkinter", "Plugins1D"]



buildOptions = dict(
        compressed = True,
        include_files = include_files,
        excludes = excludes,
        #includes = ["ctypes", "OpenGL"]
        #optimize=2,
        #packages = packages,
        #includes = ["Object3D"],
        #path = [PyMcaDir] + sys.path
        )
install_dir = PyMcaDir + " " + PyMcaMain.__version__
if os.path.exists(install_dir):
    try:
        for f in glob.glob(os.path.join(install_dir, '*')):
            if os.path.isfile(f):
                os.remove(f)
            if os.path.isdir(f):
                for f2 in glob.glob(os.path.join(f, '*')):
                    if os.path.isfile(f2):
                        os.remove(f2)
                    if os.path.isdir(f2):
                        for f3 in glob.glob(os.path.join(f2, '*')):
                            if os.path.isfile(f3):
                                os.remove(f3)
                            if os.path.isdir(f3):
                                try:
                                    os.rmdir(f3)
                                except:
                                    print f3, "not deleted"
                                    pass
                    try:
                        os.rmdir(f2)
                    except:
                        print f2, "not deleted"
                        pass
                try:
                    os.rmdir(f)
                except:
                    print f, "not deleted"
                    pass
        os.rmdir(install_dir)
    except:
        pass
installOptions = dict(
    install_dir= install_dir,
)

executables = [
        Executable(os.path.join(PyMcaDir, "PyMcaMain.py")),
        Executable(os.path.join(PyMcaDir, "PyMcaBatch.py")),
        Executable(os.path.join(PyMcaDir, "QEDFStackWidget.py")),
        Executable(os.path.join(PyMcaDir, "PeakIdentifier.py")),
        Executable(os.path.join(PyMcaDir, "EdfFileSimpleViewer.py")),
        Executable(os.path.join(PyMcaDir, "PyMcaPostBatch.py")),
        Executable(os.path.join(PyMcaDir, "Mca2Edf.py")),
        Executable(os.path.join(PyMcaDir, "ElementsInfo.py")),
]

setup(
        name = "PyMca",
        version = PyMcaMain.__version__,
        description = "PyMca %s" % PyMcaMain.__version__,
        options = dict(build_exe = buildOptions,
                       install_exe = installOptions
                       ),
        executables = executables)
#cleanup
for f in glob.glob(os.path.join(os.path.dirname(__file__),"PyMca", "*.pyc")):
    os.remove(f)
