# A cx_freeze setup script to create PyMca executables
#
# Use "python cx_setup.py install"
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
if sys.platform != "windows":
    PyMcaDir = os.path.join(PyMcaInstallationDir, "PyMca").replace(" ","_")
else:
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
flist = ["attdata",
         "HTML",
         "Scofield1973.dict",
         "changelog.txt",
         "McaTheory.cfg",
         "PyMcaSplashImage.png"]
for f in flist:
    include_files.append((os.path.join(PyMcaDir, f), f))

flist = glob.glob(os.path.join(PyMcaDir, "*.dat"))
for f in flist:
    include_files.append((f, os.path.basename(f)))

include_files.append(("qtconffile", "qt.conf"))
include_files.append((os.path.join("PyMca", "PyMcaPlugins"), "PyMcaPlugins"))

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

#I should use somehow absolute import ...
sys.path = [PyMcaDir] + sys.path
try:
    import ctypes
    import OpenGL
    import Object3D
    OBJECT3D = True
    try:
        #ESRF very special distribution ...
        import Object3DCTools
        import Object3DQhull
        OBJECT3DCTOOLS = True
    except:
        #fine, it is inside Object3D
        OBJECT3DCTOOLS = False
except:
    OBJECT3D = False

try:
    import scipy
    SCIPY = True
    if DEBUG:
        print "ADDING SciPy DOUBLES THE SIZE OF THE DOWNLOADABLE PACKAGE..."
except:
    SCIPY = False

#For the time being I leave it out
SCIPY = False

import PyMcaMain
import PyMcaPlugins
try:
    import matplotlib
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

try:
    import h5py
    if h5py.version.version < '1.2.0':
        includes = ['h5py._extras']
    elif h5py.version.version < '1.3.0':
        includes = ['h5py._stub', 'h5py._sync', 'h5py.utils']
    else:
        includes = ['h5py._extras', 'h5py._stub', 'h5py.utils',
                    'h5py._conv', 'h5py._proxy']
    includes.append('PyMca.phynx.utils.sorting')
except:
    includes = []
    
if OBJECT3D:
    includes.append("logging")
    excludes = ["OpenGL", "Tkinter", "PyMca.Object3D", "Object3D", "PyMcaPlugins",
                "scipy", "Numeric", "numarray"] 
    #This requieres the use of the environmental variable MATPLOTLIBDATA
    #pointing to mpl-data directory to work
    special_modules =[os.path.dirname(ctypes.__file__),
                  os.path.dirname(OpenGL.__file__),
                  os.path.dirname(Object3D.__file__)]
    if MATPLOTLIB:
        special_modules.append(os.path.dirname(matplotlib.__file__))
    if SCIPY:
        special_modules.append(os.path.dirname(scipy.__file__))
    for f in special_modules:
            include_files.append((f,os.path.basename(f)))
    if OBJECT3DCTOOLS:
        excludes.append("Object3DCTools")
        excludes.append("Object3DQhull")
        for f in [Object3DCTools.__file__,Object3DQhull.__file__]: 
            o3ddir = os.path.dirname(Object3D.__file__)
            include_files.append((f, 
                            os.path.join(os.path.basename(o3ddir), os.path.basename(f))))
else:
    excludes = ["Tkinter", "PyMcaPlugins", "scipy", "Numeric","numarray"]
    special_modules = [os.path.dirname(ctypes.__file__)]
    if MATPLOTLIB:
        special_modules.append(os.path.dirname(matplotlib.__file__))
    if SCIPY:
        special_modules.append(os.path.dirname(scipy.__file__))
    for f in special_modules:
            include_files.append((f,os.path.basename(f)))
    
for f in ['qt', 'qttable', 'qtcanvas', 'Qwt5']:
    excludes.append(f)
    

#Next line was for the plugins in frozen but now is in shared zip library
#include_files.append((PyMcaDir, "PyMca"))

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
install_dir = PyMcaDir + " " + PyMcaMain.__version__
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
                        print "file ", f,  "not deleted"
                if os.path.isdir(f):
                    dir_cleaner(f)
            try:
                os.rmdir(directory)
            except:
                print "directory ", directory, "not deleted"
        dir_cleaner(install_dir)        
    except:
        print "Unexpected error:", sys.exc_info()
        pass
        
if os.path.exists('bin'):
    for f in glob.glob(os.path.join('bin','*')):
        os.remove(f)
    os.rmdir('bin')
installOptions = dict(
    install_dir= install_dir,
)

exec_list = ["PyMcaMain",
             "PyMcaBatch",
             "QEDFStackWidget",
             "QStackWidget",
             "PeakIdentifier",
             "EdfFileSimpleViewer",
             "PyMcaPostBatch",
             "Mca2Edf",
             "ElementsInfo"]

for f in exec_list:
    executable = os.path.join(install_dir, f)
    if os.path.exists(executable):
        os.remove(executable)

         
executables = []
for python_module in exec_list:
    executables.append(Executable(os.path.join(PyMcaDir, python_module+".py")))


setup(
        name = "PyMca",
        version = PyMcaMain.__version__,
        description = "PyMca %s" % PyMcaMain.__version__,
        options = dict(build_exe = buildOptions,
                       install_exe = installOptions
                       ),
        executables = executables)

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
        elif f == "QEDFStackWidget":
            os.system("cp -f %s %s" % (executable, os.path.join(install_dir, 'pymcaroitool')))
        elif f == "EdfFileSimpleViewer":
            os.system("cp -f %s %s" % (executable, os.path.join(install_dir, 'edfviewer')))
        else:
            os.system("cp -f %s %s" % (executable,
                                       os.path.join(install_dir, f.lower())))
            if f == "PyMcaPostBatch":
                os.system("cp -f %s %s" % (executable, os.path.join(install_dir, 'rgbcorrelator')))
            

#cleanup
for f in glob.glob(os.path.join(os.path.dirname(__file__),"PyMca", "*.pyc")):
    os.remove(f)

if not sys.platform.startswith('win'):
    #Unix binary ...
    readline = 'libreadline.so.4'
    for dirname in ['/lib','/usr/lib']:
        fname = os.path.join(dirname, readline)
        if os.path.exists(fname):
            cmd =  "cp -f %s %s" % (fname, os.path.join(install_dir, readline))
            os.system(cmd)
            if dirname == '/lib':
                #readline is from suse82 systems at ESRF
                #and at a certain point I had to add these two libraries
                fname = '/usr/lib/libg2c.so.0'
                cmd = "cp -f %s %s" % (fname,
                                       os.path.join(install_dir, 'libg2c.so.0'))
                os.system(cmd)
                fname = '/usr/lib/libpng.so.3'
                cmd = "cp -f %s %s" % (fname,
                                       os.path.join(install_dir, 'libpng.so.3'))
                os.system(cmd)

        #numpy is now compiled with libgfortran at the ESRF
        for fname in glob.glob(os.path.join(dirname, "libgfortra*")):
            cmd = "cp -f %s %s" % (fname,
                                    os.path.join(install_dir, os.path.basename(fname)))
            os.system(cmd)

    #remove libX of the packaging system to use that of the target system
    for fname in glob.glob(os.path.join(install_dir,"libX*")):
        os.remove(fname)

    #remove libfontconfig.so of the package in order to use the one in the target system
    for fname in glob.glob(os.path.join(install_dir,"libfontconf*")):
        os.remove(fname)

    #end linux binary
library = os.path.join(install_dir,"library.zip")
if not os.path.exists(library):
    print "PROBLEM"
    print "Cannot find zipped library: "
    print library
    print "Please use python cx_setup.py install"
else:
    if DEBUG:
        print "NO PROBLEM"

#Add modules to library.zip for easy access from Plugins directory
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
            print "NOT ADDING", path
        if not full:
            return
    if path.upper().endswith(".PYC"):
        PYC_COUNTER += 1
        if SKIP_PYC:
            if not full:
                if DEBUG:
                    print "NOT ADDING", path
                return
    if path.upper().endswith(".PY"):
        PY_COUNTER += 1
        if SKIP_PY:
            if not full:
                if DEBUG:
                    print "NOT ADDING", path
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
                print "NOT ADDING", path

addToZip(zf, PyMcaDir, os.path.basename(PyMcaDir), full=False)
    
#if PY_COUNTER > PYC_COUNTER:
#    print "WARNING: More .py files than  .pyc files. Check cx_setup.py"
if PY_COUNTER < PYC_COUNTER:
    print "WARNING: More .pyc files than  .py files. Check cx_setup.py"

if os.path.exists('bin'):
    for f in glob.glob(os.path.join('bin','*')):
        os.remove(f)
    os.rmdir('bin')
