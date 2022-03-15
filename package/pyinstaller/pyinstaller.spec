# -*- mode: python -*-
import sys
import os.path
from pathlib import Path
import shutil
import subprocess
import time


from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = []

PROJECT_PATH = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
datas.append((os.path.join(PROJECT_PATH, "README.rst"), "."))
datas.append((os.path.join(PROJECT_PATH, "LICENSE"), "."))
datas.append((os.path.join(PROJECT_PATH, "copyright"), "."))
#datas += collect_data_files("silx.resources")

if sys.platform.startswith('darwin'):
   icon = os.path.join(PROJECT_PATH, "icons", "PyMca.icns")
else:
   icon = os.path.join(PROJECT_PATH, "icons", "PyMca.ico")

hiddenimports = []
hiddenimports += collect_submodules('encodings.ascii')
hiddenimports += collect_submodules('encodings.utf_8')
hiddenimports += collect_submodules('encodings.latin_1')
#hiddenimports += collect_submodules('fabio')
#hiddenimports += collect_submodules('PyQt5.uic')
#hiddenimports += collect_submodules('hdf5plugin')
#hiddenimports += collect_submodules('fisx')
#hiddenimports += collect_submodules('PyMca5.PyMcaGui.PyMcaQt')
#hiddenimports += collect_submodules('PyMca5.PyMcaGui.pymca')

excludes = ["fabio", "hdf5plugin", "silx"]


# get the script list
import PyMca5
version = PyMca5.version()
PyMcaDir = os.path.abspath(os.path.dirname(PyMca5.__file__))
exec_dict = {"PyMcaMain": os.path.join(PyMcaDir, "PyMcaGui", \
                                     "pymca", "PyMcaMain.py"),
             "PyMcaBatch": os.path.join(PyMcaDir, "PyMcaGui", \
                                   "pymca", "PyMcaBatch.py"),
             "QStackWidget":os.path.join(PyMcaDir, "PyMcaGui", \
                                    "pymca", "QStackWidget.py"),
             "PyMcaPostBatch": os.path.join(PyMcaDir, "PyMcaGui", \
                                    "pymca", "PyMcaPostBatch.py"),
            }

if not sys.platform.startswith("darwin"):
    exec_dict["PeakIdentifier"] = os.path.join(PyMcaDir, "PyMcaGui", \
                                      "physics", "xrf", "PeakIdentifier.py")
    exec_dict["EdfFileSimpleViewer"] = os.path.join(PyMcaDir, "PyMcaGui", \
                                      "pymca", "EdfFileSimpleViewer.py")
    exec_dict["Mca2Edf"] = os.path.join(PyMcaDir, "PyMcaGui", \
                                      "pymca", "Mca2Edf.py")
    exec_dict["ElementsInfo"] = os.path.join(PyMcaDir, "PyMcaGui", \
                                      "physics", "xrf", "ElementsInfo.py")

script_n = []
script_l = []
for key in exec_dict:
    script_l.append(exec_dict[key])
    script_n.append(key)

block_cipher = None

script_a = []

for i in range(len(script_l)):
    script = script_l[i]
    cwd = Path(SPECPATH)
    print("Copying %s to %s" % (script, script_n[i]))
    if os.path.exists(script_n[i]):
        os.remove(script_n[i])
    shutil.copy2(
        src = script,
        dst = os.path.join(cwd, script_n[i]),
    )
    script_a.append(Analysis(
                            [script_n[i]],
                            pathex=[],
                            binaries=[],
                            datas=datas,
                            hiddenimports=hiddenimports,
                            hookspath=[],
                            runtime_hooks=[],
                            excludes=excludes,
                            win_no_prefer_redirects=False,
                            win_private_assemblies=False,
                            cipher=block_cipher,
                            noarchive=False))
    #script_a[-1].pure = [x for x in script_a[-1].pure if not x[0].startswith("PyMca5.")]


if 0: # avoid merge
    if len(script_a) == 2:
        MERGE(
            (script_a[0], script_n[0], os.path.join(script_n[0], script_n[0])), 
            (script_a[1], script_n[1], os.path.join(script_n[1], script_n[1])),
        )
    elif len(script_a) == 3:
        MERGE(
            (script_a[0], script_n[0], os.path.join(script_n[0], script_n[0])), 
            (script_a[1], script_n[1], os.path.join(script_n[1], script_n[1])),
            (script_a[2], script_n[2], os.path.join(script_n[2], script_n[2])), 
        )
    elif len(script_a) == 4:
        MERGE(
            (script_a[0], script_n[0], os.path.join(script_n[0], script_n[0])), 
            (script_a[1], script_n[1], os.path.join(script_n[1], script_n[1])),
            (script_a[2], script_n[2], os.path.join(script_n[2], script_n[2])), 
            (script_a[3], script_n[3], os.path.join(script_n[3], script_n[3])),
        )
    elif len(script_a) >= 5:
        MERGE(
            (script_a[0], script_n[0], os.path.join(script_n[0], script_n[0])), 
            (script_a[1], script_n[1], os.path.join(script_n[1], script_n[1])),
            (script_a[2], script_n[2], os.path.join(script_n[2], script_n[2])), 
            (script_a[3], script_n[3], os.path.join(script_n[3], script_n[3])),
            (script_a[4], script_n[4], os.path.join(script_n[4], script_n[4])),
        )

script_pyz = []
script_exe = []
script_col = []
for i in range(len(script_a)):        
    script_pyz.append(PYZ(script_a[i].pure,
                          script_a[i].zipped_data,
                          cipher=block_cipher))

    script_exe.append(
        EXE(
            script_pyz[i],
            script_a[i].scripts,
            script_a[i].dependencies,
            [],
            exclude_binaries=True,
            name=script_n[i],
            debug=False,
            bootloader_ignore_signals=False,
            strip=False,
            upx=False,
            console=True,
            icon=icon)
        )
    script_col.append(
        COLLECT(
            script_exe[i],
            script_a[i].binaries,
            script_a[i].zipfiles,
            script_a[i].datas,
            strip=False,
            upx=False,
            name=script_n[i])
        )

def move_exe(name):
    dist = os.path.join(Path(SPECPATH), 'dist')
    if sys.platform.startswith("darwin"):
        shutil.copy2(
            src=os.path.join(dist, name, name),
            dst=os.path.join(dist, script_n[0])
        )
    else:
        shutil.copy2(
            src=os.path.join(dist, name, name + ".exe"),
            dst=os.path.join(dist, script_n[0])
        )
    shutil.rmtree(os.path.join(dist, name))

if len(script_a) > 1:
    for n in script_n[1:]:
        move_exe(n)


# Mandatory modules to be integrally included in the frozen version.
# One can add other modules here if not properly detected
# (PyQt5 seems to be properly handled, if not, add to special_modules)
import PyMca5
import fisx
import h5py
import numpy
import matplotlib
import ctypes
import hdf5plugin

program = "PyMca"
version = PyMca5.version()
special_modules = [os.path.dirname(PyMca5.__file__),
                   os.path.dirname(fisx.__file__),
                   os.path.dirname(h5py.__file__),
                   #os.path.dirname(numpy.__file__),
                   #os.path.dirname(ctypes.__file__),
                   os.path.dirname(hdf5plugin.__file__),
                   ]

# this block  can be needed for matplotlib
MATPLOTLIB_FROM_PYINSTALLER = True
if not MATPLOTLIB_FROM_PYINSTALLER:
    special_modules.append(os.path.dirname(matplotlib.__file__))

    # recent versions of matplotlib need packaging, PIL and pyparsing
    try:
        import packaging
        special_modules.append(os.path.dirname(packaging.__file__))
    except:
        pass
    try:
        import PIL
        special_modules.append(os.path.dirname(PIL.__file__))
    except:
        pass
    try:
        import pyparsing
        special_modules.append(os.path.dirname(pyparsing.__file__))
    except:
        pass
    try:
        import dateutil
        special_modules.append(os.path.dirname(dateutil.__file__))
    except:
        pass
    try:
        import mpl_toolkits.mplot3d.axes3d as axes3d
        special_modules.append(os.path.dirname(os.path.dirname(axes3d.__file__)))
    except:
        pass
    try:
        import six
        special_modules.append(six.__file__)
    except:
        pass
    try:
        import cycler
        special_modules.append(cycler.__file__)
    except:
        pass
    try:
        import uuid
        special_modules.append(uuid.__file__)
    except:
        pass

excludes = []

try:
    import OpenGL
    special_modules += [os.path.dirname(OpenGL.__file__)]
except ImportError:
    print("OpenGL not available, not added to the frozen executables")

# This adds the interactive console but probably I should aim at an older
# version to reduce binary size. Tested with IPython 7.4.0
try:
    import IPython
    import pygments
    import qtconsole
    import asyncio
    import ipykernel
    import zmq
    includes.append("colorsys")
    special_modules += [os.path.dirname(IPython.__file__)]
    special_modules += [os.path.dirname(pygments.__file__)]
    special_modules += [os.path.dirname(qtconsole.__file__)]
    special_modules += [os.path.dirname(asyncio.__file__)]
    special_modules += [os.path.dirname(ipykernel.__file__)]
    special_modules += [os.path.dirname(zmq.__file__)]
except ImportError:
    print("qtconsole not available, not added to the frozen executables")

try:
    import silx
    import fabio
    import xml # needed by fabio
    special_modules += [os.path.dirname(silx.__file__),
                        os.path.dirname(fabio.__file__),
                        os.path.dirname(xml.__file__),]
    SILX = True
except ImportError:
    print("silx not available, not added to the frozen executables")
    SILX = False

try:
    import freeart
    import tomogui
    special_modules += [os.path.dirname(freeart.__file__),
                        os.path.dirname(tomogui.__file__)]
except ImportError:
    print("tomogui not available, not added to the frozen executables")

# package used by silx and probably others that is not always added properly
# always add it because it is small
try:
    import pkg_resources
    special_modules += [os.path.dirname(pkg_resources.__file__)]
    excludes += ["pkg_resources"]
except ImportError:
    print("pkg_resources could not be imported")

# pyopencl needs special treatment
try:
    import pyopencl
    import mako
    import cffi
    import pytools
    OPENCL = True
except:
    OPENCL = False

if sys.platform.lower().startswith("linux"):
    # no sense to freeze
    OPENCL = False

if OPENCL:
    special_modules.append(os.path.dirname(pyopencl.__file__))
    special_modules.append(os.path.dirname(mako.__file__))
    special_modules.append(os.path.dirname(cffi.__file__))
    special_modules.append(os.path.dirname(pytools.__file__))
    includes.append("decorator")
else:
    excludes.append("pyopencl")

# other generic packages not always properly detected but that are small and
# desirable to have
import collections
special_modules += [os.path.dirname(collections.__file__)]

# no scipy (huge package not used by PyMca)
excludes += ["scipy"]

# give some time to read the output
time.sleep(2)

def cleanup_cache(topdir): 
    # cleanup __pycache__
    for root, dirs, files in os.walk(topdir):
        for dirname in dirs:
            if dirname == "__pycache__": 
               print("deleting ",os.path.join(topdir, dirname))
               shutil.rmtree(os.path.join(topdir, dirname))
            else:
               cleanup_cache(os.path.join(topdir, dirname))

def replace_module(name):
    dest = os.path.join(Path(SPECPATH), 'dist', script_n[0])
    target = os.path.join(dest, os.path.basename(name))
    print("source = ", name)
    print("dest = ", target)
    if os.path.exists(target):
        if os.path.isdir(target):
            shutil.rmtree(target)
        else:
            os.remove(target)
    if os.path.isdir(name):
        shutil.copytree(name, target)
    else:
        shutil.copy2(src=name, dst=target)
    cleanup_cache(target)

for name in special_modules:
    replace_module(name)

# cleanup copied files
for fname in script_n:
    os.remove(fname)


# cleanup copied files
for fname in script_n:
    if os.path.exists(fname):
        os.remove(fname)

# patch silx
if SILX:
    fname = os.path.join(SPECPATH, "dist", script_n[0], "silx", "gui","qt","_qt.py")
    if os.path.exists(fname):
        print("###################################################################")
        print("Patching silx")
        print(fname)
        print("###################################################################")
        f = open(fname, "r")
        content = f.readlines()
        f.close()
        f = open(fname, "w")
        for line in content:
            f.write(line.replace("from PyQt5.uic import loadUi", "pass"))
        f.close()

# patch OpenCL
if OPENCL:
    # pyopencl __init__.py needs to be patched
    exe_win_dir = os.path.join(SPECPATH, "dist", script_n[0])
    initFile = os.path.join(exe_win_dir, "pyopencl", "__init__.py")
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
time.sleep(2)

for i in range(len(script_col)):
    app = BUNDLE(
        script_col[i],
        name=script_n[i] + ".app",
        icon=icon,
        bundle_identifier=None,
        info_plist={
            "CFBundleIdentifier": "com.github.vasole.pymca",
            "CFBundleShortVersionString": version,
            "CFBundleVersion": "PyMca " + version,
            "LSTypeIsPackage": True,
            "LSMinimumSystemVersion": "10.9.0",
            "NSHumanReadableCopyright": "MIT",
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
            "NSAppleScriptEnabled": False,
       },
    )

# make all the .app share the same resources
if sys.platform.startswith("darwin"):
    source = os.path.join(SPECPATH, "dist", script_n[0],"")
    destination = os.path.join(SPECPATH, "dist", script_n[0] + ".app", "Contents", "MacOS")
    cmd = "cp -Rf %s %s" % (source, destination)
    result = os.system(cmd)
    if result:
        raise IOError("Unsuccessful copy command <%s>" % cmd)
    subprocess.call(
        [
             "codesign",
             "--remove-signature",
             os.path.join("dist", script_n[0] + ".app", "Contents", "MacOS", "Python"),
        ]
    )
    if len(script_n) > 1:
        cwd = os.getcwd()
        for script in script_n[1:]:
            source = os.path.join(SPECPATH, "dist", script + ".app" ,"Contents", "MacOS")
            os.chdir(os.path.dirname(source))
            cmd = "rm -Rf MacOS"
            result = os.system(cmd)
            if result:
                os.chdir(cwd)
                raise IOError("Unsuccessful command <%s>" % cmd)
            target = os.path.join("..", "..", script_n[0] + ".app", "Contents", "MacOS")
            cmd = "ln -s %s %s" % (target, "MacOS")
            result = os.system(cmd)
            if result:
                os.chdir(cwd)
                raise IOError("Unsuccessful %s" % cmd)
            os.chdir(cwd)

    # rename the application
    version = PyMca5.version()
    source = os.path.join(SPECPATH, "dist", script_n[0] + ".app")
    dest = os.path.join(SPECPATH, "dist", "PyMca%s.app" % version)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.rename(source, dest)

    # Pack the application
    outFile = os.path.join(SPECPATH, "create-dmg.sh")
    f = open(os.path.join(SPECPATH, "create-dmg.sh.in"), "r")
    content = f.readlines()
    f.close()
    if os.path.exists(outFile):
        os.remove(outFile)
    f = open(outFile, "w")
    for line in content:
        if "__VERSION__" in line:
            line = line.replace("__VERSION__", version)
        f.write(line)
    f.close()
    subprocess.call(["bash", "create-dmg.sh"])

    # move the image to the top level dist directory
    dist = os.path.join(PROJECT_PATH, "dist")
    if not os.path.exists(dist):
        os.mkdir(dist)
    source = os.path.join(SPECPATH, "artifacts", "PyMca%s.dmg" % version)
    destination = os.path.join(PROJECT_PATH, "dist", "PyMca%s.dmg" % version)
    if os.path.exists(destination):
        os.remove(destination)
    os.rename(source, destination)
    shutil.rmtree(os.path.dirname(source))
else:
    # move generated directory to top level dist
    program = "PyMca"
    version = PyMca5.version()
    source = os.path.join(SPECPATH, "dist", script_n[0])
    dist = os.path.join(PROJECT_PATH, "dist",)
    if not os.path.exists(dist):
        os.mkdir(dist)
    target = os.path.join(dist, "%s%s" % (program, version))
    if sys.platform.startswith("darwin"):
        target += ".app"
    if os.path.exists(target):
        print("Removing target")
        shutil.rmtree(target)
    shutil.move(source, target)
    frozenDir = target

    #  generation of the NSIS executable
    nsis = os.path.join("\Program Files (x86)", "NSIS", "makensis.exe")
    if sys.platform.startswith("win") and os.path.exists(nsis):
        # check if we can perform the packaging
        outFile = os.path.join(PROJECT_PATH, "nsisscript.nsi")
        f = open(os.path.join(PROJECT_PATH,"nsisscript.nsi.in"), "r")
        content = f.readlines()
        f.close()
        if os.path.exists(outFile):
            os.remove(outFile)
        pymcaexe = os.path.join(PROJECT_PATH, "%s%s-win64.exe" % (program.lower(), version))
        if os.path.exists(pymcaexe):
            os.remove(pymcaexe)
        pymcalicense = os.path.join(PROJECT_PATH, "PyMca.txt")
        f = open(outFile, "w")
        for line in content:
            if "__LICENSE_FILE__" in line:
                line = line.replace("__LICENSE_FILE__", pymcalicense)
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

# cleanup intermediate files
for dname in ["build", "dist", "__pycache__"]:
    ddir = os.path.join(SPECPATH, dname)
    if os.path.exists(ddir):
        shutil.rmtree(ddir)

