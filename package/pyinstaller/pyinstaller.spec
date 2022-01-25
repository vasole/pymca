# -*- mode: python -*-
import sys
import os.path
from pathlib import Path
import shutil
import subprocess

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = []

PROJECT_PATH = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
datas.append((os.path.join(PROJECT_PATH, "README.rst"), "."))
datas.append((os.path.join(PROJECT_PATH, "LICENSE"), "."))
datas.append((os.path.join(PROJECT_PATH, "copyright"), "."))
#datas += collect_data_files("silx.resources")


hiddenimports = []
hiddenimports += collect_submodules('encodings.ascii')
hiddenimports += collect_submodules('encodings.utf_8')
hiddenimports += collect_submodules('encodings.latin_1')
hiddenimports += collect_submodules('fabio')
hiddenimports += collect_submodules('PyQt5.uic')
#hiddenimports += collect_submodules('hdf5plugin')
#hiddenimports += collect_submodules('fisx')
#hiddenimports += collect_submodules('PyMca5.PyMcaGui.PyMcaQt')
#hiddenimports += collect_submodules('PyMca5.PyMcaGui.pymca')

excludes = ["fabio", "hdf5plugin", "silx"]

# det the script list
from PyMca5.PyMcaGui.pymca import PyMcaMain, PyMcaBatch, PyMcaPostBatch, QStackWidget
script_l = [os.path.abspath(PyMcaMain.__file__),
            os.path.abspath(PyMcaBatch.__file__),
            os.path.abspath(PyMcaPostBatch.__file__),
            os.path.abspath(QStackWidget.__file__),
            os.path.join(SPECPATH, "pymca_launcher.py"),
           ]
#script_l = [os.path.join(SPECPATH, "mymain.py"),
#            os.path.join(SPECPATH, "mymain2.py"),
#            ]
script_n = []
for s in script_l:
    script_n.append(os.path.basename(s).split(".")[0].lower())

block_cipher = None

script_a = []

for script in script_l:
    cwd = Path(SPECPATH)
    try:
        print("Copying %s to %s" % (script, os.path.basename(script).split(".")[0].lower()))
        shutil.copy2(
            src = script,
            dst = os.path.join(cwd, os.path.basename(script))# .split(".")[0].lower()),
        )
    except:
        pass
    print("passed")

    script_a.append(Analysis(
                            [os.path.basename(script)],
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
    script_a[-1].pure = [x for x in script_a[-1].pure if not x[0].startswith("PyMca5.")]


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

#MERGE(
#    (silx_a, 'silx', os.path.join('silx', 'silx')),
#    (silx_view_a, 'silx-view', os.path.join('silx-view', 'silx-view')),
#)


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
            icon='silx.ico')
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


# Fix MERGE by copying produced exe files to the location of the first one
def move_exe(name):
    dist = os.path.join(Path(SPECPATH), 'dist')
    shutil.copy2(
        src=os.path.join(dist, name, name + '.exe'),
        dst=os.path.join(dist, script_n[0])
    )
    shutil.rmtree(os.path.join(dist, name))

if len(script_a) > 1:
    for n in script_n[1:]:
        move_exe(n)


# Mandatory modules to be integrally included in the frozen version.
# One can add other modules here if not properly detected
# (PyQt5 seems to be properly handled, if not, add to special_modules)
import time
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
                   os.path.dirname(matplotlib.__file__),
                   #os.path.dirname(ctypes.__file__),
                   os.path.dirname(hdf5plugin.__file__),
                   ]
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
except ImportError:
    print("silx not available, not added to the frozen executables")

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
def replace_module(name):
    dest = os.path.join(Path(SPECPATH), 'dist', script_n[0])
    target = os.path.join(dest, os.path.basename(name))
    if os.path.exists(target):
        shutil.rmtree(target)
    print("source = ", name)
    print("dest = ", target)
    shutil.copytree(name, target)

for name in special_modules:
    replace_module(name)

if 0:
    # Run innosetup
    def innosetup():
        from silx import version

        config_name = "create-installer.iss"
        with open(config_name + '.template') as f:
            content = f.read().replace("#Version", version)
        with open(config_name, "w") as f:
            f.write(content)

        subprocess.call(["iscc", os.path.join(SPECPATH, config_name)])
        os.remove(config_name)

    innosetup()
