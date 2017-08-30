#
# This Python module has been developed by V.A. Sole, from the European
# Synchrotron Radiation Facility (ESRF) to build PyMca.
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
import sys,os
import glob
import platform
import time
USING_SETUPTOOLS = True
if 'bdist_wheel' in sys.argv:
    # wheels require setuptools
    from setuptools import setup
    from setuptools.command.install import install as dftinstall
    from setuptools import Command
    from setuptools.extension import Extension
    from setuptools.command.build_py import build_py
    from distutils.command.install_data import install_data
    from setuptools.command.install_scripts import install_scripts
elif '--distutils' in sys.argv:
    # The cx_setup.py machinery works with distutils
    sys.argv.remove("--distutils")
    from distutils.core import setup
    from distutils.command.install import install as dftinstall
    from distutils.core import Command
    from distutils.core import Extension
    from distutils.command.build_py import build_py
    from distutils.command.install_data import install_data
    from distutils.command.install_scripts import install_scripts
    USING_SETUPTOOLS = False
else:
    try:
        from setuptools import setup
        from setuptools.command.install import install as dftinstall
        from setuptools import Command
        from setuptools.extension import Extension
        from setuptools.command.build_py import build_py
        from distutils.command.install_data import install_data
        from setuptools.command.install_scripts import install_scripts
    except ImportError:
        from distutils.core import setup
        from distutils.command.install import install as dftinstall
        from distutils.core import Command
        from distutils.core import Extension
        from distutils.command.build_py import build_py
        from distutils.command.install_data import install_data
        from distutils.command.install_scripts import install_scripts
        USING_SETUPTOOLS = False
try:
    import numpy
except ImportError:
    text  = "You must have numpy installed.\n"
    text += "See http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103\n"
    raise ImportError(text)
import distutils.sysconfig
try:
    from Cython.Distutils import build_ext
    import Cython.Compiler.Version
    if Cython.Compiler.Version.version < '0.18':
        build_ext = None
except:
    build_ext = None
global PYMCA_INSTALL_DIR
global PYMCA_SCRIPTS_DIR
global USE_SMART_INSTALL_SCRIPTS


#package maintainers customization
# Dear (Debian, RedHat, ...) package makers, please feel free to customize the
# following paths to the directory containing module's data relative to the
# directory containing the python modules (aka. installation directory).
# The sift module implements a patented algorithm. The algorithm can be used
# for non-commercial research purposes. If you do not want to distribute it
# with the PyMca sources you just need to delete the PyMca5/PyMcaMath/sift directory.
PYMCA_DATA_DIR = os.getenv("PYMCA_DATA_DIR")
PYMCA_DOC_DIR = os.getenv("PYMCA_DOC_DIR")
if PYMCA_DATA_DIR is None:
    PYMCA_DATA_DIR = os.path.join('PyMca5','PyMcaData')
if PYMCA_DOC_DIR is None:
    PYMCA_DOC_DIR = os.path.join('PyMca5','PyMcaData')

USE_SMART_INSTALL_SCRIPTS = False
if "--install-scripts" in sys.argv:
    USE_SMART_INSTALL_SCRIPTS = True

SPECFILE_USE_GNU_SOURCE = os.getenv("SPECFILE_USE_GNU_SOURCE")
if SPECFILE_USE_GNU_SOURCE is None:
    SPECFILE_USE_GNU_SOURCE = 0
    if sys.platform.lower().startswith("linux"):
        print("WARNING:")
        print("A cleaner locale independent implementation")
        print("may be achieved setting SPECFILE_USE_GNU_SOURCE to 1")
        print("For instance running this script as:")
        print("SPECFILE_USE_GNU_SOURCE=1 python setup.py build")
else:
    SPECFILE_USE_GNU_SOURCE = int(SPECFILE_USE_GNU_SOURCE)

# check if cython is not to be used despite being present
def use_cython():
    """
    Check if cython is disabled from the command line or the environment.
    """
    if "WITH_CYTHON" in os.environ:
        if os.environ["WITH_CYTHON"] in ["False", "0", 0]:
            print("No Cython requested by environment")
            return False

    if ("--no-cython" in sys.argv):
        sys.argv.remove("--no-cython")
        os.environ["WITH_CYTHON"] = "False"
        print("No Cython requested by command line")
        return False

    return True

if build_ext is not None:
    # we can use cython, but it may have been explicitely disabled
    if not use_cython():
        build_ext = None

# check if fisx library is to be build
def use_fisx():
    """
    Check if fisx is requested from the command line or the environment.
    """
    if "WITH_FISX" in os.environ:
        if os.environ["WITH_FISX"] in ["True", "1", 1]:
            print("Use of fisx library requested by environment")
            return True

    if ("--fisx" in sys.argv):
        sys.argv.remove("--fisx")
        os.environ["WITH_FISX"] = "True"
        print("Use fisx library requested by command line")
        return True

    return False

if use_fisx():
    # fisx is expected to be an independent library and
    # ideally one would use git subtree to put it in third-party
    # but one needs  git > 1.9.x
    fisx_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "third-party", "fisx")
    if not os.path.exists(fisx_src):
        print("fisx library not found but installation requested")
        raise IOError("Installation of fisx requested but library not found")
else:
    fisx_src = None

ffile = open(os.path.join('PyMca5', '__init__.py'), 'r').readlines()
for line in ffile:
    if line.startswith('__version__'):
        #remove spaces and split
        __version__ = "%s" % line.replace(' ','').split("=")[-1][:-1]
        #remove " or ' present
        __version__ = __version__[1:-1]
        break

# Make sure we work with a clean MANIFEST file
DEBIAN_SRC = False
if "sdist" in sys.argv:
    manifestFile  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "MANIFEST")
    if os.path.exists(manifestFile):
        os.remove(manifestFile)

    if "DEBIAN_SRC" in os.environ:
        if os.environ["DEBIAN_SRC"] in ["True", "1", 1]:
            print("Generating Debian specific source distribution without cython generated files")
            print("If you cython version is incompatible, it will be your problem")
            DEBIAN_SRC = True

print("PyMca X-Ray Fluorescence Toolkit %s\n" % __version__)

# The following is not supported by python-2.3:
#package_data = {'PyMca': ['attdata/*', 'HTML/*.*', 'HTML/IMAGES/*', 'HTML/PyMCA_files/*']}
packages = ['PyMca5','PyMca5.PyMcaPlugins', 'PyMca5.tests',
            'PyMca5.PyMca',
            'PyMca5.PyMcaCore',
            'PyMca5.PyMcaPhysics',
            'PyMca5.PyMcaPhysics.xrf',
            'PyMca5.PyMcaPhysics.xrf.XRFMC',
            'PyMca5.PyMcaPhysics.xas',
            'PyMca5.PyMcaIO',
            'PyMca5.PyMcaMisc',
            'PyMca5.PyMcaMath',
            'PyMca5.PyMcaMath.fitting',
            'PyMca5.PyMcaMath.mva',
            'PyMca5.PyMcaMath.mva.py_nnma',
            'PyMca5.PyMcaGraph','PyMca5.PyMcaGraph.backends',
            'PyMca5.PyMcaGui', 'PyMca5.PyMcaGui.plotting',
            'PyMca5.PyMcaGui.physics',
            'PyMca5.PyMcaGui.physics.xas',
            'PyMca5.PyMcaGui.physics.xrf',
            'PyMca5.PyMcaGui.pymca',
            'PyMca5.PyMcaGui.misc',
            'PyMca5.PyMcaGui.io',
            'PyMca5.PyMcaGui.io.hdf5',
            'PyMca5.PyMcaGui.math',
            'PyMca5.PyMcaGui.math.fitting',]
py_modules = []

# Specify all the required PyMca data
data_files = [(PYMCA_DATA_DIR, ['LICENSE',
                         'LICENSE.GPL',
                         'LICENSE.LGPL',
                         'LICENSE.MIT',                         
                         'PyMca5/PyMcaData/Scofield1973.dict',
                         'changelog.txt',
                         'PyMca5/PyMcaData/McaTheory.cfg',
                         'PyMca5/PyMcaData/PyMcaSplashImage.png',
                         'PyMca5/PyMcaData/KShellRatesScofieldHS.dat',
                         'PyMca5/PyMcaData/LShellRatesCampbell.dat',
                         'PyMca5/PyMcaData/LShellRatesScofieldHS.dat',
                         'PyMca5/PyMcaData/EXAFS_Cu.dat',
                         'PyMca5/PyMcaData/EXAFS_Ge.dat',
                         'PyMca5/PyMcaData/XRFSpectrum.mca']),
              (PYMCA_DATA_DIR+'/attdata', glob.glob('PyMca5/PyMcaData/attdata/*')),
              (PYMCA_DOC_DIR+'/HTML', glob.glob('PyMca5/PyMcaData/HTML/*.*')),
              (PYMCA_DOC_DIR+'/HTML/IMAGES', glob.glob('PyMca5/PyMcaData/HTML/IMAGES/*')),
              (PYMCA_DOC_DIR+'/HTML/PyMCA_files', glob.glob('PyMca5/HTML/PyMCA_files/*'))]

if os.path.exists(os.path.join("PyMca5", "EPDL97")):
    packages.append('PyMca5.EPDL97')
    data_files.append((PYMCA_DATA_DIR+'/EPDL97',glob.glob('PyMca5/EPDL97/*.DAT')))
    data_files.append((PYMCA_DATA_DIR+'/EPDL97',['PyMca5/EPDL97/LICENSE']))

global SIFT_OPENCL_FILES
SIFT_OPENCL_FILES = []
if os.path.exists(os.path.join("PyMca5", "PyMcaMath", "sift")):
    packages.append('PyMca5.PyMcaMath.sift')
    SIFT_OPENCL_FILES = glob.glob('PyMca5/PyMcaMath/sift/*.cl')
    data_files.append((os.path.join('PyMca5', 'PyMcaMath', 'sift'),
                       SIFT_OPENCL_FILES))

LOCAL_OBJECT3D =False
if os.path.exists(os.path.join("PyMca5", "Object3D")):
    LOCAL_OBJECT3D = True

if os.path.exists(os.path.join("PyMca5", "PyMcaGraph", "backends", "GLSupport")):
    packages.append('PyMca5.PyMcaGraph.backends.GLSupport')

if os.path.exists(os.path.join("PyMca5", "PyMcaGraph", "backends", "GLSupport", "gl")):
    packages.append('PyMca5.PyMcaGraph.backends.GLSupport.gl')


sources = glob.glob('*.c')
if sys.platform == "win32":
    define_macros = [('WIN32',None)]
    script_files = glob.glob('PyMca5/scripts/*')
    script_files += glob.glob('scripts/*.bat')
    script_files.append('scripts/pymca_win_post_install.py')
else:
    define_macros = []
    script_files = glob.glob('PyMca5/scripts/*')

def build_FastEdf(ext_modules):
    module  = Extension(name = 'PyMca5.FastEdf',
                        sources = glob.glob('PyMca5/PyMcaIO/edf/*.c'),
                        define_macros = define_macros,
                        include_dirs = [numpy.get_include()])
    ext_modules.append(module)

def build_specfile(ext_modules):
    if os.name.lower().startswith('posix'):
        specfile_define_macros = [('PYMCA_POSIX', None)]
        #the best choice is to use _GNU_SOURCE if possible
        #because that enables the use of strtod_l
        if SPECFILE_USE_GNU_SOURCE:
            specfile_define_macros = [('_GNU_SOURCE', 1)]
    else:
        specfile_define_macros = define_macros
    srcfiles = [ 'sfheader','sfinit','sflists','sfdata','sfindex',
             'sflabel' ,'sfmca', 'sftools','locale_management','specfile_py']
    if sys.version >= '3.0':
        srcfiles[-1] += '3'
    sources = []
    specfile_source_dir = os.path.join('PyMca5', 'PyMcaIO', 'specfile', 'src')
    specfile_include_dir = os.path.join('PyMca5', 'PyMcaIO', 'specfile', 'include')
    for ffile in srcfiles:
      sources.append(os.path.join(specfile_source_dir, ffile+'.c'))
    module  = Extension(name = 'PyMca5.PyMcaIO.specfile',
                        sources = sources,
                        define_macros = specfile_define_macros,
                        include_dirs = [specfile_include_dir,
                                            numpy.get_include()])
    ext_modules.append(module)

def build_specfit(ext_modules):
    module  = Extension(name = 'PyMca5.PyMcaMath.fitting.SpecfitFuns',
                        sources = glob.glob('PyMca5/PyMcaMath/fitting/specfit/*.c'),
                        define_macros = define_macros,
                        include_dirs = ['PyMca5/PyMcaMath/fitting/specfit',
                                         numpy.get_include()])
    ext_modules.append(module)

def build_sps(ext_modules):
    if platform.system() == 'Linux' :
        extra_compile_args = ['-pthread']
        #extra_compile_args = []
    elif platform.system() == 'SunOS' :
        #extra_compile_args = ['-pthreads']
        extra_compile_args = []
    else:
        extra_compile_args = []

    module  = Extension(name = 'PyMca5.spslut',
                         sources = ['PyMca5/PyMcaIO/sps/Src/sps_lut.c',
                                    'PyMca5/PyMcaIO/sps/Src/spslut_py.c'],
                         define_macros = define_macros,
                         extra_compile_args = extra_compile_args,
                         include_dirs = ['PyMca5/PyMcaIO/sps/Include',
                                          numpy.get_include()])
    ext_modules.append(module)
    if sys.platform != "win32":
        module = (Extension(name = 'PyMca5.PyMcaIO.sps',
                                            sources = ['PyMca5/PyMcaIO/sps/Src/sps.c',
                                                       'PyMca5/PyMcaIO/sps/Src/sps_py.c'],
                                            define_macros = define_macros,
                                 extra_compile_args = extra_compile_args,
                                            include_dirs = ['PyMca5/PyMcaIO/sps/Include',
                                                             numpy.get_include()]))
        ext_modules.append(module)

def build_PyMcaIOHelper(ext_modules):
    module  = Extension(name = 'PyMca5.PyMcaIO.PyMcaIOHelper',
                        sources = glob.glob('PyMca5/PyMcaIO/PyMcaIOHelper/*.c'),
                        define_macros = define_macros,
                        include_dirs = ['PyMca5/PyMcaIO/PyMcaIOHelper',
                                        numpy.get_include()])
    ext_modules.append(module)

def build_Object3DCTools(ext_modules):
    includes = [numpy.get_include()]
    if sys.platform == "win32":
        libraries = ['opengl32', 'glu32']
        # include headers missing in microsoft implementation of OpenGL
        includes.append('third-party/khronos_headers')
    elif sys.platform == "darwin":
        libraries = []
    else:
        libraries = ['GL', 'GLU']
    if sys.platform == 'windows':
        WindowsSDK = os.getenv('WindowsSdkDir')
        #if WindowsSDK is not None:
        #    includes.append(WindowsSDK)

    module  = Extension(name = 'PyMca5.Object3D.Object3DCTools',
                        sources = glob.glob('PyMca5/Object3D/Object3DCTools/*.c'),
                        define_macros = define_macros,
                        libraries  = libraries,
                        include_dirs = includes)
    ext_modules.append(module)


def build_Object3DQhull(extensions):
    libraries = []
    sources = ["PyMca5/Object3D/Object3DQhull/Object3DQhull.c"]
    include_dirs = [numpy.get_include()]

    # check if the user provide some information about a system qhull
    # library
    QHULL_CFLAGS = os.getenv("QHULL_CFLAGS")
    QHULL_LIBS = os.getenv("QHULL_LIBS")

    extra_compile_args = []
    extra_link_args = []
    if QHULL_CFLAGS and QHULL_LIBS:
        extra_compile_args += [QHULL_CFLAGS]
        extra_link_args += [QHULL_LIBS]
    else:
        sources += glob.glob("third-party/qhull/src/*.c")
        include_dirs += ["third-party/qhull/src"]

    module = Extension(name='PyMca5.Object3D.Object3DQhull',
                       sources=sources,
                       define_macros=define_macros,
                       libraries=libraries,
                       include_dirs=include_dirs,
                       extra_compile_args=extra_compile_args,
                       extra_link_args=extra_link_args)

    extensions.append(module)


def build_PyMcaSciPy(ext_modules):
    packages.append('PyMca5.PyMcaMath.PyMcaSciPy')
    packages.append('PyMca5.PyMcaMath.PyMcaSciPy.signal')
    module = Extension(name = 'PyMca5.PyMcaMath.PyMcaSciPy.signal.mediantools',
                       sources = glob.glob('PyMca5/PyMcaMath/PyMcaSciPy/signal/*.c'),
                       define_macros = [],
                       include_dirs = [numpy.get_include()])
    ext_modules.append(module)

def build_plotting_ctools(ext_modules):
    packages.append('PyMca5.PyMcaGraph.ctools')
    basedir = os.path.join('PyMca5', 'PyMcaGraph','ctools', '_ctools')
    c_files = [os.path.join(basedir, 'src', 'InsidePolygonWithBounds.c'),
               os.path.join(basedir, 'src', 'MinMaxImpl.c'),
               os.path.join(basedir, 'src', 'Colormap.c')]

    if build_ext:
        src = [os.path.join(basedir, 'cython','_ctools.pyx')]
    else:
        src = []
        for fname in glob.glob(os.path.join(basedir, 'cython','*.c')): 
            src.append(os.path.join(basedir, 'cython', os.path.basename(fname))) 
    src += c_files

    if sys.platform == 'win32':
        extra_compile_args = []
        extra_link_args = []
    else:
        extra_compile_args = []
        extra_link_args = []

    module = Extension(name="PyMca5.PyMcaGraph.ctools._ctools",
                        sources=src,
                        include_dirs=[numpy.get_include(),
                                      os.path.join(basedir, "include")],
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args,
                        language="c",
                        )
    """
    setup(
        name='ctools',
        ext_modules=[Extension(name="_ctools",
                    sources=src,
                    include_dirs=[numpy.get_include(),
                                  os.path.join(os.getcwd(),"include")],
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                    language="c",
                    )] ,
        cmdclass={'build_ext': build_ext},
    )
    """
    ext_modules.append(module)


def build_xas_xas(ext_modules):
    basedir = os.path.join('PyMca5', 'PyMcaPhysics','xas', '_xas')
    c_files = [os.path.join(basedir, 'src', 'polspl.c'),
               os.path.join(basedir, 'src', 'bessel0.c')]
    if build_ext:
        src = [os.path.join(basedir, 'cython','_xas.pyx')]
    else:
        src = []
        for fname in glob.glob(os.path.join(basedir, 'cython','*.c')): 
            src.append(os.path.join(basedir, 'cython', os.path.basename(fname))) 
    src += c_files
    if sys.platform == 'win32':
        extra_compile_args = []
        extra_link_args = []
    else:
        extra_compile_args = []
        extra_link_args = []
    module = Extension(name="PyMca5.PyMcaPhysics.xas._xas",
                        sources=src,
                        include_dirs=[numpy.get_include(),
                                      os.path.join(basedir, "include")],
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args,
                        language="c",
                        )
    ext_modules.append(module)

ext_modules = []
if sys.version < '3.0':
    build_FastEdf(ext_modules)
build_specfile(ext_modules)
build_specfit(ext_modules)
build_sps(ext_modules)
build_PyMcaIOHelper(ext_modules)
if LOCAL_OBJECT3D:
    try:
        build_Object3DCTools(ext_modules)
        build_Object3DQhull(ext_modules)
        for python_file in glob.glob('PyMca5/Object3D/*.py'):
            if python_file in ['setup.py', 'cx_setup.py']:
                continue
            m = "PyMca5.Object3D.%s" % os.path.basename(python_file)[:-3]
            py_modules.append(m)
        for python_file in glob.glob('PyMca5/Object3D/Object3DPlugins/*.py'):
            m = "PyMca5.Object3D.Object3DPlugins.%s" %\
                                    os.path.basename(python_file)[:-3]
            py_modules.append(m)
    except:
        print("Object3D Module could not be built")
        print(sys.exc_info())
build_PyMcaSciPy(ext_modules)
build_plotting_ctools(ext_modules)
build_xas_xas(ext_modules)


class smart_build_py(build_py):
    def run (self):
        toReturn = build_py.run(self)
        global PYMCA_DATA_DIR
        global PYMCA_DOC_DIR
        global PYMCA_INSTALL_DIR
        defaultPath = os.path.join('PyMca5','PyMcaData')
        if (PYMCA_DATA_DIR == defaultPath) or\
           (PYMCA_DOC_DIR == defaultPath):
            #default, just make sure the complete path is there
            install_cmd = self.get_finalized_command('install')
            PYMCA_INSTALL_DIR = getattr(install_cmd, 'install_lib')

        #packager should have given the complete path
        #in other cases
        if PYMCA_DATA_DIR == defaultPath:
            PYMCA_DATA_DIR = os.path.join(PYMCA_INSTALL_DIR,
                                          PYMCA_DATA_DIR)
        if PYMCA_DOC_DIR == defaultPath:
            #default, just make sure the complete path is there
            PYMCA_DOC_DIR = os.path.join(PYMCA_INSTALL_DIR,
                                          PYMCA_DOC_DIR)
        target = os.path.join(self.build_lib, "PyMca5", "PyMcaDataDir.py")
        fid = open(target,'r')
        content = fid.readlines()
        fid.close()
        fid = open(target,'w')
        for line in content:
            lineToBeWritten = line
            txt = 'DATA_DIR_FROM_SETUP'
            if txt in line:
                lineToBeWritten = line.replace(txt, PYMCA_DATA_DIR)
            txt = 'DOC_DIR_FROM_SETUP'
            if txt in line:
                lineToBeWritten = line.replace(txt, PYMCA_DOC_DIR)
            fid.write(lineToBeWritten)
        fid.close()
        return toReturn

# data_files fix from http://wiki.python.org/moin/DistutilsInstallDataScattered
class smart_install_data(install_data):
    if USING_SETUPTOOLS:
        def initialize_options (self):
            self.outfiles = []
            self.data_files = data_files

        def finalize_options(self):
            pass

        def get_outputs(self):
            return self.outfiles

    def run(self):
        global PYMCA_INSTALL_DIR
        global PYMCA_DATA_DIR
        global PYMCA_DOC_DIR
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        PYMCA_INSTALL_DIR = self.install_dir
        print("PyMca to be installed in %s" %  self.install_dir)

        #cleanup old stuff if present
        pymcaOld = os.path.join(PYMCA_INSTALL_DIR, "PyMca5", "Plugins1D")
        if os.path.exists(pymcaOld):
            for f in glob.glob(os.path.join(pymcaOld,"*.py")):
                print("Removing previously installed file %s" % f)
                os.remove(f)
            for f in glob.glob(os.path.join(pymcaOld,"*.pyc")):
                print("Removing previously installed file %s" % f)
                os.remove(f)
            print("Removing previously installed directory %s" % pymcaOld)
            os.rmdir(pymcaOld)
        pymcaOld = os.path.join(PYMCA_INSTALL_DIR, "PyMca5", "PyMca.py")
        if os.path.exists(pymcaOld):
            print("Removing previously installed file %s" % pymcaOld)
            os.remove(pymcaOld)
        pymcaOld += "c"
        if os.path.exists(pymcaOld):
            print("Removing previously installed file %s" % pymcaOld)
            os.remove(pymcaOld)
        return install_data.run(self)


# smart_install_scripts
if USE_SMART_INSTALL_SCRIPTS:
    class smart_install_scripts(install_scripts):
        if USING_SETUPTOOLS:
            def initialize_options (self):
                self.outfiles = []

            def finalize_options(self):
                pass

            def get_outputs(self):
                return self.outfiles

        def run (self):
            global PYMCA_SCRIPTS_DIR
            #I prefer not to translate the python used during the build
            #process for the case of having an installation on a disk shared
            #by different machines and starting python from a shell script
            #that positions the environment
            from distutils import log
            from stat import ST_MODE
            install_cmd = self.get_finalized_command('install')
            #This is to ignore the --install-scripts keyword
            #I do not know if to leave it optional ...
            if False:
                self.install_dir = os.path.join(getattr(install_cmd, 'install_lib'), 'PyMca5')
                self.install_dir = os.path.join(self.install_dir, 'bin')
            else:
                self.install_dir = getattr(install_cmd, 'install_scripts')
            self.install_data = getattr(install_cmd, 'install_data')
            if "." in self.install_dir:
                self.install_dir = os.path.abspath(self.install_dir)
            if "." in self.install_data:
                self.install_data = os.path.abspath(self.install_data)
            global PYMCA_INSTALL_DIR
            if "." in PYMCA_INSTALL_DIR:
                PYMCA_INSTALL_DIR = os.path.abspath(PYMCA_INSTALL_DIR)
            PYMCA_SCRIPTS_DIR = self.install_dir
            PYMCA_DATA_DIR = self.install_data
            if sys.platform != "win32":
                print("PyMca scripts to be installed in %s" %  self.install_dir)
            self.outfiles = self.copy_tree(self.build_dir, self.install_dir)
            self.outfiles = []
            for filein in glob.glob('PyMca5/scripts/*'):
                filedest = os.path.join(self.install_dir, os.path.basename(filein))
                if os.path.exists(filedest):
                    os.remove(filedest)
                moddir = os.path.join(PYMCA_INSTALL_DIR, "PyMca5", "PyMcaGui")
                if 0:
                    f = open(filein, 'r')
                    modfile = f.readline().replace("\n","")
                    f.close()
                else:
                    basename = os.path.basename(filein)
                    if basename.startswith('pymcabatch'):
                        modfile = os.path.join("pymca", 'PyMcaBatch.py')
                    elif basename.startswith('pymcapostbatch') or\
                         basename.startswith('rgbcorrelator'):
                        modfile = os.path.join("pymca", 'PyMcaPostBatch.py')
                    elif basename.startswith('pymcaroitool'):
                        modfile = os.path.join("pymca", 'QStackWidget.py')
                    elif basename.startswith('mca2edf'):
                        modfile = os.path.join("pymca", 'Mca2Edf.py')
                    elif basename.startswith('edfviewer'):
                        modfile = os.path.join("pymca", 'EdfFileSimpleViewer.py')
                    elif basename.startswith('peakidentifier'):
                        modfile = os.path.join("physics", "xrf", 'PeakIdentifier.py')
                    elif basename.startswith('elementsinfo'):
                        modfile = os.path.join("physics", "xrf", 'ElementsInfo.py')
                    elif basename.startswith('pymca'):
                        modfile = os.path.join("pymca", 'PyMcaMain.py')
                    else:
                        print("ignored %s" % filein)
                        continue
                text  = "#!/bin/bash\n"
                text += "export PYTHONPATH=%s:${PYTHONPATH}\n" % PYMCA_INSTALL_DIR
                #deal with sys.executables not named python
                text += "exec %s %s $*\n" %  (
                    sys.executable,
                    os.path.join(moddir, modfile)
                    )

                f=open(filedest, 'w')
                f.write(text)
                f.close()
                #self.copy_file(filein, filedest)
                self.outfiles.append(filedest)
            if os.name == 'posix':
                # Set the executable bits (owner, group, and world) on
                # all the scripts we just installed.
                for ffile in self.get_outputs():
                    if self.dry_run:
                        log.info("changing mode of %s", ffile)
                    else:
                        # python 2.5 does not accept next line
                        #mode = ((os.stat(ffile)[ST_MODE]) | 0o555) & 0o7777
                        mode = ((os.stat(ffile)[ST_MODE]) | 365) & 4095
                        log.info("changing mode of %s to %o", ffile, mode)
                        os.chmod(ffile, mode)

# man pages handling
def abspath(*path):
    """A method to determine absolute path for a given relative path to the
    directory where this setup.py script is located"""
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(setup_dir, *path)


class install_man(Command):

    user_options = [
        ('install-dir=', 'd', 'base directory for installing man page files')]

    def initialize_options(self):
        self.install_dir = None
        if USING_SETUPTOOLS:
            self.outfiles = []

    def finalize_options(self):
        self.set_undefined_options('install',
                                   ('install_man', 'install_dir'))

    if USING_SETUPTOOLS:
        def get_outputs(self):
            return self.outfiles

    def run(self):
        if self.install_dir is None:
            return
        src_man_dir = abspath('doc', 'man')
        man_elems = os.listdir(src_man_dir)
        man_pages = []
        for f in man_elems:
            f = os.path.join(src_man_dir,f)
            if not os.path.isfile(f):
                continue
            if not f.endswith(".1"):
                continue
            man_pages.append(f)

        install_dir = os.path.join(self.install_dir, 'man1')

        if not os.path.isdir(install_dir):
            os.makedirs(install_dir)

        for man_page in man_pages:
            self.copy_file(man_page, install_dir)

class install(dftinstall):

    user_options = list(dftinstall.user_options)
    user_options.extend([
        ('install-man=', None, 'installation directory for Unix man pages')])

    def initialize_options(self):
        self.install_man = None
        dftinstall.initialize_options(self)

    def finalize_options(self):
        # We do a hack here. We cannot trust the 'install_base' value because it
        # is not always the final target. For example, in unix, the install_base
        # is '/usr' and all other install_* are directly relative to it. However,
        # in unix-local (like ubuntu) install_base is still '/usr' but, for
        # example, install_data, is '$install_base/local' which breaks everything.
        #
        # The hack consists in using install_data instead of install_base since
        # install_data seems to be, in practice, the proper install_base on all
        # different systems.
        global USE_SMART_INSTALL_SCRIPTS
        dftinstall.finalize_options(self)
        if os.name != "posix":
            if self.install_man is not None:
                self.warn("install-man option ignored on this platform")
                self.install_man = None
        else:
            if self.install_man is None:
                if not USE_SMART_INSTALL_SCRIPTS:
                    # if one is installing the scripts somewhere else
                    # he can be smart enough to pass install_man
                    self.install_man = os.path.join(self.install_data,\
                                                    'share', 'man')
        if self.install_man is not None:
            if not os.path.exists(self.install_man):
                try:
                    os.makedirs(self.install_man)
                except:
                    #we'll get the error in the next check
                    pass
            #check if we can write
            if not os.access(self.install_man, os.W_OK):
                print("********************************")
                print("")
                print("No permission to write man pages")
                print("")
                print("********************************")
                self.install_man = None
        self.dump_dirs("Installation directories")

    def expand_dirs(self):
        dftinstall.expand_dirs(self)
        self._expand_attrs(['install_man'])

    def has_man(self):
        return os.name == "posix"

    sub_commands = list(dftinstall.sub_commands)
    sub_commands.append(('install_man', has_man))


# end of man pages handling
cmdclass = {'install_data':smart_install_data,
            'build_py':smart_build_py}
if build_ext is not None:
    cmdclass['build_ext'] = build_ext

if USE_SMART_INSTALL_SCRIPTS:
    # typical use of user without superuser privileges
    cmdclass['install_scripts'] = smart_install_scripts

if os.name == "posix":
    cmdclass['install'] = install
    cmdclass['install_man'] = install_man

if DEBIAN_SRC:
    from distutils.command.sdist import sdist
    class sdist_debian(sdist):
        def prune_file_list(self):
            sdist.prune_file_list(self)
            directories = ["third-party/fisx/python/cython",
                           "PyMca5/PyMcaGraph/ctools/_ctools/cython",
                           "PyMca5/PyMcaPhysics/xas/_xas/cython"]
            print("Removing files for debian source distribution")
            for directory in directories:
                for pyxfile in glob.glob(os.path.join(directory, "*.pyx")):
                    for extension in [".c", ".cpp"]:
                        cf = os.path.splitext(pyxfile)[0] + extension
                        if os.path.isfile(cf):
                            print("Excluding file %s" % cf)
                            self.filelist.exclude_pattern(pattern=cf)

    cmdclass['sdist'] = sdist_debian

description = "Mapping and X-Ray Fluorescence Analysis"
long_description = """Stand-alone application and Python tools for interactive and/or batch processing analysis of X-Ray Fluorescence Spectra. Graphical user interface (GUI) and batch processing capabilities provided
"""

#######################
# build_doc commands #
#######################

try:
    import sphinx
    import sphinx.util.console
    sphinx.util.console.color_terminal = lambda: False
    from sphinx.setup_command import BuildDoc
except:
    sphinx = None

if sphinx:
    class build_doc(BuildDoc):

        def run(self):

            # make sure the python path is pointing to the newly built
            # code so that the documentation is built on this and not a
            # previously installed version

            build = self.get_finalized_command('build')
            sys.path.insert(0, os.path.abspath(build.build_lib))

            # Build the Users Guide in HTML and TeX format
            for builder in ('html', 'latex'):
                self.builder = builder
                self.builder_target_dir = os.path.join(self.build_dir, builder)
                self.mkpath(self.builder_target_dir)
                builder_index = 'index_{0}.txt'.format(builder)
                BuildDoc.run(self)
            sys.path.pop(0)
    cmdclass['build_doc'] = build_doc

classifiers = ["Development Status :: 5 - Production/Stable",
               "Programming Language :: Python :: 2",
               "Programming Language :: Python :: 3",
               "Intended Audience :: Developers",
               "Intended Audience :: End Users/Desktop",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Topic :: Software Development :: Libraries :: Python Modules",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: Unix",
               "Operating System :: MacOS :: MacOS X",
               "Operating System :: POSIX",
               "Topic :: Scientific/Engineering :: Chemistry",
               "Topic :: Scientific/Engineering :: Physics",
               "Topic :: Scientific/Engineering :: Visualization",
               ]

install_requires = ["numpy", "matplotlib", "fisx>=1.1.4"]
setup_requires = ["numpy"]

distrib = setup(name="PyMca5",
                version= __version__,
                description = description,
                author = "V. Armando Sole",
                author_email="sole@esrf.fr",
                license= "MIT",
                url = "http://pymca.sourceforge.net",
                download_url="https://github.com/vasole/pymca/archive/v%s.tar.gz" % __version__,  		
                long_description = long_description,
                packages = packages,
                platforms='any',
                ext_modules = ext_modules,
                data_files = data_files,
##                package_data = package_data,
##                package_dir = {'':'PyMca', 'PyMca.tests':'tests'},
                cmdclass = cmdclass,
                scripts=script_files,
                py_modules=py_modules,
                classifiers=classifiers,
                install_requires=install_requires,
                setup_requires=setup_requires,
                )

try:
    print("PyMca is installed in %s " % PYMCA_INSTALL_DIR)
    print("PyMca data files are installed in %s " % PYMCA_DATA_DIR)
    print("HTML help files are installed in %s " % PYMCA_DOC_DIR)
except:
    #I really do not see how this may happen but ...
    pass

if fisx_src is None or ("sdist" in sys.argv):
    sys.exit(0)

#
# What follows is ugly and it is not intended for
# use outside the ESRF
#
# fisx should be installed as a separated Debian/RedHat/... package
#

fisx_src = os.path.abspath(fisx_src)

# reproduce fisx setup.py file here ...
# deal with required data

#for the time being there is no doc directory
FISX_DATA_DIR = os.getenv("FISX_DATA_DIR")
FISX_DOC_DIR = os.getenv("FISX_DOC_DIR")

if FISX_DATA_DIR is None:
    FISX_DATA_DIR = os.path.join('fisx', 'fisx_data')
if FISX_DOC_DIR is None:
    FISX_DOC_DIR = os.path.join('fisx', 'fisx_data')

def getFisxVersion():
    cppDir = os.path.join(fisx_src, "src")
    content = open(os.path.join(cppDir, "fisx_version.h"), "r").readlines()
    for line in content:
        if "FISX_VERSION_STR" in line:
            version = line.split("FISX_VERSION_STR")[-1].replace("\n","")
            version = version.replace(" ","")
            return version[1:-1]

__version__ = getFisxVersion()

print("Processing fisx library %s\n" % __version__)

class smart_build_fisx_py(build_py):
    def run (self):
        toReturn = build_py.run(self)
        global FISX_DATA_DIR
        global FISX_DOC_DIR
        global INSTALL_DIR
        defaultDataPath = os.path.join('fisx', 'fisx_data')
        defaultDocPath = os.path.join('fisx', 'fisx_data')
        if (FISX_DATA_DIR == defaultDataPath) or\
           (FISX_DOC_DIR == defaultDocPath):
            #default, just make sure the complete path is there
            install_cmd = self.get_finalized_command('install')
            INSTALL_DIR = getattr(install_cmd, 'install_lib')

        #packager should have given the complete path
        #in other cases
        if FISX_DATA_DIR == defaultDataPath:
            FISX_DATA_DIR = os.path.join(INSTALL_DIR, FISX_DATA_DIR)
        if FISX_DOC_DIR == defaultDocPath:
            #default, just make sure the complete path is there
            FISX_DOC_DIR = os.path.join(INSTALL_DIR, FISX_DOC_DIR)
        target = os.path.join(self.build_lib, "fisx", "DataDir.py")
        fid = open(target,'r')
        content = fid.readlines()
        fid.close()
        fid = open(target,'w')
        for line in content:
            lineToBeWritten = line
            if lineToBeWritten.startswith("FISX_DATA_DIR"):
                lineToBeWritten = "FISX_DATA_DIR = r'%s'\n" % FISX_DATA_DIR
            if line.startswith("FISX_DOC_DIR"):
                lineToBeWritten = "FISX_DOC_DIR = r'%s'\n" % FISX_DOC_DIR
            fid.write(lineToBeWritten)
        fid.close()
        return toReturn

class smart_install_fisx_data(install_data):
    def run(self):
        global INSTALL_DIR
        global FISX_DATA_DIR
        global FISX_DOC_DIR
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        INSTALL_DIR = self.install_dir
        print("fisx to be installed in %s" %  self.install_dir)
        return install_data.run(self)

topLevel = fisx_src
fileList = glob.glob(os.path.join(topLevel, "fisx_data", "*.dat"))
fileList.append(os.path.join(topLevel, "changelog.txt"))
fileList.append(os.path.join(topLevel, "LICENSE"))
fileList.append(os.path.join(topLevel, "README.rst"))
fileList.append(os.path.join(topLevel, "TODO"))
data_files = [(FISX_DATA_DIR, fileList)]

# actual build stuff
FORCE = False
#cython_dir = os.path.join(os.getcwd(), "python", "cython")
cython_dir = os.path.join(fisx_src, "python", "cython")
if build_ext:
    #make sure everything is totally clean?
    fileList  = glob.glob(os.path.join(cython_dir, "*.cpp"))
    fileList += glob.glob(os.path.join(cython_dir, "*.h"))
    if FORCE:
        for fname in fileList:
            if os.path.exists(fname):
                os.remove(fname)
    #this does not work:
    #src = glob.glob(os.path.join(cython_dir, "*pyx"))
    multiple_pyx = os.path.join(cython_dir, "_fisx.pyx")
    if os.path.exists(multiple_pyx):
        try:
            os.remove(multiple_pyx)
        except:
            print("WARNING: Could not delete file. Assuming up-to-date.")
    if not os.path.exists(multiple_pyx):
        pyx = glob.glob(os.path.join(cython_dir, "*pyx"))
        # until _fisx.pyx gets updated, we have to put
        # the name PyEPDL after PyElement
        pyx = sorted(pyx, key=str.lower)
        f = open(multiple_pyx, 'wb')
        for fname in pyx:
            inFile = open(fname, 'rb')
            lines = inFile.readlines()
            inFile.close()
            for line in lines:
                f.write(line)
        f.close()
    src = [multiple_pyx]
else:
    src = glob.glob(os.path.join(cython_dir,'*.cpp'))
src += glob.glob(os.path.join(fisx_src,
                'src', 'fisx_*.cpp'))

include_dirs = [numpy.get_include(),
                os.path.join(fisx_src, "src")]

if sys.platform == 'win32':
    extra_compile_args = ['/EHsc']
    extra_link_args = []
else:
    extra_compile_args = []
    extra_link_args = []

def buildExtension():
    module = Extension(name="fisx._fisx",
                    sources=src,
                    include_dirs=include_dirs,
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                    language="c++",
                    )
    return module

ext_modules = [buildExtension()]

cmdclass = {'install_data':smart_install_fisx_data,
            'build_py':smart_build_fisx_py,
            'build_ext': build_ext,
            }

description = "Quantitative X-Ray Fluorescence Analysis Support Library"
long_description = """
Tools to evaluate the expected X-ray fluorescence measured when a sample is excitated by an X-ray beam. Secondary and tertiary excitation effects taken into account.
"""

# tell distutils where to find the packages
package_dir = {"":os.path.join(fisx_src, "python")}
packages = ['fisx', 'fisx.tests']
setup(
    name='fisx',
    version=__version__,
    author="V. Armando Sole",
    author_email="sole@esrf.fr",
    description=description,
    long_description=long_description,
    license="MIT",
    url="https://github.com/vasole/fisx",
    package_dir=package_dir,
    packages=packages,
    ext_modules=ext_modules,
    data_files=data_files,
    cmdclass=cmdclass,
)
os.chdir(os.path.abspath(os.path.dirname(__file__)))

