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
from distutils.core import Extension, setup, Command
from distutils.command.install import install as dftinstall
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
PYMCA_DATA_DIR = os.path.join('PyMca5','PyMcaData')
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

ffile = open(os.path.join('PyMca5', 'PyMcaGui', 'pymca', 'PyMcaMain.py'), 'r').readlines()
for line in ffile:
    if line.startswith('__version__'):
        #remove spaces and split
        __version__ = "%s" % line.replace(' ','').split("=")[-1][:-1]
        #remove " or ' present
        __version__ = __version__[1:-1]
        break

print("PyMca X-Ray Fluorescence Toolkit %s\n" % __version__)

# The following is not supported by python-2.3:
#package_data = {'PyMca': ['attdata/*', 'HTML/*.*', 'HTML/IMAGES/*', 'HTML/PyMCA_files/*']}
packages = ['PyMca5','PyMca5.PyMcaPlugins', 'PyMca5.tests',
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
            'PyMca5.PyMcaGui.physics.xrf',
            'PyMca5.PyMcaGui.pymca',
            'PyMca5.PyMcaGui.misc',
            'PyMca5.PyMcaGui.io',
            'PyMca5.PyMcaGui.io.hdf5',
            'PyMca5.PyMcaGui.math',
            'PyMca5.PyMcaGui.math.fitting',]
py_modules = []

# Specify all the required PyMca data
data_files = [(PYMCA_DATA_DIR, ['LICENSE.GPL',
                         'PyMca5/PyMcaData/Scofield1973.dict',
                         'changelog.txt',
                         'PyMca5/PyMcaData/McaTheory.cfg',
                         'PyMca5/PyMcaData/PyMcaSplashImage.png',
                         'PyMca5/PyMcaData/BindingEnergies.dat',
                         'PyMca5/PyMcaData/KShellRates.dat',
                         'PyMca5/PyMcaData/KShellRatesScofieldHS.dat',
                         'PyMca5/PyMcaData/KShellConstants.dat',
                         'PyMca5/PyMcaData/LShellRates.dat',
                         'PyMca5/PyMcaData/LShellConstants.dat',
                         'PyMca5/PyMcaData/LShellRatesCampbell.dat',
                         'PyMca5/PyMcaData/LShellRatesScofieldHS.dat',
                         'PyMca5/PyMcaData/MShellRates.dat',
                         'PyMca5/PyMcaData/MShellConstants.dat',
                         'PyMca5/PyMcaData/EADL97_BindingEnergies.dat',
                         'PyMca5/PyMcaData/EADL97_KShellConstants.dat',
                         'PyMca5/PyMcaData/EADL97_LShellConstants.dat',
                         'PyMca5/PyMcaData/EADL97_MShellConstants.dat',
                         'PyMca5/PyMcaData/EPDL97_CrossSections.dat',
                         'PyMca5/PyMcaData/XCOM_CrossSections.dat',
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
    if (sys.platform != "win32") and (sys.version < '3.0'):
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
    basedir = os.path.join(os.getcwd(),'PyMca5', 'PyMcaGraph','ctools', '_ctools')
    c_files = glob.glob(os.path.join(basedir, 'src', 'InsidePolygonWithBounds.c'))
    if build_ext:
        src = glob.glob(os.path.join(basedir, 'cython','_ctools.pyx'))
    else:
        src = glob.glob(os.path.join(basedir, 'cython','*.c'))
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


ext_modules = []
if sys.version < '3.0':
    build_FastEdf(ext_modules)
build_specfile(ext_modules)
build_specfit(ext_modules)
build_sps(ext_modules)
build_PyMcaIOHelper(ext_modules)
if (sys.version < '3.0') and LOCAL_OBJECT3D:
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

from distutils.command.build_py import build_py
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
from distutils.command.install_data import install_data
class smart_install_data(install_data):
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
    from distutils.command.install_scripts import install_scripts
    class smart_install_scripts(install_scripts):
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
                moddir = os.path.join(getattr(install_cmd,'install_lib'), "PyMca5")
                if 0:
                    f = open(filein, 'r')
                    modfile = f.readline().replace("\n","")
                    f.close()
                else:
                    basename = os.path.basename(filein) 
                    if basename.startswith('pymcabatch'):
                        modfile = 'PyMcaBatch.py'
                    elif basename.startswith('pymcapostbatch') or\
                         basename.startswith('rgbcorrelator'):
                        modfile = 'PyMcaPostBatch.py' 
                    elif basename.startswith('pymcaroitool'):
                        modfile = 'QStackWidget.py'
                    elif basename.startswith('mca2edf'):
                        modfile = 'Mca2Edf.py'
                    elif basename.startswith('edfviewer'):
                        modfile = 'EdfFileSimpleViewer.py'
                    elif basename.startswith('peakidentifier'):
                        modfile = 'PeakIdentifier.py'
                    elif basename.startswith('elementsinfo'):
                        modfile = 'ElementsInfo.py'
                    elif basename.startswith('pymca'):
                        modfile = 'PyMcaMain.py'
                    else:
                        print("ignored %s" % filein)
                        continue   
                text  = "#!/bin/bash\n"
                text += "export PYTHONPATH=%s:${PYTHONPATH}\n" % moddir
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

    def finalize_options(self):
        self.set_undefined_options('install',
                                   ('install_man', 'install_dir'))

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
            'build_py':smart_build_py,
            'build_ext': build_ext}

if USE_SMART_INSTALL_SCRIPTS:
    # typical use of user without superuser privileges
    cmdclass['install_scripts'] = smart_install_scripts

if os.name == "posix":
    cmdclass['install'] = install
    cmdclass['install_man'] = install_man

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

distrib = setup(name="PyMca5",
                version= __version__,
                description = description,
                author = "V. Armando Sole",
                author_email="sole@esrf.fr",
                license= "GPL - Please read LICENSE.GPL for details",
                url = "http://pymca.sourceforge.net",
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
                )

try:
    print("PyMca is installed in %s " % PYMCA_INSTALL_DIR)
    print("PyMca data files are installed in %s " % PYMCA_DATA_DIR)
    print("HTML help files are installed in %s " % PYMCA_DOC_DIR)
except:
    #I really do not see how this may happen but ...
    pass
