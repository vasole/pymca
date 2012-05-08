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
global PYMCA_INSTALL_DIR
global PYMCA_SCRIPTS_DIR
global USE_SMART_INSTALL_SCRIPTS 


#package maintainers customization
# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to the directory containing module's data relative to the
# directory containing the python modules (aka. installation directory) 
PYMCA_DATA_DIR = 'PyMca/PyMcaData'
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

ffile = open(os.path.join('PyMca', 'PyMcaMain.py'), 'r').readlines()
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
packages = ['PyMca','PyMca.PyMcaPlugins', 'PyMca.tests']
py_modules = []

# Specify all the required PyMca data
data_files = [(PYMCA_DATA_DIR, ['LICENSE.GPL',
                         'PyMca/PyMcaData/Scofield1973.dict',
                         'changelog.txt',
                         'PyMca/PyMcaData/McaTheory.cfg',
                         'PyMca/PyMcaData/PyMcaSplashImage.png',
                         'PyMca/PyMcaData/BindingEnergies.dat',
                         'PyMca/PyMcaData/KShellRates.dat',
                         'PyMca/PyMcaData/KShellRatesScofieldHS.dat',
                         'PyMca/PyMcaData/KShellConstants.dat',
                         'PyMca/PyMcaData/LShellRates.dat',
                         'PyMca/PyMcaData/LShellConstants.dat',
                         'PyMca/PyMcaData/LShellRatesCampbell.dat',
                         'PyMca/PyMcaData/LShellRatesScofieldHS.dat',
                         'PyMca/PyMcaData/MShellRates.dat',
                         'PyMca/PyMcaData/MShellConstants.dat',
                         'PyMca/PyMcaData/EADL97_BindingEnergies.dat',
                         'PyMca/PyMcaData/EADL97_KShellConstants.dat',
                         'PyMca/PyMcaData/EADL97_LShellConstants.dat',
                         'PyMca/PyMcaData/EADL97_MShellConstants.dat',
                         'PyMca/PyMcaData/EPDL97_CrossSections.dat',
                         'PyMca/PyMcaData/XCOM_CrossSections.dat',
                         'PyMca/PyMcaData/XRFSpectrum.mca']),
              (PYMCA_DATA_DIR+'/attdata', glob.glob('PyMca/PyMcaData/attdata/*')),
              (PYMCA_DATA_DIR+'/HTML', glob.glob('PyMca/PyMcaData/HTML/*.*')),
              (PYMCA_DATA_DIR+'/HTML/IMAGES', glob.glob('PyMca/PyMcaData/HTML/IMAGES/*')),
              (PYMCA_DATA_DIR+'/HTML/PyMCA_files', glob.glob('PyMca/HTML/PyMCA_files/*'))]

if os.path.exists(os.path.join("PyMca", "EPDL97")):
    packages.append('PyMca.EPDL97')
    data_files.append((PYMCA_DATA_DIR+'/EPDL97',glob.glob('PyMca/EPDL97/*.DAT')))
    data_files.append((PYMCA_DATA_DIR+'/EPDL97',['PyMca/EPDL97/LICENSE']))

NNMA_PATH = os.path.join("PyMca", "py_nnma")
if os.path.exists(NNMA_PATH):
    py_modules.append('PyMca.py_nnma.__init__')
    py_modules.append('PyMca.py_nnma.nnma')
    
LOCAL_OBJECT3D =False
if os.path.exists(os.path.join("PyMca", "Object3D")):
    LOCAL_OBJECT3D = True

sources = glob.glob('*.c')
if sys.platform == "win32":
    define_macros = [('WIN32',None)]
    script_files = glob.glob('PyMca/scripts/*')
    script_files += glob.glob('scripts/*.bat')
    script_files.append('scripts/pymca_win_post_install.py')
else:
    define_macros = []
    script_files = glob.glob('PyMca/scripts/*')
            
def build_FastEdf(ext_modules):
    module  = Extension(name = 'PyMca.FastEdf',
                        sources = glob.glob('PyMca/edf/*.c'),
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
    sources = glob.glob('PyMca/specfile/src/*.c')
    if sys.version < '3.0':
        todelete = 'specfile_py3.c'
    else:
        todelete = 'specfile_py.c'
    for i in range(len(sources)):
        if todelete in sources[i]:
            del sources[i]
            break
    module  = Extension(name = 'PyMca.specfile',
                        sources = sources,
                        define_macros = specfile_define_macros,
                        include_dirs = ['PyMca/specfile/include',
                                            numpy.get_include()])
    ext_modules.append(module)

def build_specfit(ext_modules):
    module  = Extension(name = 'PyMca.SpecfitFuns',
                        sources = glob.glob('PyMca/specfit/*.c'),
                        define_macros = define_macros,
                        include_dirs = ['PyMca/specfit',
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

    module  = Extension(name = 'PyMca.spslut',
                         sources = ['PyMca/sps/Src/sps_lut.c',
                                    'PyMca/sps/Src/spslut_py.c'],
                         define_macros = define_macros,
                         extra_compile_args = extra_compile_args,          
                         include_dirs = ['PyMca/sps/Include',
                                          numpy.get_include()])
    ext_modules.append(module)
    if (sys.platform != "win32") and (sys.version < '3.0'):
        module = (Extension(name = 'PyMca.sps',
                                            sources = ['PyMca/sps/Src/sps.c',
                                                       'PyMca/sps/Src/sps_py.c'],
                                            define_macros = define_macros,
                                 extra_compile_args = extra_compile_args,          
                                            include_dirs = ['PyMca/sps/Include',
                                                             numpy.get_include()]))
        ext_modules.append(module)

def build_PyMcaIOHelper(ext_modules):
    module  = Extension(name = 'PyMca.PyMcaIOHelper',
                        sources = glob.glob('PyMca/PyMcaIOHelper/*.c'),
                        define_macros = define_macros,
                        include_dirs = ['PyMca/PyMcaIOHelper',
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

    module  = Extension(name = 'PyMca.Object3D.Object3DCTools',
                        sources = glob.glob('PyMca/Object3D/Object3DCTools/*.c'),
                        define_macros = define_macros,
                        libraries  = libraries,
                        include_dirs = includes)
    ext_modules.append(module)

def build_Object3DQhull(ext_modules):
    if sys.platform == "win32":
        libraries = ['opengl32', 'glu32']
    else:
        libraries = ['GL', 'GLU']        
    module  = Extension(name = 'PyMca.Object3D.Object3DQhull',
                        sources = glob.glob('PyMca/Object3D/Object3DQhull/src/*.c'),
                        define_macros = define_macros,
                        include_dirs = [numpy.get_include()])

    ext_modules.append(module)

def build_PyMcaSciPy(ext_modules):
    packages.append('PyMca.PyMcaSciPy')
    packages.append('PyMca.PyMcaSciPy.signal')
    module = Extension(name = 'PyMca.PyMcaSciPy.signal.mediantools',
                       sources = glob.glob('PyMca/PyMcaSciPy/signal/*.c'),
                       define_macros = [],
                       include_dirs = [numpy.get_include()])
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
        for python_file in glob.glob('PyMca/Object3D/*.py'):
            if python_file in ['setup.py', 'cx_setup.py']:
                continue
            m = "PyMca.Object3D.%s" % os.path.basename(python_file)[:-3] 
            py_modules.append(m)
        for python_file in glob.glob('PyMca/Object3D/Object3DPlugins/*.py'):
            m = "PyMca.Object3D.Object3DPlugins.%s" %\
                                    os.path.basename(python_file)[:-3] 
            py_modules.append(m)
    except:
        print("Object3D Module could not be built")
        print(sys.exc_info())
build_PyMcaSciPy(ext_modules)

# data_files fix from http://wiki.python.org/moin/DistutilsInstallDataScattered
from distutils.command.install_data import install_data
class smart_install_data(install_data):
    def run(self):
        global PYMCA_INSTALL_DIR
        global PYMCA_DATA_DIR
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        PYMCA_INSTALL_DIR = self.install_dir
        print("PyMca to be installed in %s" %  self.install_dir)
        pymcaOld = os.path.join(PYMCA_INSTALL_DIR, "PyMca", "Plugins1D")
        if os.path.exists(pymcaOld):
            for f in glob.glob(os.path.join(pymcaOld,"*.py")):
                print("Removing previously installed file %s" % f)
                os.remove(f)
            for f in glob.glob(os.path.join(pymcaOld,"*.pyc")):
                print("Removing previously installed file %s" % f)
                os.remove(f)
            print("Removing previously installed directory %s" % pymcaOld)
            os.rmdir(pymcaOld)
        pymcaOld = os.path.join(PYMCA_INSTALL_DIR, "PyMca", "PyMca.py")
        if os.path.exists(pymcaOld):
            print("Removing previously installed file %s" % pymcaOld)
            os.remove(pymcaOld)
        pymcaOld += "c"
        if os.path.exists(pymcaOld):
            print("Removing previously installed file %s" % pymcaOld)
            os.remove(pymcaOld)
        #create file with package data information destination
        pymca_directory = os.path.join(PYMCA_INSTALL_DIR, "PyMca")
        if not os.path.exists(pymca_directory):
            os.mkdir(pymca_directory)
        tmpName = os.path.join(pymca_directory, 'PyMcaDataDir.py')
        if os.path.exists(tmpName):
            print("Removing previously installed file %s" % tmpName)
            os.remove(tmpName)
        f = open(tmpName, 'w')
        if PYMCA_DATA_DIR == 'PyMca/PyMcaData':
            #default, just make sure the complete path is there
            PYMCA_DATA_DIR = os.path.join(PYMCA_INSTALL_DIR,
                                          PYMCA_DATA_DIR)
            #packager should have given the complete path
            #in other cases
        f.write("import os\nPYMCA_DATA_DIR = '%s'\n" % PYMCA_DATA_DIR)
        f.write("# what follows is only used in frozen versions\n")
        f.write("if not os.path.exists(PYMCA_DATA_DIR):\n")
        f.write("    tmp_dir = os.path.dirname(__file__)\n")
        f.write("    basename = os.path.basename(PYMCA_DATA_DIR)\n")
        f.write("    PYMCA_DATA_DIR = os.path.join(tmp_dir,basename)\n")
        f.write("    while len(PYMCA_DATA_DIR) > 14:\n")
        f.write("        if os.path.exists(PYMCA_DATA_DIR):\n")
        f.write("            break\n")
        f.write("        tmp_dir = os.path.dirname(tmp_dir)\n")
        f.write("        PYMCA_DATA_DIR = os.path.join(tmp_dir, basename)\n")
        f.write("if not os.path.exists(PYMCA_DATA_DIR):\n")
        f.write("    raise IOError('%s directory not found' % basename)\n")
        #f.write("print('Using: %s' % PYMCA_DATA_DIR)\n")
        f.close()
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
                self.install_dir = os.path.join(getattr(install_cmd, 'install_lib'), 'PyMca')
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
            for filein in glob.glob('PyMca/scripts/*'):
                filedest = os.path.join(self.install_dir, os.path.basename(filein))
                if os.path.exists(filedest):
                    os.remove(filedest)
                moddir = os.path.join(getattr(install_cmd,'install_lib'), "PyMca")
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
cmdclass = {'install_data':smart_install_data}

if USE_SMART_INSTALL_SCRIPTS:
    # typical use of user without superuser privileges
    cmdclass['install_scripts'] = smart_install_scripts

if os.name == "posix":
    cmdclass['install'] = install
    cmdclass['install_man'] = install_man

description = "Mapping and X-Ray Fluorescence Analysis"
long_description = """Stand-alone application and Python tools for interactive and/or batch processing analysis of X-Ray Fluorescence Spectra. Graphical user interface (GUI) and batch processing capabilities provided
"""

distrib = setup(name="PyMca",
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
except:
    #I really do not see how this may happen but ...
    pass
