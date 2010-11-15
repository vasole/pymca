import sys,os
import glob
import platform
from distutils.core import Extension, setup
try:
    import numpy
except ImportError:
    text  = "You must have numpy installed.\n"
    text += "See http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103\n"
    raise ImportError, text
import distutils.sysconfig
global PYMCA_INSTALL_DIR
global PYMCA_SCRIPTS_DIR
import string

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

for line in file(os.path.join('PyMca', 'PyMcaMain.py')).readlines():
    if line[:11] == '__version__':
        exec(line)
        # Append cvs tag if working from cvs tree
        if os.path.isdir('.svn') and os.path.isfile(os.sep.join(['.svn', 'entries'])):
            import re
            revision = 0
            revre = re.compile('committed-rev="(\d+)"')
            for match in revre.finditer(open(os.sep.join(['.svn', 'entries'])).read()):
                revision = max(revision, int(match.group(1)))
            __version__ += 'dev_r%i' % revision
        break

print "PyMca X-Ray Fluorescence Toolkit %s" % __version__
print 

print "Type 'L' to view the license."
print "Type 'yes' to accept the terms of the license."
print "Type 'no' to decline the terms of the license."
print

while 1:
    try:
        resp = raw_input("Do you accept the terms of the license? ")
    except KeyboardInterrupt:
        raise SystemExit
    except:
        resp = ""

    resp = string.lower(string.strip(resp))

    if resp == "yes":
        break

    if resp == "no":
        sys.exit(1)

    if resp == "l":
        os.system("more LICENSE.GPL")


# Specify all the required PyMca data
data_files = [('PyMca', ['LICENSE.GPL',
                         'PyMca/Scofield1973.dict',
                         'PyMca/changelog.txt',
                         'PyMca/McaTheory.cfg',
                         'PyMca/PyMcaSplashImage.png',
                         'PyMca/BindingEnergies.dat',
                         'PyMca/KShellRates.dat','PyMca/KShellRatesScofieldHS.dat','PyMca/KShellConstants.dat',
                         'PyMca/LShellRates.dat','PyMca/LShellConstants.dat',
                         'PyMca/LShellRatesCampbell.dat','PyMca/LShellRatesScofieldHS.dat',
                         'PyMca/MShellRates.dat','PyMca/MShellConstants.dat',
                         'PyMca/EADL97_BindingEnergies.dat',
                         'PyMca/EADL97_LShellConstants.dat',
                         'PyMca/EADL97_MShellConstants.dat',
                         'PyMca/EPDL97_CrossSections.dat']),
              ('PyMca/attdata', glob.glob('PyMca/attdata/*')),
              ('PyMca/PyMcaPlugins', glob.glob('PyMca/PyMcaPlugins/*')),
              ('PyMca/HTML', glob.glob('PyMca/HTML/*.*')),
              ('PyMca/HTML/IMAGES', glob.glob('PyMca/HTML/IMAGES/*')),
              ('PyMca/HTML/PyMCA_files', glob.glob('PyMca/HTML/PyMCA_files/*'))]

if os.path.exists(os.path.join("PyMca", "EPDL97")):
    data_files.append(('PyMca/EPDL97',glob.glob('PyMca/EPDL97/*.py')))
    data_files.append(('PyMca/EPDL97',glob.glob('PyMca/EPDL97/*.DAT')))
    data_files.append(('PyMca/EPDL97',['PyMca/EPDL97/LICENSE']))

LOCAL_PHYNX =False
if os.path.exists(os.path.join("PyMca", "phynx")):
    LOCAL_PHYNX = True
    data_files.append(('PyMca/phynx', glob.glob('PyMca/phynx/*.py')))
    data_files.append(('PyMca/phynx/utils', glob.glob('PyMca/phynx/utils/*.py')))

NNMA_PATH = os.path.join("PyMca", "nnma", "py_nnma")
if os.path.exists(NNMA_PATH):
    NNMA_PATH = os.path.join(NNMA_PATH, "*.py")
    data_files.append(('PyMca/py_nnma', glob.glob(NNMA_PATH)))

LOCAL_OBJECT3D =False
if os.path.exists(os.path.join("PyMca", "object3d")):
    LOCAL_OBJECT3D = True

# The following is not supported by python-2.3:
#package_data = {'PyMca': ['attdata/*', 'HTML/*.*', 'HTML/IMAGES/*', 'HTML/PyMCA_files/*']}
packages = ['PyMca']

sources = glob.glob('*.c')
if sys.platform == "win32":
    define_macros = [('WIN32',None)]
    script_files = []
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
    module  = Extension(name = 'PyMca.specfile',
                        sources = glob.glob('PyMca/specfile/src/*.c'),
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
    if sys.platform != "win32":
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
                        sources = glob.glob('PyMca/object3d/Object3D/Object3DCTools/*.c'),
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
                        sources = glob.glob('PyMca/object3d/Object3D/Object3DQhull/src/*.c'),
                        define_macros = define_macros,
                        include_dirs = [numpy.get_include()])

    ext_modules.append(module)

def build_PyMcaSciPy(ext_modules):
    data_files.append(('PyMca/PyMcaSciPy', ['PyMca/PyMcaSciPy/__init__.py']))
    data_files.append(('PyMca/PyMcaSciPy/signal', ['PyMca/PyMcaSciPy/signal/__init__.py',
                                      'PyMca/PyMcaSciPy/signal/median.py']))                   
    module = Extension(name = 'PyMca.PyMcaSciPy.signal.mediantools',
                       sources = glob.glob('PyMca/PyMcaSciPy/signal/*.c'),
                       define_macros = [],
                       include_dirs = [numpy.get_include()])
    ext_modules.append(module)

ext_modules = []
build_FastEdf(ext_modules)
build_specfile(ext_modules)
build_specfit(ext_modules)
build_sps(ext_modules)
build_PyMcaIOHelper(ext_modules)
if LOCAL_OBJECT3D:
    try:
        build_Object3DCTools(ext_modules)
        build_Object3DQhull(ext_modules)
        data_files.append(('PyMca/Object3D', glob.glob('PyMca/object3d/Object3D/*.py')))
        data_files.append(('PyMca/Object3D/Object3DPlugins',
                       glob.glob('PyMca/object3d/Object3D/Object3DPlugins/*.py')))
    except:
        print "Object3D Module could not be built"
        print sys.exc_info()
build_PyMcaSciPy(ext_modules)

# data_files fix from http://wiki.python.org/moin/DistutilsInstallDataScattered
from distutils.command.install_data import install_data
class smart_install_data(install_data):
    def run(self):
        global PYMCA_INSTALL_DIR
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        PYMCA_INSTALL_DIR = self.install_dir
        print "PyMca to be installed in %s" %  self.install_dir
        pymcaOld = os.path.join(PYMCA_INSTALL_DIR, "PyMca", "Plugins1D")
        if os.path.exists(pymcaOld):
            for f in glob.glob(os.path.join(pymcaOld,"*.py")):
                print "Removing previously installed file %s" % f
                os.remove(f)
            for f in glob.glob(os.path.join(pymcaOld,"*.pyc")):
                print "Removing previously installed file %s" % f
                os.remove(f)
            print "Removing previously installed directory %s" % pymcaOld
            os.rmdir(pymcaOld)
        pymcaOld = os.path.join(PYMCA_INSTALL_DIR, "PyMca", "PyMca.py")
        if os.path.exists(pymcaOld):
            print "Removing previously installed file %s" % pymcaOld
            os.remove(pymcaOld)
        pymcaOld += "c"
        if os.path.exists(pymcaOld):
            print "Removing previously installed file %s" % pymcaOld
            os.remove(pymcaOld)
        return install_data.run(self)

from distutils.command.install_scripts import install_scripts
"""
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
        PYMCA_SCRIPTS_DIR = self.install_dir        
        if sys.platform != "win32":
            print "PyMca scripts to be installed in %s" %  self.install_dir
        self.outfiles = self.copy_tree(self.build_dir, self.install_dir)
        self.outfiles = []
        for filein in glob.glob('PyMca/scripts/*'):
            filedest = os.path.join(self.install_dir, os.path.basename(filein))
            if os.path.exists(filedest):
                os.remove(filedest)
            moddir = os.path.join(getattr(install_cmd,'install_lib'), "PyMca")
            f = open(filein, 'r')
            modfile = f.readline().replace("\n","")
            f.close()
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
            for file in self.get_outputs():
                if self.dry_run:
                    log.info("changing mode of %s", file)
                else:
                    mode = ((os.stat(file)[ST_MODE]) | 0555) & 07777
                    log.info("changing mode of %s to %o", file, mode)
                    os.chmod(file, mode)
"""   
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
##                package_dir = {'': 'lib'},
                cmdclass = {'install_data':smart_install_data}, 
#                            'install_scripts':smart_install_scripts},
                scripts=script_files,
                )



#post installation checks
try:
    import sip
    SIP = True
except ImportError:
    SIP = False
    print "sip must be installed for full pymca functionality."

badtext  = "No valid PyQt  with PyQwt4 or PyQwt5 installation found.\n"
badtext += "No valid PyQt4 with PyQwt5 installation found.\n"
badtext += "You will only be able to develop applications using  a very \n"
badtext += "small subset of PyMca."

try:
    print "PyMca is installed in %s " % PYMCA_INSTALL_DIR
except NameError:
    #I really do not see how this may happen but ...
    pass
    
if SIP:
    try:
        import PyQt4.QtCore
        QT4 = True
    except ImportError:
        QT4 = False
    except:
        QT4 = True

    try:        
        from PyQt4 import Qwt5
        QWT5 = True        
    except ImportError:
        QWT5 = False

    QT3  = False
    QWT4 = False
    if not QT4:
        try:
            import qt
            QT3 = True
        except ImportError:
            QT3 = False
        except:
	    pass

        try:
            import Qwt5 as qwt
            QWT5 = True
        except ImportError:
            QWT5 = False
        except:
            pass

        if not QWT5:
            try:
                import Qwt4 as qwt
                QWT4 = True        
            except ImportError:
                QWT4 = False

            if not QWT4:
                try:
                    import qwt
                    QWT4 = True        
                except ImportError:
                    QWT4 = False

    if QT4 and QT3:
        #print "PyMca does not work in a mixed Qt4 and qt installation (yet)"
        if QWT5:
            print "You have PyQt4 and PyQwt5 installed."
            print "PyMca is fully functional under PyQt4 with PyQwt5."
            print "You can easily embed PyMca fitting in your Qt4 graphical "
            print "applications using McaAdvancedFit.py"
        else:
            print badtext
    elif QT3 and QWT5:
        print "PyMca PyQt installations tested with PyQwt4"
        print "You have PyQwt5 installed. It should also work."
        print "PyMca installation successfully completed."
    elif QT3 and not QWT4:
        print "PyMca PyQt installations need PyQwt5 or PyQwt4"
        print badtext
    elif QT4 and QWT5:
        print "You have PyQt4 and PyQwt5 installed."
        print "PyMca is fully functional under PyQt4 with PyQwt5."
        print "You can easily embed PyMca fitting in your Qt4 graphical "
        print "applications using McaAdvancedFit.py"
        try:
            if sys.platform != 'win32':
                print "Please make sure %s is in your path" % PYMCA_SCRIPTS_DIR
                print "and try the scripts:"
                for script in script_files:
                    s = os.path.basename(script)
                    #if s.upper() == "PYMCA":continue
                    #if s.upper() == "MCA2EDF":continue
                    print s
        except NameError:
            pass
    elif QT3 and QWT4:
        print "PyMca installation successfully completed."
        try:
            if sys.platform != 'win32':
                print "Please make sure %s is in your path" % PYMCA_SCRIPTS_DIR
                print "and try the scripts:"
                for script in script_files:
                    print os.path.basename(script)
        except NameError:
            pass
    else:
        print badtext
else:
    print "No valid PyQt with qwt or PyQt4 with PyQwt5 installation found."
    print "You will only be able to develop applications using  a very "
    print "small subset of PyMca."
