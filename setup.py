import sys,os
import glob
from distutils.core import Extension, setup
import distutils.sysconfig

for line in file('PyMca/PyMca.py').readlines():
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

# Specify all the required PyMca data
data_files = [('PyMca', ['PyMca/Scofield1973.dict', 'PyMca/McaTheory.cfg']),
              ('PyMca/attdata', glob.glob('PyMca/attdata/*')),
              ('PyMca/HTML', glob.glob('PyMca/HTML/*.*')),
              ('PyMca/HTML/IMAGES', glob.glob('PyMca/HTML/IMAGES/*')),
              ('PyMca/HTML/PyMCA_files', glob.glob('PyMca/HTML/PyMCA_files/*'))]
# The following is not supported by python-2.3:
#package_data = {'PyMca': ['attdata/*', 'HTML/*.*', 'HTML/IMAGES/*', 'HTML/PyMCA_files/*']}
packages = ['PyMca']

sources = glob.glob('*.c')
if sys.platform == "win32":
    define_macros = [('WIN32',None)]
else:
    define_macros = []

def build_FastEdf(ext_modules):
    module  = Extension(name = 'PyMca.FastEdf',
                                            sources = glob.glob('PyMca/edf/*.c'),
                                            define_macros = define_macros)
    ext_modules.append(module)

def build_specfile(ext_modules):
    module  = Extension(name = 'PyMca.specfile',
                                            sources = glob.glob('PyMca/specfile/src/*.c'),
                                            define_macros = define_macros,
                                            include_dirs = ['PyMca/specfile/include'])
    ext_modules.append(module)

def build_specfit(ext_modules):
    module  = Extension(name = 'PyMca.SpecfitFuns',
                                            sources = glob.glob('PyMca/specfit/*.c'),
                                            define_macros = define_macros,
                                            include_dirs = ['PyMca/specfit'])
    ext_modules.append(module)

def build_sps(ext_modules):
    module  = Extension(name = 'PyMca.spslut',
                         sources = ['PyMca/sps/Src/sps_lut.c',
                                    'PyMca/sps/Src/spslut_py.c'],
                         define_macros = define_macros,
                         include_dirs = ['PyMca/sps/Include'])
    ext_modules.append(module)
    if sys.platform != "win32":
        module = (Extension(name = 'PyMca.sps',
                                            sources = ['PyMca/sps/Src/sps.c',
                                                       'PyMca/sps/Src/sps_py.c'],
                                            define_macros = define_macros,
                                            include_dirs = ['PyMca/sps/Include']))
        ext_modules.append(module)

ext_modules = []
build_FastEdf(ext_modules)
build_specfile(ext_modules)
build_specfit(ext_modules)
build_sps(ext_modules)

# data_files fix from http://wiki.python.org/moin/DistutilsInstallDataScattered
from distutils.command.install_data import install_data
class smart_install_data(install_data):
    def run(self):
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        return install_data.run(self)

description = ""
long_description = """
"""

distrib = setup(name="PyMca",
                license = "GPL - Please read LICENSE.GPL for details",
                version= __version__,
                description = description,
                author = "V. Armando Sole",
                author_email="sole@esrf.fr",
                url = "http://sourceforge.net/projects/pymca",
                long_description = long_description,
                packages = packages,
                platforms='any',
                ext_modules = ext_modules,
                data_files = data_files,
##                package_data = package_data,
##                package_dir = {'': 'lib'},
                cmdclass = {'install_data':smart_install_data},
                )
