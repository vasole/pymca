#/*##########################################################################
# Copyright (C) 2004-2020 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys,os
import glob
import platform
from distutils.core import Extension, setup
try:
    import numpy
except ImportError:
    text  = "You must have numpy installed.\n"
    text += "See http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103\n"
    raise ImportError(text)
import distutils.sysconfig
global OBJECT3D_INSTALL_DIR
global OBJECT3D_SCRIPTS_DIR
import string

__version__ = '1.0'
for line in file('SceneGLWindow.py').readlines():
    if line[:11] == '__version__':
        exec(line)
        # Append cvs tag if working from cvs tree
        if os.path.isdir('.svn') and os.path.isfile(os.sep.join(['.svn', 'entries'])):
            import re
            revision = 0
            revre = re.compile(r'committed-rev="(\d+)"')
            for match in revre.finditer(open(os.sep.join(['.svn', 'entries'])).read()):
                revision = max(revision, int(match.group(1)))
            __version__ += 'dev_r%i' % revision
        break

print("Object3D Toolkit %s" % __version__)
print()
print("Type 'L' to view the license.")
print("Type 'yes' to accept the terms of the license.")
print("Type 'no' to decline the terms of the license.")
print()

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
        os.system("more LICENSE.LGPL")


# Specify all the required Object3D data
data_files = [('Object3D', ['LICENSE.LGPL',])]

packages = ['Object3D']
package_dir = {'Object3D':os.path.dirname(__file__)}

"""
#py_modules = []
py_modules = []
for python_file in glob.glob('*.py'):
    if python_file not in ['setup.py', 'cx_setup.py']:
        continue
    m = "Object3D.%s" % (os.path.basename(python_file)[:-3])
    py_modules.append(m)
"""

py_modules = []
for python_file in glob.glob('Object3DPlugins/*.py'):
    m = "Object3D.Object3DPlugins.%s" % (os.path.basename(python_file)[:-3])
    py_modules.append(m)

sources = glob.glob('*.c')
if sys.platform == "win32":
    libraries = ['opengl32', 'glu32']
    define_macros = [('WIN32',None)]
    script_files = []
    script_files.append('scripts/object3d_win_post_install.py')
else:
    libraries = ['GL', 'GLU']
    if sys.platform == 'darwin':
        libraries=[]
    define_macros = []
    script_files = []
    for f in glob.glob('scripts/*'):
        if f.endswith('.py'):
            continue
        script_files.append(f)


def build_Object3DCTools(ext_modules):
    includes = [numpy.get_include()]
    if sys.platform == 'windows':
        WindowsSDK = os.getenv('WindowsSdkDir')
        #if WindowsSDK is not None:
        #    includes.append(WindowsSDK)
    module  = Extension(name = 'Object3D.Object3DCTools',
                        sources = glob.glob('Object3DCTools/*.c'),
                        define_macros = define_macros,
                        libraries  = libraries,
                        include_dirs = includes)
    ext_modules.append(module)

def build_Object3DQhull(ext_modules):
    module  = Extension(name = 'Object3D.Object3DQhull',
                        sources = glob.glob('Object3DQhull/src/*.c'),
                        define_macros = define_macros,
                        include_dirs = [numpy.get_include()])

    ext_modules.append(module)

ext_modules = []
build_Object3DCTools(ext_modules)
build_Object3DQhull(ext_modules)

# data_files fix from http://wiki.python.org/moin/DistutilsInstallDataScattered
from distutils.command.install_data import install_data
class smart_install_data(install_data):
    def run(self):
        global OBJECT3D_INSTALL_DIR
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        OBJECT3D_INSTALL_DIR = self.install_dir
        print("Object3D to be installed in %s" %  self.install_dir)
        return install_data.run(self)

from distutils.command.install_scripts import install_scripts
class smart_install_scripts(install_scripts):
    def run (self):
        global OBJECT3D_SCRIPTS_DIR
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
            self.install_dir = os.path.join(getattr(install_cmd, 'install_lib'), 'Object3D')
            self.install_dir = os.path.join(self.install_dir, 'bin')
        else:
            self.install_dir = getattr(install_cmd, 'install_scripts')
        OBJECT3D_SCRIPTS_DIR = self.install_dir
        if sys.platform != "win32":
            print("Object3D scripts to be installed in %s" %  self.install_dir)
        self.outfiles = self.copy_tree(self.build_dir, self.install_dir)
        self.outfiles = []
        for filein in glob.glob('scripts/*'):
            filedest = os.path.join(self.install_dir, os.path.basename(filein))
            if os.path.exists(filedest):
                os.remove(filedest)
            moddir = os.path.join(getattr(install_cmd,'install_lib'), "Object3D")
            f = open(filein, 'r')
            modfile = f.readline().replace("\n","")
            f.close()
            text  = "#!/bin/bash\n"
            text += "export PYTHONPATH=%s:${PYTHONPATH}\n" % moddir
            text += "exec python %s $*\n" %  os.path.join(moddir, modfile)
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
                    mode = ((os.stat(ffile)[ST_MODE]) | 365) & 4095
                    log.info("changing mode of %s to %o", ffile, mode)
                    os.chmod(ffile, mode)

description = "LGPL License unless a commercial license is bought. Please contact industry@esrf.fr if needed."
long_description = """Stand-alone python application and tools for multidimensional data visualization"""

distrib = setup(name="Object3D",
                version= __version__,
                description = description,
                author = "V. Armando Sole",
                author_email="sole@esrf.fr",
                license= "LGPL2+ - Please read LICENSE.LGPL for details",
                url = "http://pymca.sourceforge.net",
                long_description = long_description,
                packages = packages,
                package_dir=package_dir,
                platforms='any',
                ext_modules = ext_modules,
                data_files = data_files,
                cmdclass = {'install_data':smart_install_data,
                            'install_scripts':smart_install_scripts},
                scripts=script_files,
                py_modules=py_modules,
                )
#cleanup files
for fname in ['setup.py', 'cx_setup.py']:
    file_to_remove = os.path.join(OBJECT3D_INSTALL_DIR, 'Object3D',fname)
    if os.path.exists(file_to_remove):
        os.remove(file_to_remove)
    if os.path.exists(file_to_remove+'c'):
        os.remove(file_to_remove+'c')
    if os.path.exists(file_to_remove+'o'):
        os.remove(file_to_remove+'o')

badtext = "No valid PyQt4 and PyOpenGL installation found.\n"

try:
    print("Object3D is installed in %s " % OBJECT3D_INSTALL_DIR)
except NameError:
    #I really do not see how this may happen but ...
    pass
