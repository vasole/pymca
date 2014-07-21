#!python
#
#    Copyright (C) 2004-2013 V. Armando Sole - ESRF
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
__license__ = "MIT"
"""Windows-specific part of the installation"""

import os, sys, shutil

def mkshortcut(target,description,link_file,*args,**kw):
    """make a shortcut if it doesn't exist, and register its creation"""

    create_shortcut(target, description, link_file,*args,**kw)
    file_created(link_file)

def install():
    """Routine to be run by the win32 installer with the -install switch."""

    # Get some system constants
    prefix = sys.prefix

    # This does not show the console ...
    python = prefix + r'\pythonw.exe'

    # This shows it
    python_console = prefix + r'\python.exe'

    # Lookup path to common startmenu ...
    ip_dir = get_special_folder_path('CSIDL_COMMON_PROGRAMS') + r'\PyMca5'
    lib_dir = prefix+'\Lib\site-packages\PyMca5'

    if not os.path.isdir(ip_dir):
        os.mkdir(ip_dir)
        directory_created(ip_dir)

    # Create program shortcuts ...
    name = 'PyMcaMain'
    script = '"'+lib_dir+r'\%s.py"'%name
    fname = 'PyMca'
    f = ip_dir + r'\%s.lnk' % fname
    mkshortcut(python_console,name,f,script, "%HOMEDRIVE%%HOMEPATH%")

    name = 'PyMcaMain'
    script = '"'+lib_dir+r'\%s.py" -f'%name
    fname = 'PyMca Fresh Start'
    f = ip_dir + r'\%s.lnk' % fname
    mkshortcut(python_console,name,f,script, "%HOMEDRIVE%%HOMEPATH%")

    name = 'EdfFileSimpleViewer'
    script = '"'+lib_dir+r'\%s.py"'%name
    fname = 'EDF Viewer'
    f = ip_dir + r'\%s.lnk'%fname
    mkshortcut(python,name,f,script, "%HOMEDRIVE%%HOMEPATH%")

    name = 'ElementsInfo'
    script = '"'+lib_dir+r'\%s.py"'%name
    f = ip_dir + r'\%s.lnk'%name
    mkshortcut(python,name,f,script,"%HOMEDRIVE%%HOMEPATH%")

    name = 'Mca2Edf'
    script = '"'+lib_dir+r'\%s.py"'%name
    fname = 'Mca to Edf Converter'
    f = ip_dir + r'\%s.lnk'%fname
    mkshortcut(python,name,f,script,"%HOMEDRIVE%%HOMEPATH%")

    name = 'PeakIdentifier'
    script = '"'+lib_dir+r'\%s.py"'%name
    f = ip_dir + r'\%s.lnk'%name
    mkshortcut(python,name,f,script,"%HOMEDRIVE%%HOMEPATH%")

    name = 'PyMcaBatch'
    script = '"'+lib_dir+r'\%s.py"'%name
    f = ip_dir + r'\%s.lnk'%name
    mkshortcut(python_console,name,f,script,"%HOMEDRIVE%%HOMEPATH%")

    name = 'PyMcaPostBatch'
    script = '"'+lib_dir+r'\%s.py"'%name
    fname = 'RGB Correlator'
    f = ip_dir + r'\%s.lnk'%fname
    mkshortcut(python,name,f,script,"%HOMEDRIVE%%HOMEPATH%")

    name = 'QStackWidget'
    script = '"'+lib_dir+r'\%s.py"'%name
    fname = 'ROI Imaging Tool'
    f = ip_dir + r'\%s.lnk'%fname
    mkshortcut(python_console,name,f,script,"%HOMEDRIVE%%HOMEPATH%")

    name = 'QEDFStackWidget'
    script = '"'+lib_dir+r'\%s.py"'%name
    fname = 'ROI Imaging Tool(OLD)'
    f = ip_dir + r'\%s.lnk'%fname
    mkshortcut(python_console,name,f,script,"%HOMEDRIVE%%HOMEPATH%")

    name = 'ChangeLog'
    script = '"'+lib_dir+r'\%s.py" LICENSE.GPL'%name
    fname = 'License'
    f = ip_dir + r'\%s.lnk'%fname
    mkshortcut(python,name,f,script,"%HOMEDRIVE%%HOMEPATH%")

    # Create documentation shortcuts ...

def remove():
    """Routine to be run by the win32 installer with the -remove switch."""
    pass

# main()
if len(sys.argv) > 1:
    if sys.argv[1] in ['-install', 'install']:
        install()
    elif sys.argv[1] in ['-remove', 'remove']:
        remove()
    else:
        print("Script was called with option %s" % sys.argv[1])
        print("It has to be called with option -install or -remove")
