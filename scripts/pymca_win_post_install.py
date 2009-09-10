#!python
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
    ip_dir = get_special_folder_path('CSIDL_COMMON_PROGRAMS') + r'\PyMca'
    lib_dir = prefix+'\Lib\site-packages\PyMca'

    if not os.path.isdir(ip_dir):
        os.mkdir(ip_dir)
        directory_created(ip_dir)

    # Create program shortcuts ...
    name = 'PyMcaMain'
    script = '"'+lib_dir+r'\%s.py"'%name
    fname = 'PyMca'
    f = ip_dir + r'\%s.lnk' % fname 
    mkshortcut(python_console,name,f,script)

    name = 'PyMcaMain'
    script = '"'+lib_dir+r'\%s.py" -f'%name
    fname = 'PyMca Fresh Start'
    f = ip_dir + r'\%s.lnk' % fname
    mkshortcut(python_console,name,f,script)

    name = 'EdfFileSimpleViewer'
    script = '"'+lib_dir+r'\%s.py"'%name
    fname = 'EDF Viewer'
    f = ip_dir + r'\%s.lnk'%fname
    mkshortcut(python,name,f,script)

    name = 'ElementsInfo'
    script = '"'+lib_dir+r'\%s.py"'%name
    f = ip_dir + r'\%s.lnk'%name
    mkshortcut(python,name,f,script)

    name = 'Mca2Edf'
    script = '"'+lib_dir+r'\%s.py"'%name
    fname = 'Mca to Edf Converter'
    f = ip_dir + r'\%s.lnk'%fname
    mkshortcut(python,name,f,script)

    name = 'PeakIdentifier'
    script = '"'+lib_dir+r'\%s.py"'%name
    f = ip_dir + r'\%s.lnk'%name
    mkshortcut(python,name,f,script)

    name = 'PyMcaBatch'
    script = '"'+lib_dir+r'\%s.py"'%name
    f = ip_dir + r'\%s.lnk'%name
    mkshortcut(python_console,name,f,script)

    name = 'PyMcaPostBatch'
    script = '"'+lib_dir+r'\%s.py"'%name
    fname = 'RGB Correlator'
    f = ip_dir + r'\%s.lnk'%fname
    mkshortcut(python,name,f,script)

    name = 'QEDFStackWidget'
    script = '"'+lib_dir+r'\%s.py"'%name
    fname = 'ROI Imaging Tool'
    f = ip_dir + r'\%s.lnk'%fname
    mkshortcut(python_console,name,f,script)

    name = 'ChangeLog'
    script = '"'+lib_dir+r'\%s.py" LICENSE.GPL'%name
    fname = 'License'
    f = ip_dir + r'\%s.lnk'%fname
    mkshortcut(python,name,f,script)

    # Create documentation shortcuts ...

def remove():
    """Routine to be run by the win32 installer with the -remove switch."""
    pass

# main()
if len(sys.argv) > 1:
    if sys.argv[1] == '-install':
        install()
    elif sys.argv[1] == '-remove':
        remove()
    else:
        print "Script was called with option %s" % sys.argv[1]
