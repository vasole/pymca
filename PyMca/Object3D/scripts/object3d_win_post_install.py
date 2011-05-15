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
    ip_dir = get_special_folder_path('CSIDL_COMMON_PROGRAMS') + r'\Object3D'
    lib_dir = prefix+'\Lib\site-packages\Object3D'

    if not os.path.isdir(ip_dir):
        os.mkdir(ip_dir)
        directory_created(ip_dir)

    # Create program shortcuts ...
    name = 'Object3D Scene'
    module = 'SceneGLWindow'
    script = '"'+lib_dir+r'\%s.py"' % module
    f = ip_dir + r'\%s.lnk'%name
    mkshortcut(python_console,name,f,script)

    """
    name = 'ChangeLog'
    script = '"'+lib_dir+r'\%s.py" LICENSE.GPL'%name
    fname = 'License'
    f = ip_dir + r'\%s.lnk'%fname
    mkshortcut(python,name,f,script)
    """

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
