import os
import sys

cwd = os.path.abspath(os.getcwd())
cmd = r"cd %s; pyinstaller pyinstaller.spec --noconfirm --workpath %s --distpath %s" % \
              (os.path.join(".", "package", "pyinstaller"),
               os.path.join(".", "build-" + sys.platform),
               os.path.join(".", "dist-" + sys.platform))

# patch PyOpenGL
try:
    import OpenGL
    if OpenGL.__version__ == '3.1.7':
        fname = OpenGL.__file__
        if fname[-1] == "c":
            fname = fname[:-1]
        infile = open(fname, "rb").read()
        infile = infile.replace(b'_bi + ".CArgObject",' ,
                                b'("_ctypes" if sys.version_info[:2] >= (3,12) else _bi) + ".CArgObject",')
        outfile = open(fname, "wb").write(infile)
        infile = None
        outfile = None
except ImportError:
    pass
except Exception:
    print("Cannot patch PyOpenGL")

if sys.platform.startswith("darwin"):
    if "arm64" in sys.argv:
        os.putenv("PYMCA_PYINSTALLER_TARGET_ARCH", "arm64")
    elif "universal2" in sys.argv:
        os.putenv("PYMCA_PYINSTALLER_TARGET_ARCH", "universal2")
    elif "x86_64" in sys.argv:
        os.putenv("PYMCA_PYINSTALLER_TARGET_ARCH", "x86_64")
    else:
        # let PyInstaller choose according to platform
        pass

if sys.platform.startswith("win"):
    cmd = cmd.replace(";", "&")    
result = os.system(cmd)

os.chdir(cwd)
if result:
    print("Unsuccessful command <%s>" % cmd)
sys.exit(result)
