import os
import sys

cwd = os.path.abspath(os.getcwd())
cmd = r"cd %s; pyinstaller pyinstaller.spec --noconfirm --workpath %s --distpath %s" % \
              (os.path.join(".", "package", "pyinstaller"),
               os.path.join(".", "build-" + sys.platform),
               os.path.join(".", "dist-" + sys.platform))

if sys.platform.startswith("darwin"):
    if "arm64" in sys.argv:
        cmd += " --target-arch arm64"
    elif "universal2" in sys.argv:
        cmd += " --target-arch universal2"
    else:
        cmd += " --target-arch x86_64"

if sys.platform.startswith("win"):
    cmd = cmd.replace(";", "&")    
result = os.system(cmd)

os.chdir(cwd)
if result:
    print("Unsuccessful command <%s>" % cmd)
sys.exit(result)
