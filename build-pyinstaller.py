import os
import sys

cwd = os.path.abspath(os.getcwd())
cmd = r"cd %s; pyinstaller pyinstaller.spec --noconfirm --workpath %s --distpath %s" % \
              (os.path.join(".", "package", "pyinstaller"),
               os.path.join(".", "build-" + sys.platform),
               os.path.join(".", "dist-" + sys.platform))

if sys.platform.startswith("win"):
    cmd = cmd.replace(";", "&")    
result = os.system(cmd)

os.chdir(cwd)
if result:
    print("Unsuccessful command <%s>" % cmd)
sys.exit(result)
