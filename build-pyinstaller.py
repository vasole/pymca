import os
import sys

cwd = os.path.abspath(os.getcwd())
cmd = r"cd %s; pyinstaller pyinstaller.spec --noconfirm" % \
              os.path.join(".", "package", "pyinstaller")

if sys.platform.startswith("win"):
    cmd = cmd.replace(";", "&")    
result = os.system(cmd)

os.chdir(cwd)
if result:
    print("Unsuccessful command <%s>" % cmd)
sys.exit(result)
