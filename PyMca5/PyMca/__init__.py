import os
import glob

def getPackages(directory):
    packages = []
    fileList = glob.glob(os.path.join(directory, "*"))
    for fileName in fileList:
        if fileName.endswith(".py") or fileName.endswith(".pyc"):
            continue
        if os.path.isdir(fileName):
            if os.path.exists(os.path.join(fileName, "__init__.py")):
                # is a package
                packages.append(fileName)
                # that may contain packages
                packages += getPackages(fileName)                
    return packages


# this is the package level directory PyMca5
baseDirectory = os.path.dirname(os.path.dirname(__file__))
__path__ += [baseDirectory]
for directory in ["PyMcaCore", "PyMcaGraph", "PyMcaGui",
                  "PyMcaIO", "PyMcaMath", "PyMcaMisc", "PyMcaPhysics"]:
    __path__ += getPackages(os.path.join(baseDirectory, directory))
