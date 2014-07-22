import os
import glob

def getPackages(directory):
    packages = []
    fileList = glob.glob(os.path.join(directory, "*", "__init__.py"))
    for fileName in fileList:
        dirName = os.path.dirname(fileName)
        packages.append(dirName)
        packages += getPackages(dirName)
    return packages


# this is the package level directory PyMca5
baseDirectory = os.path.dirname(os.path.dirname(__file__))
__path__ += [baseDirectory]
for directory in ["PyMcaCore", "PyMcaGraph", "PyMcaGui",
                  "PyMcaIO", "PyMcaMath", "PyMcaMisc", "PyMcaPhysics"]:
    tmpDir = os.path.join(baseDirectory, directory)
    if os.path.exists(os.path.join(tmpDir, "__init__.py")):
        __path__ += [tmpDir]
    __path__ += getPackages(tmpDir)
