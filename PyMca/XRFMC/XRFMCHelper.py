import sys
import os
import tempfile
try:
    from PyMca.XRFMC import XMSOParser
except ImportError:
    print("Trying to import XMSOParser directly")
    import XMSOParser

getXMSOFileFluorescenceInformation =\
                        XMSOParser.getXMSOFileFluorescenceInformation

XMIMSIM_PYMCA = None
if sys.platform == "win32":
    try:
            
        # try to get the installation directory from the registry
        if sys.version < '3.0':
            import _winreg as winreg
        else:
            import winreg
        HKLM = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        try:
            # 32 bit
            softwareKey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "Software\XMI-MSIM")
        except:
            try:
                # 64 bit
                softwareKey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "Software\Wow6432Node\XMI-MSIM")
            except:
                # XMI-MSIM not installed ...
                softwareKey = None
        if softwareKey is not None:
            subKeyName = "InstallationDirectory"
            value = winreg.QueryValueEx(softwareKey, subKeyName)
            pathToExecutable = os.path.join(value[0], "bin", "xmimsim-pymca.exe")
            if not os.path.exists(pathToExecutable):
                pathToExecutable = None
            XMIMSIM_PYMCA = pathToExecutable
    except:
        # this cannot afford failing
        pass

def getScriptFile(pathToExecutable=None, args=None, name=None):
    if pathToExecutable is None:
        pathToExecutable = XMIMSIM_PYMCA
    if pathToExecutable is None:
        raise ValueError("Path to xmimsim-pymca needed")
    if not os.path.exists(pathToExecutable):
        raise IOError("xmimsim-pymca executable does not exist")
    if args is None:
        args = []
    executable = os.path.basename(pathToExecutable)
    if not executable.startswith("xmimsim-pymca"):
        if sys.platform == "win32":
            raise ValueError("Path to xmimsim-pymca.exe needed")
        else:
            raise ValueError("Path to xmimsim-pymca needed")
    xmimsim_directory = os.path.dirname(pathToExecutable)
    if os.path.basename(xmimsim_directory).lower() == "bin":
        xmimsim_directory = os.path.dirname(xmimsim_directory)

    binDir = os.path.join(xmimsim_directory, "bin")
    libDir = os.path.join(xmimsim_directory, "lib")
    gtk2Dir = os.path.join(xmimsim_directory, "GTK2")
    if sys.platform == "win32":
        path = os.getenv("PATH")
        txt = "echo off\n"
        txt += "set PATH=%s;%s;%s;%s\n" % (binDir, libDir, gtk2Dir, os.getenv("PATH"))
        txt += "%s " % executable
        if len(args):
            for arg in args:
                txt += arg + " ";
            txt += "\n"
        else:
            txt += "%*"
        if name is None:
            handle, fullPath = tempfile.mkstemp(suffix=".bat", prefix="pymca", text=False)
            os.write(handle, txt)
            os.close(handle)
        else:
            fullPath = name
            if not fullPath.endswith(".bat"):
                fullPath = name + ".bat"
            f = open(fullPath, "wb")
            f.write(txt)
            f.close()
    else:
        raise NotImplemented("Sorry, platform not implemented yet")
    return fullPath

def getOutputFileNames(fitFile, outputDir=None):
    if outputDir is None:
        outputDir = os.path.dirname(fitFile)
    ddict = {}
    newFile = os.path.join(outputDir, os.path.basename(fitFile))
    if newFile.lower().endswith(".fit"):
        rootName = newFile[:-4]
    else:
        rootName = newFile
    scriptName = rootName + "_script"
    csvName = rootName + ".csv"
    speName = rootName + ".spe"
    xmsoName = rootName + ".xmso"
    if sys.platform == 'win32':
        scriptName = scriptName + ".bat"
    fitName = rootName + ".fit"
    ddict={}
    ddict['fit'] = rootName + ".fit"
    ddict['script'] = scriptName
    ddict['csv'] = csvName
    ddict['spe'] = speName
    ddict['xmso'] = xmsoName
    return ddict

def getBasicSubprocessCommand(fitFile, outputDir=None, xmimsim_pymca=None):
     ddict = getOutputFileNames(fitFile, outputDir)
     scriptFile = getScriptFile(pathToExecutable=xmimsim_pymca,
                                    name=ddict['script'])
     if ddict['fit'] != fitFile:
         if outputDir is None:
             # this should never happen
            raise ValueError("Inconsistent internal behaviour!")
         # recreate input in output directory
         new = ConfigDict.ConfigDict()
         new.read(fitFile)
         if os.path.exists(ddict['fit']):
             os.remove(ddict['fit'])
         new.write(ddict['fit'])
         new = None
     speName = ddict['spe']
     csvName = ddict['csv']
     newFitFile = ddict['fit']
     xmsoName = ddict['xmsoName']
     args = [scriptFile,
            #"--enable-single-run",
            "--verbose",
            "--spe-file=%s" % speName,
            "--csv-file=%s" % csvName,
            #"--enable-roi-normalization",
            #"--disable-roi-normalization", #default
            #"--enable-pile-up"
            #"--disable-pile-up" #default
            #"--enable-poisson",
            #"--disable-poisson", #default no noise
            #"--set-threads=2", #overwrite default maximum
            newFitFile,
            xmsoName]
     return args

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(getScriptFile(sys.argv[1]))
    else:
        print("Usage:")
        print("python XRFMCCommand.py path_to_xmimsim-pymca_executable")
        print("The returned file name is to be deleted!!!")

