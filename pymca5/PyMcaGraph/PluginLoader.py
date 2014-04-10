#/*##########################################################################
# Copyright (C) 2004-2014 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Software Group"
__license__ = "LGPL"
__doc__ = """

Class to handle loading of plugins according to target method.

On instantiation, this clase imports all the plugins found in the PLUGINS_DIR
directory and stores them into the attributes pluginList and pluginInstanceDict

"""
import os
import sys
import glob

PLUGINS_DIR = None

DEBUG = 0

class PluginLoader(object):
    def __init__(self, method=None, directoryList=None):
        self._pluginDirList = []
        self.pluginList = []
        self.pluginInstanceDict = {}
        self.getPlugins(method=method, directoryList=directoryList)

    def setPluginDirectoryList(self, dirlist):
        """
        :param dirlist: Set directories to search for Plot1D plugins
        :type dirlist: list
        """
        for directory in dirlist:
            if not os.path.exists(directory):
                raise IOError("Directory:\n%s\ndoes not exist." % directory)                
        self._pluginDirList = dirlist

    def getPluginDirectoryList(self):
        """
        :return dirlist: List of directories for searching plugins
        """
        return self._pluginDirList

    def getPlugins(self, method=None, directoryList=None):
        """
        Import or reloads all the available plugins with the target method
        :param method: The method to be searched for.
        :type method: string, default "getPlugin1DInstance"
        :param directoryList: The list of directories for the search.
        :type directoryList: list or None (default).
        :return: The number of plugins loaded.
        """
        if method is None:
            method = 'getPlugin1DInstance'
        targetMethod = method
        if directoryList in [None, [] ]:
            directoryList = self._pluginDirList
            if directoryList in [None, []]:
                directoryList = [self.PLUGINS_DIR]
        if DEBUG:
            print("method: %s" % targetMethod)
            print("directoryList: %s" % directoryList)
        self._pluginDirList = directoryList
        self.pluginList = []
        for directory in self._pluginDirList:
            if directory is None:
                continue
            if not os.path.exists(directory):
                raise IOError("Directory:\n%s\ndoes not exist." % directory)

            fileList = glob.glob(os.path.join(directory, "*.py"))
            # prevent unnecessary imports
            moduleList = []
            for fname in fileList:
                # in Python 3, rb implies bytes and not strings
                f = open(fname, 'r')
                lines = f.readlines()
                f.close()
                f = None
                for line in lines:
                    if line.startswith("def"):
                        if line.split(" ")[1].startswith(targetMethod):
                            moduleList.append(fname)
                            break
            for module in moduleList:
                try:
                    pluginName = os.path.basename(module)[:-3]
                    if DEBUG:
                        print("pluginName %s" % pluginName)
                    plugin = pluginName
                    if directory not in sys.path:
                        sys.path.insert(0, directory)
                    if pluginName in self.pluginList:
                        idx = self.pluginList.index(pluginName)
                        del self.pluginList[idx]
                    if plugin in self.pluginInstanceDict.keys():
                        del self.pluginInstanceDict[plugin]
                    if plugin in sys.modules:
                        if hasattr(sys.modules[plugin], targetMethod):
                            if sys.version.startswith('3.3'):
                                import imp
                                imp.reload(sys.modules[plugin])
                            else:
                                reload(sys.modules[plugin])
                    else:
                        __import__(plugin)
                    if hasattr(sys.modules[plugin], targetMethod):
                        theCall = getattr(sys.modules[plugin], targetMethod)
                        self.pluginInstanceDict[plugin] = theCall(self)
                        self.pluginList.append(plugin)
                except:                    
                    if DEBUG:
                        print("Problem importing module %s" % plugin)
                        raise
        return len(self.pluginList)

def main(targetMethod, directoryList):
    loader = PluginLoader()
    n = loader.getPlugins(targetMethod, directoryList)
    print("Loaded %d plugins" % n)
    for m in loader.pluginList:
        print("Module %s" % m)
        module = sys.modules[m]
        if hasattr(module, 'MENU_TEXT'):
            text = module.MENU_TEXT
        else:
            text = os.path.basename(module.__file__)
            if text.endswith('.pyc'):
                text = text[:-4]
            elif text.endswith('.py'):
                text = text[:-3]
        print("\tMENU TEXT: %s" % text)  
        methods = loader.pluginInstanceDict[m].getMethods()
        if not len(methods):
            continue
        for method in methods:
            print("\t\t %s" % method)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        targetMethod  = sys.argv[1]
        directoryList = sys.argv[2:len(sys.argv)]
    elif len(sys.argv) > 1:
        targetMethod = None
        directoryList = sys.argv[1:len(sys.argv)]
    else:
        print("Usage: python PluginLoader.py [targetMethod] directory")
    main(targetMethod, directoryList)
