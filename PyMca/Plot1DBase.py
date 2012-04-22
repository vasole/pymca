#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
__author__ = "V.A. Sole - ESRF Software Group"
"""

Any window willing to accept 1D plugins should implement the methods defined
in this class.

The plugins will be compatible with any 1D-plot window that provides the methods:
    addCurve
    removeCurve
    getActiveCurve
    getAllCurves
    getGraphXLimits
    getGraphYLimits
    setActiveCurve

On instantiation, this clase imports all the plugins found in the PyMcaPlugins
directory and stores them into the attributes pluginList and pluginInstanceDict

"""
import os
import sys
import glob
PLUGINS_DIR = None
try:
    if os.path.exists(os.path.join(os.path.dirname(__file__), "PyMcaPlugins")):
        from PyMca import PyMcaPlugins
        PLUGINS_DIR = os.path.dirname(PyMcaPlugins.__file__)
    else:
        directory = os.path.dirname(__file__)
        while True:
            if os.path.exists(os.path.join(directory, "PyMcaPlugins")):
                PLUGINS_DIR = os.path.join(directory, "PyMcaPlugins")
                break
            directory = os.path.dirname(directory)
            if len(directory) < 5:
                break
except:
    pass
DEBUG = 0

class Plot1DBase(object):
    def __init__(self):
        self.__pluginDirList = []
        self.pluginList = []
        self.pluginInstanceDict = {}
        self.getPlugins()

    def setPluginDirectoryList(self, dirlist):
        for directory in dirlist:
            if not os.path.exists(directory):
                raise IOError("Directory:\n%s\ndoes not exist." % directory)                

        self.__pluginDirList = dirlist

    def getPluginDirectoryList(self):
        return self.__pluginDirList

    def getPlugins(self):
        """
        Import or reloads all the available plugins.
        It returns the number of plugins loaded.
        """
        if self.__pluginDirList == []:
           self.__pluginDirList = [PLUGINS_DIR] 
        self.pluginList = []
        for directory in self.__pluginDirList:
            if directory is None:
                continue
            if not os.path.exists(directory):
                raise IOError("Directory:\n%s\ndoes not exist." % directory)

            fileList = glob.glob(os.path.join(directory, "*.py"))
            targetMethod = 'getPlugin1DInstance'
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
                    if directory == PLUGINS_DIR:
                        plugin = "PyMcaPlugins." + pluginName
                    else:
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
                            reload(sys.modules[plugin])
                    else:
                        __import__(plugin)
                    if hasattr(sys.modules[plugin], targetMethod):
                        self.pluginInstanceDict[plugin] = \
                                sys.modules[plugin].getPlugin1DInstance(self)
                        self.pluginList.append(plugin)
                except:
                    if DEBUG:
                        print("Problem importing module %s" % plugin)
                        raise
        return len(self.pluginList)
    
    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True):
        """
        Add the 1D curve given by x an y to the graph.
        """
        print("addCurve not implemented")
        return None

    def removeCurve(self, legend, replot=True):
        """
        Remove the curve associated to the supplied legend from the graph.
        The graph will be updated if replot is true.
        """
        print("removeCurve not implemented")
        return None
    
    def getActiveCurve(self, just_legend=False):
        """
        Function to access the currently active curve.
        It returns None in case of not having an active curve.

        Default output has the form:
            xvalues, yvalues, legend, dict
            where dict is a dictionnary containing curve info.
            For the time being, only the plot labels associated to the
            curve are warranted to be present under the keys xlabel, ylabel.

        If just_legend is True:
            The legend of the active curve (or None) is returned.
        """
        print("getActiveCurve not implemented")
        return None

    def getAllCurves(self, just_legend=False):
        """
        If just_legend is False:
            It returns a list of the form:
                [[xvalues0, yvalues0, legend0, dict0],
                 [xvalues1, yvalues1, legend1, dict1],
                 [...],
                 [xvaluesn, yvaluesn, legendn, dictn]]
            or just an empty list.
        If just_legend is True:
            It returns a list of the form:
                [legend0, legend1, ..., legendn]
            or just an empty list.
        """
        print("getAllCurves not implemented")
        return []

    def getGraphXLimits(self):
        """
        Get the graph X limits. 
        """
        print("getGraphXLimits not implemented")
        return 0.0, 100.0

    def getGraphYLimits(self):
        """
        Get the graph Y (left) limits. 
        """
        print("getGraphYLimits not implemented")
        return 0.0, 100.0

    def setActiveCurve(self, legend):
        """
        Funtion to request the plot window to set the curve with the specified legend
        as the active curve.
        """
        print("setActiveCurve not implemented")
        return None

    def setGraphTitle(self, title):
        print("setGraphTitle not implemented")

    def setGraphXTitle(self, title):
        print("setGraphXTitle not implemented")

    def setGraphYTitle(self, title):
        print("setGraphYTitle not implemented")

    def getGraphTitle(self):
        print("getGraphTitle not implemented")
        return "Title"

    def getGraphXTitle(self):
        print("getGraphXTitle not implemented")
        return "X"

    def getGraphYTitle(self):
        print("getGraphYTitle not implemented")
        return "Y"
