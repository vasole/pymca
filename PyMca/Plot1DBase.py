"""

Any window willing to accept 1D plugins should implement the methods defined
in this class.

The plugins will be compatible with any 1D-plot window that provides the methods:
    addCurve
    getActiveCurve
    getAllCurves
    getGraphXLimits
    getGraphYLimits
    setActiveCurve

On instantiation, this clase imports all the plugins found in the Plugins1D directory
and stores them into the attributes pluginList and pluginInstanceDict
"""
import os
import sys
import glob
import Plugins1D
DEBUG = 1

class Plot1DBase:
    def __init__(self):
        self.pluginList = []
        self.pluginInstanceDict = {}
        self.getPlugins()

    def getPlugins(self):
        """
        Import or reloads all the available plugins.
        It returns the number of plugins loaded.
        """
        directory = os.path.dirname(Plugins1D.__file__)
        if not os.path.exists(directory):
            raise IOError, "Directory:\n%s\ndoes not exist." % directory

        self.pluginList = []
        fileList = glob.glob(os.path.join(directory, "*.py"))
        targetMethod = 'getPlugin1DInstance'
        for module in fileList:
            if 1:
                pluginName = os.path.basename(module)[:-3]
                plugin = "Plugins1D." + pluginName
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
                    self.pluginInstanceDict[plugin] = sys.modules[plugin].getPlugin1DInstance(self)
                    self.pluginList.append(plugin)
            else:
                if DEBUG:
                    print "Problem importing module %s" % plugin
        return len(self.pluginList)
    
    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True):
        print "addCurve not implemented"
        return None
    
    def getActiveCurve(self):
        """
        Function to access the currently active curve.
        It returns None in case of not having an active curve.

        Output has the form:
            xvalues, yvalues, legend, dict
            where dict is a dictionnary containing curve info.
            For the time being, only the plot labels associated to the
            curve are warranted to be present under the keys xlabel, ylabel.
        """
        print "getActiveCurve not implemented"
        return None

    def getAllCurves(self):
        """
        It returns a list of the form:
            [[xvalues0, xvalues1, ..., xvaluesn],
             [yvalues0, yvalues1, ..., yvaluesn],
             [legend0,  legend1,  ..., legendn ],
             [dict0,    dict1,    ..., dictn]]
        or just an empty list.
        """
        print "getAllCurves not implemented"
        return []

    def getGraphXLimits(self):
        """
        Get the graph X limits. 
        """
        print "getGraphXLimits not implemented"
        return 0.0, 100.0

    def getGraphYLimits(self):
        """
        Get the graph Y (left) limits. 
        """
        print "getGraphYLimits not implemented"
        return 0.0, 100.0

    def setActiveCurve(self, legend):
        """
        Funtion to request the plot window to set the curve with the specified legend
        as the active curve.
        """
        print "setActiveCurve not implemented"
        return None
    
