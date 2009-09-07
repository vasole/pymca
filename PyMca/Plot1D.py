"""
This module can be used for plugin testing purposes as well as for doing
the bookkeeping of actual plot windows.

It implements the Plot1D interface:

    addCurve(self, x, y, legend=None, info=None, replace=False, replot=True)
    getActiveCurve(self, just_legend=False)
    getAllCurves(self, just_legend=False)
    getGraphXLimits(self)
    getGraphYLimits(self)
    removeCurve(self, legend, replot=True)
    setActiveCurve(self)
"""
import Plot1DBase

class Plot1D(Plot1DBase.Plot1DBase):
    def __init__(self):
        Plot1DBase.Plot1DBase.__init__(self)
        self.curveList = []
        self.curveDict = {}
        self.activeCurve = None
        
    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True):
        """
        Add the 1D curve given by x an y to the graph.
        """
        if legend is None:
            key = "Unnamed curve 1.1"
        else:
            key = str(legend)
        if info is None:
            info = {}
        xlabel = info.get('xlabel', 'X')
        ylabel = info.get('ylabel', 'Y')
        info['xlabel'] = str(xlabel)
        info['ylabel'] = str(ylabel)

        if replace:
            self.curveList = []
            self.curveDict = {}
            
        if key in self.curveList:
            idx = self.curveList.index(key)
            self.curveList[idx] = key
        else:
            self.curveList.append(key)
        self.curveDict[key] = [x, y, key, info]
        if len(self.curveList) == 1:
            self.activeCurve = key
        return

    def removeCurve(self, legend, replot=True):
        """
        Remove the curve associated to the supplied legend from the graph.
        The graph will be updated if replot is true.
        """
        if legend is None:
            return        
        if legend in self.curveList:
            idx = self.curveList.index(legend)
            del self.curveList[idx]
        if legend in self.curveDict.keys(): 
            del self.curveDict[legend]
        return
    
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
        if self.activeCurve not in self.curveDict.keys():
            self.activeCurve = None
        if just_legend:
            return self.activeCurve
        if self.activeCurve is None:
            return None
        else:
            return self.curveDict[self.activeCurve]

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
        output = []
        keys = self.curveDict.keys()
        for key in self.curveList:
            if key in keys:
                output.append(self.curveDict[key])
        return output

    def __getAllLimits(self):
        keys = self.curveDict.keys()
        if not len(keys):
            return 0.0, 0.0, 100., 100.
        xmin = None
        ymin = None
        xmax = None
        ymax = None
        for key in keys:
            x = self.curveDict[key][0]
            y = self.curveDict[key][1]
            if xmin is None:
                xmin = x.min()
            else:
                xmin = min(xmin, x.min())
            if ymin is None:
                ymin = y.min()
            else:
                ymin = min(ymin, y.min())
            if xmax is None:
                xmax = x.max()
            else:
                xmax = max(xmax, x.max())
            if ymax is None:
                ymax = y.max()
            else:
                ymax = max(ymax, y.max())
        return xmin, ymin, xmax, ymax

    def getGraphXLimits(self):
        """
        Get the graph X limits. 
        """
        xmin, ymin, xmax, ymax = self.__getAllLimits()
        return xmin, xmax
        

    def getGraphYLimits(self):
        """
        Get the graph Y (left) limits. 
        """
        xmin, ymin, xmax, ymax = self.__getAllLimits()
        return ymin, ymax

    def setActiveCurve(self, legend):
        """
        Funtion to request the plot window to set the curve with the specified
        legend as the active curve.
        It returns the active curve legend.
        """
        key = str(legend)
        if key in self.curveDict.keys():
            self.activeCurve = key
        return self.activeCurve

if __name__ == "__main__":
    import numpy
    x = numpy.arange(100.)
    y = x * x
    plot = Plot1D()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x+100, -x*x)
    print "Active curve = ", plot.getActiveCurve()
    print "X Limits = ",     plot.getGraphXLimits()
    print "Y Limits = ",     plot.getGraphYLimits()
    print "All curves = ",   plot.getAllCurves()
    plot.removeCurve("dummy")
    print "All curves = ",   plot.getAllCurves()

    
