import Plugin1DBase
import numpy
try:
    import PyMca.PyMca_Icons as PyMca_Icons
except ImportError:
    #This happens in frozen versions
    import PyMca_Icons


class NormalizationPlugins(Plugin1DBase.Plugin1DBase):
    def __init__(self, plotWindow, **kw):
        Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)
        self.methodDict = {'y/max(y)':[self.toMaximum,
                        "Normalize to maximum",
                                       None],
                '(y-min(y))/(max(y)-min(y))':[self.offsetAndMaximum,
                       "Subtract offset and normalize to new maximum",
                                       None],
                '(y-min(y))/trapz(max(y)-min(y),x)':[self.offsetAndArea,
                       "Subtract offset and normalize to integrated are",
                                       None],
                '(y-min(y))/sum(max(y)-min(y))':[self.offsetAndCounts,
                       "Subtract offset and normalize to counts",
                                       None]}
        
    #Methods to be implemented by the plugin
    def getMethods(self, plottype=None):
        """
        A list with the NAMES  associated to the callable methods
        that are applicable to the specified plot.

        Plot type can be "SCAN", "MCA", None, ...        
        """
        names = self.methodDict.keys()
        names.sort()
        return names

    def getMethodToolTip(self, name):
        """
        Returns the help associated to the particular method name or None.
        """
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        """
        Returns the pixmap associated to the particular method name or None.
        """
        return self.methodDict[name][2]

    def applyMethod(self, name):
        """
        The plugin is asked to apply the method associated to name.
        """
        apply(self.methodDict[name][0])
        return

    def toMaximum(self):
        curves = self.getAllCurves()
        nCurves = len(curves)
        if not nCurves:
            return
        i = 0
        for curve in curves:
            x, y, legend, info = curve[0:4]
            try:
                y = y/y.max()
            except:
                continue
            if i == 0:
                replace = True
                replot = True
                i = 1
            else:
                replot = False
                replace = False
            self.addCurve(x, y,
                          legend=legend,
                          info=info,
                          replot=replot,
                          replace=replace)                
        self.addCurve(x, y,
                          legend=legend,
                          info=info,
                          replot=True,
                          replace=False)
        
    def offsetAndMaximum(self):
        curves = self.getAllCurves()
        nCurves = len(curves)
        if not nCurves:
            return
        i = 0
        for curve in curves:
            x, y, legend, info = curve[0:4]
            try:
                y = y - y.min()
                y = y/y.max()
            except:
                continue
            if i == 0:
                replace = True
                replot = True
                i = 1
            else:
                replot = False
                replace = False
            self.addCurve(x, y,
                          legend=legend,
                          info=info,
                          replot=replot,
                          replace=replace)                
        self.addCurve(x, y,
                          legend=legend,
                          info=info,
                          replot=True,
                          replace=False)

    def offsetAndCounts(self):
        curves = self.getAllCurves()
        nCurves = len(curves)
        if not nCurves:
            return
        i = 0
        for curve in curves:
            x, y, legend, info = curve[0:4]
            try:
                y = y - y.min()
                y = y/y.sum()
            except:
                continue
            if i == 0:
                replace = True
                replot = True
                i = 1
            else:
                replot = False
                replace = False
            self.addCurve(x, y,
                          legend=legend,
                          info=info,
                          replot=replot,
                          replace=replace)                
        self.addCurve(x, y,
                          legend=legend,
                          info=info,
                          replot=True,
                          replace=False)

    def offsetAndArea(self):
        curves = self.getAllCurves()
        nCurves = len(curves)
        if not nCurves:
            return
        i = 0
        for curve in curves:
            x, y, legend, info = curve[0:4]
            try:
                y = y - y.min()
                y = y/numpy.trapz(y, x)
            except:
                continue
            if i == 0:
                replace = True
                replot = True
                i = 1
            else:
                replot = False
                replace = False
            self.addCurve(x, y,
                          legend=legend,
                          info=info,
                          replot=replot,
                          replace=replace)                
        self.addCurve(x, y,
                          legend=legend,
                          info=info,
                          replot=True,
                          replace=False)


MENU_TEXT = "Normalization"
def getPlugin1DInstance(plotWindow, **kw):
    ob = NormalizationPlugins(plotWindow)
    return ob

if __name__ == "__main__":
    from PyMca import Plot1D
    x = numpy.arange(100.)
    y = x * x
    plot = Plot1D.Plot1D()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x+100, -x*x)
    plugin = getPlugin1DInstance(plot)
    for method in plugin.getMethods():
        print method, ":", plugin.getMethodToolTip(method)
    plugin.applyMethod(plugin.getMethods()[0])
    curves = plugin.getAllCurves()
    for curve in curves:
        print curve[2]
    print "LIMITS = ", plugin.getGraphYLimits()
