try:
    import Plugin1DBase
except ImportError:
    from . import Plugin1DBase
import numpy

try:
    from PyMca import SGWindow
    from PyMca import SNIPWindow
    import PyMca.PyMca_Icons as PyMca_Icons
except ImportError:
    print("Plugin importing from somewhere else")
    import SGWindow
    import SNIPWindow
    import PyMca_Icons

class BackgroundScanPlugin(Plugin1DBase.Plugin1DBase):
    def __init__(self, plotWindow, **kw):
        Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)
        SGtext  = "Replace active curve by a\n"
        SGtext += "Savitsky-Golay treated one."
        SNIP1Dtext  = "Replace active curve by a\n"
        SNIP1Dtext += "SNIP1D treated one."
        self.methodDict = {}
        function = self.replaceActiveCurveWithSavitzkyGolayFiltering
        info = SGtext
        icon = PyMca_Icons.substract
        self.methodDict["Savitzky-Golay Filtering"] =[function,
                                                      info,
                                                      icon]
        function = self.subtract1DSnipBackgroundFromActiveCurve
        info = SNIP1Dtext
        self.methodDict["Subtract SNIP 1D Background"] =[function,
                                                      info,
                                                      icon]
        function = self.deglitchActiveCurveWith1DSnipBackground
        info  = "Smooth and replace current curve\n"
        info += "by its SNIP1D background."
        self.methodDict["Deglitch with SNIP 1D Background"] =[function,
                                                              info,
                                                              PyMca_Icons.smooth]
        self.__methodKeys = ["Subtract SNIP 1D Background",
                             "Savitzky-Golay Filtering",
                             "Deglitch with SNIP 1D Background"]
        self.subtract1DSnipParameters = None
        self.deglitch1DSnipParameters = None
        
    #Methods to be implemented by the plugin
    def getMethods(self, plottype=None):
        """
        A list with the NAMES  associated to the callable methods
        that are applicable to the specified plot.

        Plot type can be "SCAN", "MCA", None, ...        
        """
        if 0:
            names = self.methodDict.keys()
            names.sort()
            return names
        else:
            return self.__methodKeys

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
        self.methodDict[name][0]()
        return

    def replaceActiveCurveWithSavitzkyGolayFiltering(self):
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        x, spectrum, legend, info = activeCurve
        snipWindow = SGWindow.SGDialog(None,
                                           spectrum, x=x)
        snipWindow.graph.setGraphXTitle(info['xlabel'])
        snipWindow.graph.setGraphYTitle(info['ylabel'])
        #snipWindow.setModal(True)
        snipWindow.show()
        ret = snipWindow.exec_()
        if ret:
            ydata = snipWindow.parametersWidget.background
            xdata = snipWindow.parametersWidget.xValues
            operations = info.get("operations", [])
            operations.append("SG Filtered")
            info['operations'] = operations
            self.removeCurve(legend, replot=False)
            self.addCurve(xdata, ydata, legend=legend, info=info, replot=True)

    def subtract1DSnipBackgroundFromActiveCurve(self, smooth=False):
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        x, spectrum, legend, info = activeCurve
        snipWindow = SNIPWindow.SNIPDialog(None,
                                           spectrum, x=x, smooth=False)
        if self.subtract1DSnipParameters is not None:
            snipWindow.setParameters(self.subtract1DSnipParameters)
        snipWindow.graph.setGraphXTitle(info['xlabel'])
        snipWindow.graph.setGraphYTitle(info['ylabel'])
        snipWindow.show()
        ret = snipWindow.exec_()
        if ret:
            ydata = snipWindow.parametersWidget.spectrum -\
                    snipWindow.parametersWidget.background
            xdata = snipWindow.parametersWidget.xValues
            operations = info.get("operations", [])
            operations.append("SNIP Background Removal")
            info['operations'] = operations
            self.removeCurve(legend, replot=False)
            self.addCurve(xdata, ydata, legend=legend +" Net", info=info, replot=True)
            self.subtract1DSnipParameters = snipWindow.getParameters()

    def deglitchActiveCurveWith1DSnipBackground(self):
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        x, spectrum, legend, info = activeCurve
        snipWindow = SNIPWindow.SNIPDialog(None,
                                           spectrum, x=x, smooth=True)
        if self.deglitch1DSnipParameters is not None:
            snipWindow.setParameters(self.deglitch1DSnipParameters)
        snipWindow.graph.setGraphXTitle(info['xlabel'])
        snipWindow.graph.setGraphYTitle(info['ylabel'])
        snipWindow.show()
        ret = snipWindow.exec_()
        if ret:
            ydata = snipWindow.parametersWidget.background
            xdata = snipWindow.parametersWidget.xValues
            operations = info.get("operations", [])
            operations.append("SNIP Deglith")
            info['operations'] = operations
            self.removeCurve(legend, replot=False)
            self.addCurve(xdata, ydata, legend=legend, info=info, replot=True)
            self.deglitch1DSnipParameters = snipWindow.getParameters()

MENU_TEXT = "Background subtraction tools"
def getPlugin1DInstance(plotWindow, **kw):
    ob = BackgroundScanPlugin(plotWindow)
    return ob

if __name__ == "__main__":
    from PyMca import PyMcaQt as qt
    app = qt.QApplication([])
    from PyMca import Plot1D
    x = numpy.arange(100.)
    y = x * x
    plot = Plot1D.Plot1D()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x+100, -x*x)
    plugin = getPlugin1DInstance(plot)
    for method in plugin.getMethods():
        print(method, ":", plugin.getMethodToolTip(method))
    plugin.applyMethod(plugin.getMethods()[0])
    curves = plugin.getAllCurves()
    for curve in curves:
        print(curve[2])
    print("LIMITS = ", plugin.getGraphYLimits())
    #app = qt.QApplication()
