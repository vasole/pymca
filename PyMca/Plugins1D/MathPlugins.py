import Plugin1DBase
import numpy
try:
    import PyMca.PyMca_Icons as PyMca_Icons
    import PyMca.SimpleMath as SimpleMath
except ImportError:
    #This happens in frozen versions
    import PyMca_Icons
    import SimpleMath

swapsign = PyMca_Icons.swapsign
derive = PyMca_Icons.derive

class MathPlugins(Plugin1DBase.Plugin1DBase):
    def __init__(self, plotWindow, **kw):
       Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)
       self.methodDict = {'Invert':[self.invert,
                                    "Multiply active curve by -1",
                                    swapsign],
                          'Derivate':[self.derivate,
                                    "Derivate zoomed active curve",
                                    derive]}
       self.simpleMath = SimpleMath.SimpleMath()

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

    def invert(self):
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        x, y, legend, info = activeCurve [0:4]
        operations = info.get("operations", [])
        operations.append("Invert")
        info['operations'] = operations
        legend = "-("+legend+")"
        self.addCurve(x, -y, legend=legend, info=info, replot=True)

    def derivate(self):
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        x, y, legend, info = activeCurve [0:4]
        xlimits=self.getGraphXLimits()
        x, y = self.simpleMath.derivate(x, y, xlimits=xlimits)
        info['ylabel'] = info['ylabel'] + "'"
        operations = info.get("operations", [])
        operations.append("derivate")
        info['operations'] = operations
        legend = legend+"'"
        self.addCurve(x, y, legend=legend, info=info, replot=True)
        

MENU_TEXT = "Built-in Math"
def getPlugin1DInstance(plotWindow, **kw):
    ob = MathPlugins(plotWindow)
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
