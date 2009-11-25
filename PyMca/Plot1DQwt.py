import Plot1D
import QtBlissGraph
qt = QtBlissGraph.qt

class Plot1DQwt(qt.QWidget, Plot1D.Plot1D):
    def __init__(self, parent=None,**kw):
        qt.QWidget.__init__(self, parent)
        Plot1D.Plot1D.__init__(self)
        mainLayout = qt.QVBoxLayout(self)
        mainLayout.setMargin(0)
        mainLayout.setSpacing(0)
        self.graph = QtBlissGraph.QtBlissGraph(self, **kw)
        mainLayout.addWidget(self.graph)
        self.curveList = []
        self.curveDict = {}
        self.activeCurve = None
        self._logY = False

    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True):
        """
        Add the 1D curve given by x an y to the graph.
        """
        Plot1D.Plot1D.addCurve(self, x, y, legend=legend,
                               info=info, replace=replace, replot=replot)        
        if replot:
            if replace:
                self.replot('REPLACE')
            else:
                self.replot('ADD')

    def removeCurve(self, legend, replot=True):
        """
        Remove the curve associated to the supplied legend from the graph.
        The graph will be updated if replot is true.
        """
        Plot1D.Plot1D.removeCurve(self, legend, replot=replot)
        if legend in self.graph.curves.keys():
            self.graph.removeCurve(legend)
        self._updateActiveCurve()
        if replot:
            self.graph.replot()
        return
        
    def replot(self, mode=None):
        if mode in [None, 'REPLACE']:
            if mode == 'REPLACE':
                self.graph.clearCurves()
            for curve in self.curveList:
                x, y, legend, info = self.curveDict[curve] 
                self.graph.newCurve(curve, x, y, logfilter=self._logY)
            self._updateActiveCurve()
            self.graph.replot()
            return
        
        if mode.upper() == 'ADD':
            currentCurves = self.graph.curves.keys()
            for curve in self.curveList:
                if curve not in currentCurves:
                    x, y, legend, info = self.curveDict[curve] 
                    self.graph.newCurve(curve, x, y, logfilter=self._logY)
            self._updateActiveCurve()
            self.graph.replot()
            return

    def _updateActiveCurve(self):
        self.activeCurve = self.getActiveCurve(just_legend=True)
        if self.activeCurve not in self.curveList:
            self.activeCurve = None
        if self.activeCurve is None:
            if len(self.curveList):
                self.activeCurve = self.curveList[0]
        if self.activeCurve is not None:
            self.graph.setActiveCurve(self.activeCurve)            

    def getGraphXLimits(self):
        """
        Get the graph X limits. 
        """
        xmin, ymin, xmax, ymax = self.graph.getX1AxisLimits()
        return xmin, xmax
        

    def getGraphYLimits(self):
        """
        Get the graph Y (left) limits. 
        """
        xmin, ymin, xmax, ymax = self.graph.getY1AxisLimits()
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
            self.graph.setActiveCurve(key)
        return self.activeCurve


if __name__ == "__main__":
    import numpy
    x = numpy.arange(100.)
    y = x * x
    app = qt.QApplication([])
    plot = Plot1DQwt(uselegendmenu=True)
    plot.show()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x+100, -x*x)
    print "Active curve = ", plot.getActiveCurve()
    print "X Limits = ",     plot.getGraphXLimits()
    print "Y Limits = ",     plot.getGraphYLimits()
    print "All curves = ",   plot.getAllCurves()
    plot.removeCurve("dummy")
    print "All curves = ",   plot.getAllCurves()
    app.exec_()

    
