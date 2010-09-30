import Plot1DWindowBase
import QtBlissGraph
qt = QtBlissGraph.qt

class Plot1DQwt(Plot1DWindowBase.Plot1DWindowBase):
    def __init__(self, parent=None,**kw):
        Plot1DWindowBase.Plot1DWindowBase.__init__(self, parent, **kw)
        mainLayout = self.layout()
        self.graph = QtBlissGraph.QtBlissGraph(self, **kw)
        mainLayout.addWidget(self.graph)
        self.newCurve = self.addCurve
        self.setTitle = self.graph.setTitle
        self.activeCurve = None
        self._logY = False

    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True, **kw):
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
        if kw.has_key('xlabel'):
            info['xlabel'] = kw['xlabel'] 
        if kw.has_key('ylabel'):
            info['ylabel'] = kw['ylabel'] 
        Plot1DWindowBase.Plot1DWindowBase.addCurve(self, x, y, legend=legend,
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
        Plot1DWindowBase.Plot1DWindowBase.removeCurve(self, legend, replot=replot)
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


    def _zoomReset(self):
        self.graph.zoomReset()

    def getGraphXLimits(self):
        """
        Get the graph X limits. 
        """
        xmin, xmax = self.graph.getX1AxisLimits()
        return xmin, xmax
        
    def getGraphYLimits(self):
        """
        Get the graph Y (left) limits. 
        """
        ymin, ymax = self.graph.getY1AxisLimits()
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

    
