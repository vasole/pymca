from PyMca5.PyMcaGui.pymca.PyMcaMain import PyMcaMain
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaCore import Plugin1DBase


class TASPlugin(Plugin1DBase.Plugin1DBase):
    '''
    - :meth:`Plugin1DBase.addCurve`
    - :meth:`Plugin1DBase.getActiveCurve`
    - :meth:`Plugin1DBase.getAllCurves`
    - :meth:`Plugin1DBase.getGraphXLimits`
    - :meth:`Plugin1DBase.getGraphYLimits`
    - :meth:`Plugin1DBase.getGraphTitle`
    - :meth:`Plugin1DBase.getGraphXLabel`
    - :meth:`Plugin1DBase.getGraphYLabel`
    - :meth:`Plugin1DBase.removeCurve`
    - :meth:`Plugin1DBase.setActiveCurve`
    - :meth:`Plugin1DBase.setGraphTitle`
    - :meth:`Plugin1DBase.setGraphXLimits`
    - :meth:`Plugin1DBase.setGraphYLimits`
    - :meth:`Plugin1DBase.setGraphXLabel`
    - :meth:`Plugin1DBase.setGraphYLabel`

    '''
    def __init__(self, plotWindow, **kw):
        Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)

    def set_xy():
        #...
        pass

if __name__ == "__main__":
    from PyMca5.PyMcaGui.plotting.PlotWindow import PlotWindow
    app = qt.QApplication([])
    #plot = PlotWindow(roi=True, fit=True)
    #plot.show()
    wind = PyMcaMain()
    app.exec()

    #active_curve = getActiveCurve()
    #x,y,legend0,info=active_curve
            
