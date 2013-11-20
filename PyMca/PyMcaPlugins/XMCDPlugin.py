import PyMca.PyMca_Icons as PyMca_Icons
from PyMca import Plugin1DBase
from PyMca import XMCDWindow

from platform import node as gethostname
    
DEBUG = 0
class XMCDAnalysis(Plugin1DBase.Plugin1DBase):
    def __init__(self,  plotWindow,  **kw):
        Plugin1DBase.Plugin1DBase.__init__(self,  plotWindow,  **kw)
        self.methodDict = {}
        text = 'Perform grouped operations as function of motor value.'
        function = self.showXMCDWindow
        icon = None
        info = text
        self.methodDict["Sort plots"] =[function, info, icon]
        self.widget = None
    
    def getMethods(self, plottype=None):
        names = list(self.methodDict.keys())
        names.sort()
        return names

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        self.methodDict[name][0]()
        return

    def showXMCDWindow(self):
        if self.widget is None:
            self._createWidget()
        else:
            self.widget.updatePlots()
        self.widget.show()
        self.widget.raise_()

    def _createWidget(self):
        guess = gethostname().lower()
        beamline = '#default#'
        for hostname in ['dragon']:
            if guess.startswith(hostname):
                beamline = 'ID08'
                break
        if DEBUG:
            print '_createWidget -- beamline = "%s"'%beamline
        parent = None
        self.widget = XMCDWindow.XMCDWidget(parent,
                                              self._plotWindow,
                                              beamline,
                                              nSelectors = 2)
        

MENU_TEXT = "XLD/XMCD Analysis"
def getPlugin1DInstance(plotWindow,  **kw):
    ob = XMCDAnalysis(plotWindow)
    return ob
    
if __name__ == "__main__":
    from PyMca import ScanWindow
    from PyMca import PyMcaQt as qt
    import numpy
    app = qt.QApplication([])
    
    # Create dummy ScanWindow
    swin = ScanWindow.ScanWindow()
    info0 = {'xlabel': 'foo',
             'ylabel': 'arb',
             'MotorNames': 'oxPS Motor11 Motor10', 
             'MotorValues': '1 8.69271399699 21.9836418539'}
    info1 = {'MotorNames': 'PhaseD oxPS Motor16 Motor15',
             'MotorValues': '0.470746882688 -0.695816070299 0.825780811755 0.25876374531'}
    info2 = {'MotorNames': 'PhaseD oxPS Motor10 Motor8',
             'MotorValues': '2 0.44400576644 0.613870067852 0.901968648111'}
    x = numpy.arange(100.,1100.)
    y0 =  10*x + 10000.*numpy.exp(-0.5*(x-500)**2/400) + 1500*numpy.random.random(1000.)
    y1 =  10*x + 10000.*numpy.exp(-0.5*(x-600)**2/400) + 1500*numpy.random.random(1000.)
    y2 =  10*x + 10000.*numpy.exp(-0.5*(x-400)**2/400) + 1500*numpy.random.random(1000.)
    
    swin.newCurve(x, y2, legend="Curve2", xlabel='ene_st2', ylabel='zratio2', info=info2, replot=False, replace=False)
    swin.newCurve(x, y0, legend="Curve0", xlabel='ene_st0', ylabel='zratio0', info=info0, replot=False, replace=False)
    swin.newCurve(x, y1, legend="Curve1", xlabel='ene_st1', ylabel='zratio1', info=info1, replot=False, replace=False)

    plugin = getPlugin1DInstance(swin)
    plugin.applyMethod(plugin.getMethods()[0])
    
    app.exec_()
