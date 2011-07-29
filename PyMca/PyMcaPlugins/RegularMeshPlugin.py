import numpy
from numpy import cos, sin
import sys
try:
    import Plugin1DBase
except ImportError:
    from . import Plugin1DBase

import os
try:
    from PyMca import PyMcaQt as qt
    from PyMca import PyMcaDirs
    PYMCADIR = True
except ImportError:
    print("WARNING: RegularMeshPlugin Using huge PyQt4 import")
    import PyQt4.Qt as qt
    PYMCADIR = False

try:
    from PyMca import QStackWidget
    from PyMca import MaskImageWidget
except ImportError:
    import QStackWidget

DEBUG = 0

class RegularMeshPlugins(Plugin1DBase.Plugin1DBase):
    def __init__(self, plotWindow, **kw):
        Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)
        self.methodDict = {}
        self.methodDict['Show Image'] = [self._convert,
                                             "Show mesh as image",
                                             None]
                           
        self.stackWidget = None
        
    #Methods to be implemented by the plugin
    def getMethods(self, plottype=None):
        """
        A list with the NAMES  associated to the callable methods
        that are applicable to the specified plot.

        Plot type can be "SCAN", "MCA", None, ...        
        """
        names = list(self.methodDict.keys())
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
        if DEBUG:
                self.methodDict[name][0]()
        else:
            try:
                self.methodDict[name][0]()
            except:
                import sys
                print(sys.exc_info())
                raise

    def _convert(self):
        x, y, legend, info = self.getActiveCurve()
        self._x = x
        self._y = y
        if 'Header' not in info:
            raise ValueError("This does not seem to be a mesh scan")
            
        #print "INFO = ", info
        header = info['Header'][0]
        #print header
        item = header.split()
        if item[2] not in ['mesh', 'hklmesh']:
            raise ValueError("This does not seem to be a mesh scan")

        self._xLabel = self.getGraphXTitle()
        self._yLabel = self.getGraphYTitle()

        self._motor0Mne = item[3]
        self._motor1Mne = item[7]

        #print("Scanned motors are %s and %s" % (motor0Mne, motor1Mne))
        
        #Assume an EXACTLY regular mesh for both motors
        self._motor0 = numpy.linspace(float(item[4]), float(item[5]), int(item[6])+1)
        self._motor1 = numpy.linspace(float(item[8]), float(item[9]), int(item[10])+1)

        try:
            if xLabel.upper() == motor0Mne.upper():
                self._motor0 = self._x
                self._motor0Mne = self._xLabel
            elif xLabel.upper() == motor1Mne.upper():
                self._motor1 = self._x
                self._motor1Mne = self._xLabel
            elif xLabel == info['selection']['cntlist'][0]:
                self._motor0 = self._x
                self._motor0Mne = self._xLabel
            elif xLabel == info['selection']['cntlist'][1]:
                self._motor1 = self._x
                self._motor1Mne = self._xLabel
        except:
            if DEBUG:
                print("XLabel should be one of the scanned motors")

        self._legend = legend
        self._info = info
        y.shape = len(self._motor1), len(self._motor0)
        if self.stackWidget is None:
            self.stackWidget = MaskImageWidget.MaskImageWidget(\
                                        imageicons=False,
                                        selection=False,
                                        profileselection=True,
                                        scanwindow=self)
        self.stackWidget.setImageData(y)
        self.stackWidget.show()

    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True):
        info['LabelNames'][1] = self._yLabel
        if info['SourceName'].startswith('Row ='):
            x = self._motor0
            info['xlabel'] = self._motor0Mne
        elif info['SourceName'].startswith('Column ='):
            x = self._motor1
            info['xlabel'] = self._motor1Mne
        info['legend'] = info['selectionlegend']
        info['ylabel'] = self._yLabel
        legend = info['selectionlegend']
        return self._plotWindow.addCurve(x, y, legend=legend,
                                         info=info,
                                         replace=replace,
                                         replot=replot)

MENU_TEXT = "RegularMeshPlugins"
def getPlugin1DInstance(plotWindow, **kw):
    ob = RegularMeshPlugins(plotWindow)
    return ob

if __name__ == "__main__":
    from PyMca import Plot1D
    app = qt.QApplication([])
    #w = ConfigurationWidget()
    #w.exec_()
    #sys.exit(0)
    
    DEBUG = 1
    x = numpy.arange(100.)
    y = x * x
    plot = Plot1D.Plot1D()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x+100, -x*x)
    plugin = getPlugin1DInstance(plot)
    for method in plugin.getMethods():
        print(method, ":", plugin.getMethodToolTip(method))
    plugin.applyMethod(plugin.getMethods()[1])
    curves = plugin.getAllCurves()
    for curve in curves:
        print(curve[2])
    print("LIMITS = ", plugin.getGraphYLimits())
