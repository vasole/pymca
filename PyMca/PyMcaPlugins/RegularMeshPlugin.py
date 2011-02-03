import Plugin1DBase
import numpy
from numpy import cos, sin
import sys
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
        if DEBUG:
                self.methodDict[name][0]()
        else:
            try:
                self.methodDict[name][0]()
            except:
                import sys
                print(sys.exc_info())

    def _convert(self):
        x, y, legend, info = self.getActiveCurve()
        
        #print "INFO = ", info
        header = info['Header'][0]
        #print header
        item = header.split()
        if (item[2] != 'mesh'):
            raise ValueError("This does not seem to be a mesh scan")

        #xLabel = self.getGraphXTitle()
        motor0Mne = item[3]
        motor1Mne = item[7]

        #print("Scanned motors are %s and %s" % (motor0Mne, motor1Mne))
        
        #Assume an EXACTLY regular mesh for both motors
        motor0 = numpy.linspace(float(item[4]), float(item[5]), int(item[6])+1)
        motor1 = numpy.linspace(float(item[8]), float(item[9]), int(item[10])+1)

        """
        if xLabel.upper() == motor0Mne.upper():
            xDimension = len(motor0)
        elif xLabel.upper() == motor1Mne.upper():
            xDimension = len(motor1)
        elif xLabel == info['selection']['cntlist'][0]:
            xDimension = len(motor0)
        elif xLabel == info['selection']['cntlist'][1]:
            xDimension = len(motor1)
        else:
            raise ValueError("XLabel should be one of the scanned motors")
        """

        y.shape = len(motor1), len(motor0)
        if self.stackWidget is None:
            self.stackWidget = MaskImageWidget.MaskImageWidget(\
                                        imageicons=False,
                                        selection=False,
                                        profileselection=True,
                                        scanwindow=self._plotWindow)
        self.stackWidget.setImageData(y)
        self.stackWidget.show()

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
        print method, ":", plugin.getMethodToolTip(method)
    plugin.applyMethod(plugin.getMethods()[1])
    curves = plugin.getAllCurves()
    for curve in curves:
        print curve[2]
    print "LIMITS = ", plugin.getGraphYLimits()
