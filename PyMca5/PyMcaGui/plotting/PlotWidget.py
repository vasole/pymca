#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys
import os
from PyMca5.PyMcaGraph import Plot

SVG = True
if "PySide" in sys.modules:
    from PySide import QtCore, QtGui
    try:
        from PySide import QtSvg
    except ImportError:
        SVG = False
else:
    from PyQt4 import QtCore, QtGui
    try:
        from PyQt4 import QtSvg
    except ImportError:
        SVG = False
if not hasattr(QtCore, "Signal"):
    QtCore.Signal = QtCore.pyqtSignal

DEBUG = 0
if DEBUG:
    Plot.DEBUG = DEBUG

class PlotWidget(QtGui.QMainWindow, Plot.Plot):
    sigPlotSignal = QtCore.Signal(object)

    def __init__(self, parent=None, backend=None,
                         legends=False, callback=None, **kw):
        QtGui.QMainWindow.__init__(self, parent)
        Plot.Plot.__init__(self, parent, backend=backend)
        if parent is not None:
            # behave as a widget
            self.setWindowFlags(QtCore.Qt.Widget)
        self.containerWidget = QtGui.QWidget()
        self.containerWidget.mainLayout = QtGui.QVBoxLayout(self.containerWidget)
        self.containerWidget.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.containerWidget.mainLayout.setSpacing(0)
        widget = self.getWidgetHandle()
        if widget is not None:
            self.containerWidget.mainLayout.addWidget(widget)
            self.setCentralWidget(self.containerWidget)
        else:
            print("WARNING: No backend. Using default.")

        # defaultPrinter
        self._printer = None

        if legends:
            print("Legends widget to be implemented")
        self.setGraphTitle("  ")
        self.setGraphXLabel("X")
        self.setGraphYLabel("Y")
        self.setCallback(callback)

    def showLegends(self, flag=True):
        print("Legends widget to be implemented")

    def graphCallback(self, ddict=None):
        if ddict is not None:
            Plot.Plot.graphCallback(self, ddict)
            self.sigPlotSignal.emit(ddict)

    def resizeEvent(self, event):
        super(PlotWidget, self).resizeEvent(event)
        #Should I reset the zoom or replot?
        #self.resetZoom()

    def replot(self):
        Plot.Plot.replot(self)
        # force update of the widget!!!
        # should this be made at the backend level?
        w = self.centralWidget()
        QtGui.qApp.postEvent(w, QtGui.QResizeEvent(w.size(),
                                                   w.size()))

    def saveGraph(self, fileName, fileFormat=None, dpi=None, **kw):
        supportedFormats = ["png", "svg", "pdf", "ps", "eps",
                            "tif", "tiff","jpeg", "jpg"]
        if fileFormat is None:
            fileFormat = (fileName.split(".")[-1]).lower()
        if fileFormat not in supportedFormats:
            print("Probably unsupported format %s" % fileFormat)
            fileFormat = "svg"
        return super(PlotWidget, self).saveGraph(fileName, fileFormat, dpi=dpi, **kw)

    def getSvgRenderer(self, printer=None):
        if not SVG:
            raise RuntimeError("QtSvg module missing. Please compile Qt with SVG support")
            return
        if sys.version < '3.0':
            import cStringIO as StringIO
            imgData = StringIO.StringIO()
        else:
            from io import BytesIO          
            imgData = BytesIO()
        self.saveGraph(imgData, fileFormat='svg')
        imgData.flush()
        imgData.seek(0)
        svgRawData = imgData.read()
        svgRendererData = QtCore.QXmlStreamReader(svgRawData)
        svgRenderer = QtSvg.QSvgRenderer(svgRendererData)
        svgRenderer._svgRawData = svgRawData
        svgRenderer._svgRendererData = svgRendererData
        return svgRenderer
        
    def printGraph(self, width=None, height=None, xOffset=0.0, yOffset=0.0,
                   units="inches", dpi=None, printer=None,
                   dialog=True, keepAspectRatio=True, **kw):
        if printer is None:
            if self._printer is None:
                printer = QtGui.QPrinter()
            else:
                printer = self._printer
        if (printer is None) or dialog:
            # allow printer selection/configuration
            printDialog = QtGui.QPrintDialog(printer, self)
            actualPrint = printDialog.exec_()
        else:
            actualPrint = True
        if actualPrint:
            self._printer = printer
            try:
                painter = QtGui.QPainter()
                if not(painter.begin(printer)):
                    return 0
                dpix    = printer.logicalDpiX()
                dpiy    = printer.logicalDpiY()

                #margin  = int((2/2.54) * dpiy) #2cm margin
                availableWidth = printer.width() #- 1 * margin
                availableHeight = printer.height() #- 2 * margin

                # get the available space
                # convert the offsets to dpi
                if units.lower() in ['inch', 'inches']:
                    xOffset = xOffset * dpix
                    yOffset = yOffset * dpiy
                    if width is not None:
                        width = width * dpix
                    if height is not None:
                        height = height * dpiy
                elif units.lower() in ['cm', 'centimeters']:
                    xOffset = (xOffset/2.54) * dpix
                    yOffset = (yOffset/2.54) * dpiy
                    if width is not None:
                        width = (width/2.54) * dpix
                    if height is not None:
                        height = (height/2.54) * dpiy
                else:
                    # page units
                    xOffset = availableWidth * xOffset
                    yOffset = availableHeight * yOffset
                    if width is not None:
                        width = availableWidth * width
                    if height is not None:
                        height = availableHeight * height
                                    
                availableWidth -= xOffset
                availableHeight -= yOffset

                if width is not None:
                    if (availableWidth + 0.1) < width:
                        txt = "Available width  %f is less than requested width %f" % \
                                      (availableWidth, width)
                        raise ValueError(txt)
                    availableWidth = width
                if height is not None:
                    if (availableHeight + 0.1) < height:
                        txt = "Available height  %f is less than requested height %f" % \
                                      (availableHeight, height)
                        raise ValueError(txt)
                    availableHeight = height

                if keepAspectRatio:
                    #get the aspect ratio
                    widget = self.getWidgetHandle()
                    if widget is None:
                        # does this make sense?
                        graphWidth = availableWidth
                        graphHeight = availableHeight
                    else:
                        graphWidth = float(widget.width())
                        graphHeight = float(widget.height())

                    graphRatio = graphHeight / graphWidth
                    # that ratio has to be respected
                    
                    bodyWidth = availableWidth
                    bodyHeight = availableWidth * graphRatio

                    if bodyHeight > availableHeight:
                        bodyHeight = availableHeight
                        bodyWidth = bodyHeight / graphRatio
                else:
                    bodyWidth = availableWidth
                    bodyHeight = availableHeight                    

                body = QtCore.QRectF(xOffset,
                                yOffset,
                                bodyWidth,
                                bodyHeight)
                svgRenderer = self.getSvgRenderer()
                svgRenderer.render(painter, body)
            finally:
                painter.end()
        
if __name__ == "__main__":
    import time
    if "matplotlib" in sys.argv:
        from PyMca5.PyMcaGraph.backends.MatplotlibBackend import MatplotlibBackend as backend
        print("USING matplotlib")
        time.sleep(1)
    else:
        try:
            from PyMca5.PyMcaGraph.backends.PyQtGraphBackend import PyQtGraphBackend as backend
            print("USING PyQtGraph")
        except:
            from PyMca5.PyMcaGraph.backends.MatplotlibBackend import MatplotlibBackend as backend
            print("USING matplotlib")
        time.sleep(1)
    import numpy
    x = numpy.arange(100.)
    y = x * x
    app = QtGui.QApplication([])
    plot = PlotWidget(None, backend=backend, legends=True)
    plot.show()
    if 1:
        plot.addCurve(x, y, "dummy")
        plot.addCurve(x+100, x*x)
        plot.addCurve(x, -y, "dummy 2")
        print("Active curve = ", plot.getActiveCurve())
        print("X Limits = ",     plot.getGraphXLimits())
        print("Y Limits = ",     plot.getGraphYLimits())
        print("All curves = ",   plot.getAllCurves())
        #print("REMOVING dummy")
        #plot.removeCurve("dummy")
        plot.insertXMarker(50., "X", label="X", draggable=True)
        #plot.insertYMarker(50., draggable=True)
        plot.setYAxisLogarithmic(True)    
    else:
        # insert a few curves
        cSin={}
        cCos={}
        nplots=50
        for i in range(nplots):
            # calculate 3 NumPy arrays
            x = numpy.arange(0.0, 10.0, 0.1)
            y = 10*numpy.sin(x+(i/10.0) * 3.14)
            z = numpy.cos(x+(i/10.0) * 3.14)
            #build a key
            a="%d" % i
            #plot the data
            cSin[a] = plot.addCurve(x, y, 'y = sin(x)' + a, replot=False)
            cCos[a] = plot.addCurve(x, z, 'y = cos(x)' + a, replot=False)
        cCos[a] = plot.addCurve(x, z, 'y = cos(x)' + a, replot=True)
        plot.insertXMarker(5., "X", label="X", draggable=True)
        plot.insertYMarker(5., "Y", label="Y", draggable=True)
    print("All curves = ", plot.getAllCurves(just_legend=True))
    app.exec_()
