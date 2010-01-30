#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem for you.
#############################################################################*/
__author__ = "V.A. Sole - ESRF Software Group"
import sys
import os
import PyMcaQt as qt
import numpy
from matplotlib import cm
from matplotlib import __version__ as matplotlib_version
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyMca_Icons import IconDict
import PyMcaPrintPreview
import PyMcaDirs

DEBUG = 0

colordict = {}
colordict['blue']   = '#0000ff'
colordict['red']    = '#ff0000'
colordict['green']  = '#00ff00'
colordict['black']  = '#000000'
colordict['white']  = '#ffffff'
colordict['pink']   = '#ff66ff'
colordict['brown']  = '#a52a2a'
colordict['orange'] = '#ff9900'
colordict['violet'] = '#6600ff'
colordict['grey']   = '#808080'
colordict['yellow'] = '#ffff00'
colordict['darkgreen'] = 'g'
colordict['darkbrown'] = '#660000' 
colordict['magenta']   = 'm' 
colordict['cyan']      = 'c'
colordict['bluegreen'] = '#33ffff'
colorlist  = [colordict['black'],
              colordict['red'],
              colordict['blue'],
              colordict['green'],
              colordict['pink'],
              colordict['brown'],
              colordict['cyan'],
              colordict['orange'],
              colordict['violet'],
              colordict['bluegreen'],
              colordict['grey'],
              colordict['magenta'],
              colordict['darkgreen'],
              colordict['darkbrown'],
              colordict['yellow']]

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)

        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                                          qt.QSizePolicy.Fixed))

class QPyMcaMatplotlibSaveDialog(qt.QDialog):
    def __init__(self, parent=None, **kw):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("Matplotlib preview - Resize to your taste")
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self.plot = QPyMcaMatplotlibSave(self, **kw)
        self.actionsWidget = qt.QWidget(self)
        layout = qt.QHBoxLayout(self.actionsWidget)
        layout.setMargin(0)
        layout.setSpacing(2)
        self.acceptButton = qt.QPushButton(self.actionsWidget)
        self.acceptButton.setText("Accept")
        self.acceptButton.setAutoDefault(False)
        self.dismissButton = qt.QPushButton(self.actionsWidget)
        self.dismissButton.setText("Dismiss")
        self.dismissButton.setAutoDefault(False)
        layout.addWidget(self.acceptButton)
        layout.addWidget(HorizontalSpacer(self.actionsWidget))
        layout.addWidget(self.dismissButton)
        self.mainLayout.addWidget(self.plot)
        self.mainLayout.addWidget(self.actionsWidget)
        
        self.connect(self.acceptButton,
                     qt.SIGNAL("clicked()"),
                     self.accept)
        self.connect(self.dismissButton,
                     qt.SIGNAL("clicked()"),
                     self.reject)

class QPyMcaMatplotlibSave(FigureCanvas):
    def __init__(self, parent=None,
                 size = (7,3.5),
                 dpi = 100,
                 logx = False,
                 logy = False,
                 legends = True,
                 bw = False):

        self.fig = Figure(figsize=size, dpi=dpi) #in inches
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self,
                                   qt.QSizePolicy.Expanding,
                                   qt.QSizePolicy.Expanding)
        
        self.dpi=dpi

        self._logX = logx
        self._logY = logy
        self._bw   = bw
        self._legend   = legends
        self._legendList = []
        self._dataCounter = 0

        if not legends:
            if self._logY:
                ax = self.fig.add_axes([.15, .15, .75, .8])
            else:
                ax = self.fig.add_axes([.15, .15, .75, .75])
        else:
            if self._logY:
                ax = self.fig.add_axes([.15, .15, .7, .8])
            else:
                ax = self.fig.add_axes([.15, .15, .7, .8])

        ax.set_axisbelow(True)

        self.ax = ax


        if self._logY:
            self._axFunction = ax.semilogy
        else:
            self._axFunction = ax.plot

        if self._bw:
            self.colorList = ['k']   #only black
            self.styleList = ['-', ':', '-.', '--']
            self.nColors   = 1
        else:
            self.colorList = colorlist
            self.styleList = ['-', '-.', ':']
            self.nColors   = len(colorlist)
        self.nStyles   = len(self.styleList)

        self.colorIndex = 0
        self.styleIndex = 0

        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.limitsSet = False

    def setLimits(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.limitsSet = True


    def _filterData(self, x, y):
        index = numpy.flatnonzero((self.xmin <= x) & (x <= self.xmax))
        x = numpy.take(x, index)
        y = numpy.take(y, index)
        index = len(index)
        if index:
            index = numpy.flatnonzero((self.ymin <= y) & (y <= self.ymax))
            index = len(index)
        return index

    def _getColorAndStyle(self):
        color = self.colorList[self.colorIndex]
        style = self.styleList[self.styleIndex]
        self.colorIndex += 1
        if self.colorIndex >= self.nColors:
            self.colorIndex = 0
            self.styleIndex += 1
            if self.styleIndex >= self.nStyles:
                self.styleIndex = 0        
        return color, style

    def addDataToPlot(self, x, y, legend = None,
                      color = None,
                      linewidth = None,
                      linestyle = None, **kw):
        n = max(x.shape)
        if self.limitsSet is not None:
            n = self._filterData(x, y)
        if n == 0:
            #nothing to plot
            if DEBUG:
                print "nothing to plot"
            return
        style = None
        if color is None:
            color, style = self._getColorAndStyle()
        if linestyle is None:
            if style is None:
                style = '-'
        else:
            style = linestyle

        if linewidth is None:linewidth = 1.0
        self._axFunction( x, y, linestyle = style, color=color, linewidth = linewidth, **kw)
        self._dataCounter += 1
        if legend is None:
            #legend = "%02d" % self._dataCounter    #01, 02, 03, ...
            legend = "%c" % (96+self._dataCounter)  #a, b, c, ..
        self._legendList.append(legend)

    def setXLabel(self, label):
        self.ax.set_xlabel(label)

    def setYLabel(self, label):
        self.ax.set_ylabel(label)

    def setTitle(self, title):
        self.ax.set_title(title)
        
    def plotLegends(self):
        if not self._legend:return
        if not len(self._legendList):return
        loc = (1.01, 0.0)
        labelsep = 0.015
        drawframe = True
        fontproperties = FontProperties(size=10)
        if len(self._legendList) > 14:
            drawframe = False
            if matplotlib_version < '0.99.0':
                fontproperties = FontProperties(size=8)
                loc = (1.05, -0.2)
            else:
                if len(self._legendList) < 18:
                    #drawframe = True
                    loc = (1.01,  0.0)
                elif len(self._legendList) < 25:
                    loc = (1.05,  0.0)
                    fontproperties = FontProperties(size=8)
                elif len(self._legendList) < 28:
                    loc = (1.05,  0.0)
                    fontproperties = FontProperties(size=6)
                else:
                    loc = (1.05,  -0.1)
                    fontproperties = FontProperties(size=6)
        
        if matplotlib_version < '0.99.0':
            legend = self.ax.legend(self._legendList,
                                loc = loc,
                                prop = fontproperties,
                                labelsep = labelsep,
                                pad = 0.15)
        else:
            legend = self.ax.legend(self._legendList,
                                loc = loc,
                                prop = fontproperties,
                                labelspacing = labelsep,
                                borderpad = 0.15)
        legend.draw_frame(drawframe)


    def saveFile(self, filename, format=None):
        if format is None:
            format = filename[-3:]

        if format.upper() not in ['EPS', 'PNG', 'SVG']:
            raise "Unknown format %s" % format

        if os.path.exists(filename):
            os.remove(filename)

        if self.limitsSet:
            self.ax.set_ylim(self.ymin, self.ymax)
            self.ax.set_xlim(self.xmin, self.xmax)
        #self.plotLegends()
        self.print_figure(filename, dpi=self.dpi)
        return

if __name__ == "__main__":
    import sys
    app = qt.QApplication([])
    w0=QPyMcaMatplotlibSaveDialog(legends=True)
    w=w0.mplt
    x = numpy.arange(1200.)
    w.setLimits(0, 1200., 0, 12000.)
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    else:
        n = 14
    for i in range(n):
        y = x * i
        w.addDataToPlot(x,y, legend="%d" % i)
    #w.setTitle('title')
    w.setXLabel('Channel')
    w.setYLabel('Counts')
    w.plotLegends()
    ret = w0.exec_()
    if ret:
        w.saveFile("filename.png")
        print "Plot filename.png saved"
