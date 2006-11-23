#/*##########################################################################
# Copyright (C) 2004-2006 European Synchrotron Radiation Facility
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
# is a problem to you.
#############################################################################*/
__author__ = "V.A. Sole - ESRF BLISS Group"
import sys
import QtBlissGraph
qt = QtBlissGraph.qt
from Icons import IconDict
import PyMcaPrintPreview

QTVERSION = qt.qVersion()
QWTVERSION4 = QtBlissGraph.QWTVERSION4
DEBUG = 0

class RGBCorrelatorGraph(qt.QWidget):
    def __init__(self, parent = None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self._buildToolBar()
        self.graph = QtBlissGraph.QtBlissGraph(self)
        self.graph.xlabel("Row")
        self.graph.ylabel("Column")
        self.graph.yAutoScale = 1
        self.graph.xAutoScale = 1
        self.mainLayout.addWidget(self.graph)
        self.printPreview = PyMcaPrintPreview.PyMcaPrintPreview(modal = 0)
        if DEBUG: print "printPreview id = ", id(self.printPreview)

    def sizeHint(self):
        return qt.QSize(1.5 * qt.QWidget.sizeHint(self).width(),
                        qt.QWidget.sizeHint(self).height())

    def _buildToolBar(self):
        if QTVERSION < '4.0.0':
            if qt.qVersion() < '3.0':
                self.colormapIcon= qt.QIconSet(qt.QPixmap(IconDict["colormap16"]))
            else:
                self.colormapIcon= qt.QIconSet(qt.QPixmap(IconDict["colormap"]))
            self.zoomResetIcon	= qt.QIconSet(qt.QPixmap(IconDict["zoomreset"]))
            self.printIcon	= qt.QIconSet(qt.QPixmap(IconDict["fileprint"]))
            self.saveIcon	= qt.QIconSet(qt.QPixmap(IconDict["filesave"]))
            self.xAutoIcon	= qt.QIconSet(qt.QPixmap(IconDict["xauto"]))
            self.yAutoIcon	= qt.QIconSet(qt.QPixmap(IconDict["yauto"]))
            if not QWTVERSION4:
                self.hFlipIcon	= qt.QIconSet(qt.QPixmap(IconDict["gioconda16mirror"]))
        else:
            self.colormapIcon   = qt.QIcon(qt.QPixmap(IconDict["colormap"]))
            self.zoomResetIcon	= qt.QIcon(qt.QPixmap(IconDict["zoomreset"]))
            self.printIcon	= qt.QIcon(qt.QPixmap(IconDict["fileprint"]))
            self.saveIcon	= qt.QIcon(qt.QPixmap(IconDict["filesave"]))            
            self.xAutoIcon	= qt.QIcon(qt.QPixmap(IconDict["xauto"]))
            self.yAutoIcon	= qt.QIcon(qt.QPixmap(IconDict["yauto"]))
            self.hFlipIcon	= qt.QIcon(qt.QPixmap(IconDict["gioconda16mirror"]))
        self.toolBar = qt.QWidget(self)
        self.toolBarLayout = qt.QHBoxLayout(self.toolBar)
        self.toolBarLayout.setMargin(0)
        self.toolBarLayout.setSpacing(0)
        self.mainLayout.addWidget(self.toolBar)
        #Autoscale
        self._addToolButton(self.zoomResetIcon,
                            self._zoomReset,
                            'Auto-Scale the Graph')

        #y Autoscale
        tb = self._addToolButton(self.yAutoIcon,
                            self._yAutoScaleToggle,
                            'Toggle Autoscale Y Axis (On/Off)',
                            toggle = True, state=True)
        if qt.qVersion() < '4.0.0':
            tb.setState(qt.QButton.On)
        else:
            tb.setDown(True)
        self.yAutoScaleToolButton = tb
        tb.setDown(True)

        #x Autoscale
        tb = self._addToolButton(self.xAutoIcon,
                            self._xAutoScaleToggle,
                            'Toggle Autoscale X Axis (On/Off)',
                            toggle = True, state=True)
        self.xAutoScaleToolButton = tb
        tb.setDown(True)

        #colormap
        #self._addToolButton(self.colormapIcon,
        #                    self.selectColormap,
        #                    'Auto-Scale the Graph')

        #flip
        if not QWTVERSION4:
            tb = self._addToolButton(self.hFlipIcon,
                                     None,
                                     'Flip Horizontal')
            self.hFlipToolButton = tb


        #save
        tb = self._addToolButton(self.saveIcon,
                                 self._saveIconSignal,
                                 'Save Graph')

        self.toolBarLayout.addWidget(HorizontalSpacer(self.toolBar))

        # ---print
        tb = self._addToolButton(self.printIcon,
                                 self.printGraph,
                                 'Prints the Graph')

    def _addToolButton(self, icon, action, tip, toggle=None, state=None):
        tb      = qt.QToolButton(self.toolBar)            
        if QTVERSION < '4.0.0':
            tb.setIconSet(icon)
            qt.QToolTip.add(tb,tip) 
            if toggle is not None:
                if toggle:
                    tb.setToggleButton(1)
                    if state is not None:
                        if state:
                            tb.setState(qt.QButton.On)
        else:
            tb.setIcon(icon)
            tb.setToolTip(tip)
            if toggle is not None:
                if toggle:
                    tb.setCheckable(1)
                    if state is not None:
                        if state:
                            tb.setChecked(state)
                    else:
                        tb.setChecked(False)
        self.toolBarLayout.addWidget(tb)
        if action is not None:
            self.connect(tb,qt.SIGNAL('clicked()'), action)
        return tb

    def _zoomReset(self):
        if DEBUG:print "_zoomReset"
        if self.graph is not None:
            self.graph.zoomReset()
            if self.graph.yAutoScale:
                if hasattr(self, '_y1Limit'):
                    self.graph.sety1axislimits(0, self._y1Limit)
            if self.graph.xAutoScale:
                if hasattr(self, '_x1Limit'):
                    self.graph.setx1axislimits(0, self._x1Limit)
            self.graph.replot()

    def _yAutoScaleToggle(self):
        if self.graph is not None:
            if self.graph.yAutoScale:
                self.graph.yAutoScale = False
                self.yAutoScaleToolButton.setDown(False)
            else:
                self.graph.yAutoScale = True
                self.yAutoScaleToolButton.setDown(True)
            
    def _xAutoScaleToggle(self):
        if self.graph is not None:
            if self.graph.xAutoScale:
                self.graph.xAutoScale = False
                self.xAutoScaleToolButton.setDown(False)
            else:
                self.graph.xAutoScale = True
                self.xAutoScaleToolButton.setDown(True)

    def _saveIconSignal(self):
        qt.QMessageBox.information(self, "Open", "Not implemented (yet)")        
    

    def printGraph(self):
        pixmap = qt.QPixmap.grabWidget(self.graph.canvas())
        self.printPreview.addPixmap(pixmap)
        if self.printPreview.isHidden():
            self.printPreview.show()
        self.printPreview.raise_()

    def selectColormap(self):
        qt.QMessageBox.information(self, "Open", "Not implemented (yet)")  

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
      
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                           qt.QSizePolicy.Fixed))

    
class MyQLabel(qt.QLabel):
    def __init__(self,parent=None,name=None,fl=0,bold=True, color= qt.Qt.red):
        qt.QLabel.__init__(self,parent)
        if qt.qVersion() <'4.0.0':
            self.color = color
            self.bold  = bold
        else:
            palette = self.palette()
            role = self.foregroundRole()
            palette.setColor(role,color)
            self.setPalette(palette)
            self.font().setBold(bold)


    if qt.qVersion() < '4.0.0':
        def drawContents(self, painter):
            painter.font().setBold(self.bold)
            pal =self.palette()
            pal.setColor(qt.QColorGroup.Foreground,self.color)
            self.setPalette(pal)
            qt.QLabel.drawContents(self,painter)
            painter.font().setBold(0)

def test():
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit()'))

    container = RGBCorrelatorGraph()
    container.show()
    if QTVERSION < '4.0.0':
        app.exec_loop()
    else:
        app.exec_()

if __name__ == "__main__":
    test()
        
