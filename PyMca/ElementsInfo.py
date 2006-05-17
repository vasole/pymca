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
try:
    import PyQt4.Qt as qt
except:
    import qt
import ElementHtml
import Elements
from QPeriodicTable import QPeriodicTable

__revision__ = "$Revision: 1.13 $"

DEBUG = 0
CLOSE_ICON =[
"16 16 18 1",
". c None",
"d c #000000",
"c c #080808",
"k c #080c08",
"b c #181818",
"a c #212021",
"# c #212421",
"j c #292829",
"e c #313031",
"f c #393839",
"i c #424542",
"m c #525152",
"h c #525552",
"g c #5a595a",
"l c #636163",
"p c #6b696b",
"n c #7b797b",
"o c #ffffff",
"................",
"................",
"......#abcd.....",
"....efghijkdd...",
"...elmgnliaddd..",
"...fmoopnhoodd..",
"..#ggooogoooddd.",
"..ahnpooooocddd.",
"..bilngoooadddd.",
"..cjihooooodddd.",
"..dkaoooaoooddd.",
"...ddoocddoodd..",
"...ddddddddddd..",
"....ddddddddd...",
"......ddddd.....",
"................"
]


class ElementsInfo(qt.QWidget):
    def __init__(self,parent=None,name="Elements Info",fl=0):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self,parent,name,fl)
            self.setCaption(name)
        else:
            if fl == 0:
                qt.QWidget.__init__(self, parent)
            else:
                qt.QWidget.__init__(self, parent, fl)
            self.setAccessibleName(name)
            self.setWindowTitle(name)
            
        layout = qt.QVBoxLayout(self)
        self.energyValue = None
        self.splitter = qt.QSplitter(self)
        layout.addWidget(self.splitter)
        self.splitter.setOrientation(qt.Qt.Horizontal)
        self.table = QPeriodicTable(self.splitter)
        self.html  = ElementHtml.ElementHtml()
        self.infoWidget = None
        if qt.qVersion() < '4.0.0':
            self.splitter.setResizeMode(self.table,qt.QSplitter.KeepSize)
            self.connect(self.table,qt.PYSIGNAL("elementClicked"),self.elementClicked)
        else:
            self.table.setMinimumSize(500,
                                      400)
                                      
            self.connect(self.table,qt.SIGNAL("elementClicked"),
                         self.elementClicked)
            
        self.lastElement = None
        Elements.registerUpdate(self._updateCallback)
        
    def elementClicked(self, symbol):
        if self.infoWidget is None: self.__createInfoWidget(symbol)
        else:
            if qt.qVersion() < '4.0.0':
                self.infoText.setText(self.html.gethtml(symbol))
            else:
                self.infoText.clear()
                self.infoText.insertHtml(self.html.gethtml(symbol))
        if self.infoWidget.isHidden():
            self.infoWidget.show()
        self.lastElement = symbol
        if qt.qVersion() < '4.0.0':
            self.infoWidget.setCaption(symbol)
            self.infoWidget.raiseW()
        else:
            self.infoWidget.setWindowTitle(symbol)
            self.infoWidget.raise_()
        
    def __createInfoWidget(self,symbol=""):
        #Dock window like widget
        frame = qt.QWidget(self.splitter)
        layout = qt.QVBoxLayout(frame)
        
        #The dock functionnality
        toolbar = qt.QWidget(frame)
        layout.addWidget(toolbar)
        layout1       = qt.QHBoxLayout(toolbar)
        # --- the line
        if qt.qVersion() < '4.0.0':
            self.line1 = Line(toolbar,"line1")
        else:
            self.line1 = Line(toolbar)
        self.line1.setFrameShape(qt.QFrame.HLine)
        self.line1.setFrameShadow(qt.QFrame.Sunken)
        self.line1.setFrameShape(qt.QFrame.HLine)
        layout1.addWidget(self.line1)
        
        # --- the close button
        self.closelabel = PixmapLabel(toolbar)
        self.closelabel.setPixmap(qt.QPixmap(CLOSE_ICON))
        layout1.addWidget(self.closelabel)
        self.closelabel.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))

        # --- connections
        if qt.qVersion() < '4.0.0':
            self.connect(self.line1,qt.PYSIGNAL("LineDoubleClickEvent"),self.infoReparent)
            self.connect(self.closelabel,qt.PYSIGNAL("PixmapLabelMousePressEvent"),self.infoToggle)
        else:
            self.connect(self.line1,qt.SIGNAL("LineDoubleClickEvent"),self.infoReparent)
            self.connect(self.closelabel,qt.SIGNAL("PixmapLabelMousePressEvent"),self.infoToggle)

        # --- The text edit widget
        w= qt.QWidget(frame)
        layout.addWidget(w)
        l=qt.QVBoxLayout(w)
        
        hbox = qt.QWidget(w)
        hbox.layout = qt.QHBoxLayout(hbox)
        l.addWidget(hbox)
        hbox.layout.addWidget(HorizontalSpacer(hbox))
        l1=qt.QLabel(hbox)
        l1.setText('<b><nobr>Excitation Energy (keV)</nobr></b>')
        self.energy=MyQLineEdit(hbox)
        self.energy.setFixedWidth(self.energy.fontMetrics().width('#####.###'))
        self.energy.setText("")
        hbox.layout.addWidget(l1)
        hbox.layout.addWidget(self.energy)
        hbox.layout.addWidget(HorizontalSpacer(hbox))
        self.connect(self.energy,qt.SIGNAL('returnPressed()'),self._energySlot)
        if qt.qVersion() < '4.0.0':
            self.connect(self.energy,qt.PYSIGNAL('focusOut'),self._energySlot)
        else:
            self.connect(self.energy,qt.SIGNAL('focusOut'),self._energySlot)
        
        if qt.qVersion() < '3.0.0':
            self.infoText = qt.QTextView(w)
        else:
            self.infoText = qt.QTextEdit(w)
            self.infoText.setReadOnly(1)
            if qt.qVersion() < '4.0.0':
                self.infoText.setText(self.html.gethtml(symbol))
            else:
                self.infoText.clear()
                self.infoText.insertHtml(self.html.gethtml(symbol))
        l.addWidget(self.infoText)
        w.show()
        self.infoWidget=frame
        if qt.qVersion() < '4.0.0':
            self.infoWidget.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,qt.QSizePolicy.Expanding))
            self.infoWidget.setMinimumWidth(self.infoWidget.sizeHint().width()*1.8)
        frame.show()
        
    def infoReparent(self):
        if qt.qVersion() < '4.0.0':
            if self.infoWidget.parent() is not None:
                self.infoWidget.reparent(None,self.cursor().pos(),1)
            else:
                self.infoWidget.reparent(self.splitter,qt.QPoint(),1)
                #self.splitter.moveToFirst(self.sourceFrame)
        else:
            if self.infoWidget.parent() is not None:
                self.infoWidget.setParent(None)
                self.infoWidget.move(self.cursor().pos())
                self.infoWidget.show()
                #,self.cursor().pos(),1)
            else:
                self.infoWidget.setParent(self.splitter)
                self.splitter.insertWidget(1,self.infoWidget)
                #,qt.QPoint(),1)
                #self.splitter.moveToFirst(self.sourceFrame)
            self.infoWidget.setFocus()
    
    def infoToggle(self,**kw):
        if DEBUG:
            print "toggleSource called"
        if self.infoWidget.isHidden():
            self.infoWidget.show()
            self.infoWidget.raiseW()
        else:
            self.infoWidget.hide()

    def _energySlot(self):
        string = str(self.energy.text())
        if len(string):
            try:
                value = float(string)
            except:
                msg=qt.QMessageBox(self.energy)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Invalid Float")
                msg.exec_loop()
                self.energy.setFocus()
                return
            if self.energyValue is not None:
                if value != self.energyValue:
                    self.energyValue = value
                    Elements.updateDict(energy=value)
            else:
                self.energyValue = value
                Elements.updateDict(energy=value)
            if qt.qVersion() < '3.0.0':
                pass
            else:
                self.energy.setPaletteBackgroundColor(qt.QColor('white'))
            self.infoWidget.setFocus()
        else:
            self.energyValue = None
            self.energy.setText("")

            
    def _updateCallback(self):
        if self.lastElement is not None:
            self.elementClicked(self.lastElement)
            if Elements.Element[self.lastElement]['buildparameters']['energy'] is not None:
                self.energy.setText("%.3f" % Elements.Element[self.lastElement]['buildparameters']['energy'])
            else:
                self.energy.setText("")

class Line(qt.QFrame):
    def mouseDoubleClickEvent(self,event):
        if DEBUG:
            print "Double Click Event"
        dict={}
        dict['event']="DoubleClick"
        dict['data'] = event
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL("LineDoubleClickEvent"),(dict,))
        else:
            self.emit(qt.SIGNAL("LineDoubleClickEvent"),dict)

class PixmapLabel(qt.QLabel):
    def mousePressEvent(self,event):
        if DEBUG:
            print "Mouse Press Event"
        dict={}
        dict['event']="MousePress"
        dict['data'] = event
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL("PixmapLabelMousePressEvent"),(dict,))
        else:
            self.emit(qt.SIGNAL("PixmapLabelMousePressEvent"),dict)

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed))

class MyQLineEdit(qt.QLineEdit):
    def __init__(self,parent=None,name=None):
        qt.QLineEdit.__init__(self,parent)

    def setPaletteBackgroundColor(self, color):
        if qt.qVersion() < '4.0.0':
            qt.QLineEdit.setPaletteBackgroundColor(self,color)
        else:
            palette = self.palette()
            role = self.backgroundRole()
            palette.setColor(role,color)
            self.setPalette(palette)

    def focusInEvent(self,event):
        if qt.qVersion() < '3.0.0':
            pass        
        elif qt.qVersion() < '4.0.0 ':
            self.backgroundcolor = self.paletteBackgroundColor()
            self.setPaletteBackgroundColor(qt.QColor('yellow'))
        else:
            self.setPaletteBackgroundColor(qt.QColor('yellow'))


    def focusOutEvent(self,event):
        if qt.qVersion() < '3.0.0':
            pass
        else:
            self.setPaletteBackgroundColor(qt.QColor('white'))
        if qt.qVersion() <'4.0.0':
            self.emit(qt.PYSIGNAL("focusOut"),())
        else:
            self.emit(qt.SIGNAL("focusOut"),())

def main():
    import sys
    app  = qt.QApplication(sys.argv)
    winpalette = qt.QPalette(qt.QColor(230,240,249),qt.QColor(238,234,238))
    app.setPalette(winpalette)
    if len(sys.argv) > 1:
        ele = sys.argv[1]
    else:
        ele = "Fe"
    w= ElementsInfo()
    if qt.qVersion() < '4.0.0':
        app.setMainWidget(w)
        w.show()
        app.exec_loop()
    else:
        w.show()
        app.exec_()
    
if __name__ == "__main__":
    main()        
