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
import sys
import os
import PyQt4.Qt as qt
DEBUG = 0
__revision__="$Revision: 1.5 $"

# TODO:
# - automatic picture centering
# - print quality

QTVERSION = qt.qVersion()

    
################################################################################
##################             PyMcaPrintPreview               ###################
################################################################################
class PyMcaPrintPreview(qt.QDialog):
    def __init__(self, parent = None, printer = None, name = "PyMcaPrintPreview", \
                 modal = 0, fl = 0):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle(name)
        self.setModal(modal)
        
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)

        self._buildToolbar()

        self.scene = qt.QGraphicsScene()
        self.scene.setBackgroundBrush(qt.QColor(qt.Qt.lightGray))

        printer = qt.QPrinter(qt.QPrinter.HighResolution)
        printer.setPageSize(qt.QPrinter.A4)
        printer.setFullPage(True)
        if QTVERSION < '4.2.0':
            filename = "PyMCA_print.pdf"
        else:
            filename = "PyMCA_print.ps"
        if sys.platform == 'win32':
            home = os.getenv('USERPROFILE')
            try:
                l = len(home)
                directory = os.path.join(home,"My Documents")
            except:
                home = '\\'
                directory = '\\'
            if os.path.isdir('%s' % directory):
                directory = os.path.join(directory,"PyMca")
            else:
                directory = os.path.join(home,"PyMca")
            if not os.path.exists('%s' % directory):
                os.mkdir('%s' % directory)
            finalfile = os.path.join(directory, filename)
        else:
            home = os.getenv('HOME')
            directory = os.path.join(home,"PyMca")
            if not os.path.exists('%s' % directory):
                os.mkdir('%s' % directory)
            finalfile =  os.path.join(directory, filename)
        printer.setOutputFileName(finalfile)
        printer.setColorMode(qt.QPrinter.Color)

        self.printer    = printer
        
        self.page = qt.QGraphicsRectItem(0,0, printer.width(), printer.height())
        self.page.setBrush(qt.QColor(qt.Qt.white))
        self.scene.setSceneRect(qt.QRectF(0,0, printer.width(), printer.height()))
        self.scene.addItem(self.page)
        
        self.view = qt.QGraphicsView(self.scene)

        self.mainLayout.addWidget(self.view)
        self._buildStatusBar()

        self.view.fitInView(self.page.rect(), qt.Qt.KeepAspectRatio)
        self._viewScale = 1.00

    def setOutputFileName(self, name):
        if self.printer is not None:
            self.printer.setOutputFileName(name)
        else:
            raise "IOError", "setOutputFileName : a printer must be defined before"

    def _buildToolbar(self):
        # --- command buttons
        buttonSize = 65
        toolBar = qt.QWidget(self)
        # a layout for the toolbar
        toolsLayout = qt.QHBoxLayout(toolBar)
        toolsLayout.setMargin(0)
        toolsLayout.setSpacing(0)

        # Margin
        """
        marginLabel = qt.QLabel("Margins:", toolBar)
        if QTVERSION < '4.0.0':
            self.marginSpin = qt.QSpinBox(0, 50, 10, toolBar)
        else:
            self.marginSpin = qt.QSpinBox(toolBar)
            self.marginSpin.setRange(0, 50)
            self.marginSpin.setSingleStep(10)
        self.connect(self.marginSpin, qt.SIGNAL("valueChanged(int)"),    \
                 self.__marginChanged)
        """
        # Scale / Zoom
        scaleLabel = qt.QLabel("Zoom:", toolBar)
        scaleCombo = qt.QComboBox(toolBar)
        self.scaleValues = [20, 40, 60, 80, 100, 150, 200]

        if QTVERSION < '4.0.0':
            for scale in self.scaleValues:
                scaleCombo.insertItem("%3d %%"%scale)
        else:
            for scale in self.scaleValues:
                scaleCombo.addItem("%3d %%"%scale)
            
        self.scaleCombo = scaleCombo
        self.connect(self.scaleCombo, qt.SIGNAL("activated(int)"),        \
                     self.__scaleChanged)
        
        hideBut   = qt.QPushButton("Hide", toolBar)
        hideBut.setFixedWidth(buttonSize-10)
        self.connect(hideBut, qt.SIGNAL("clicked()"), self.hide)

        cancelBut = qt.QPushButton("Clear All", toolBar)
        cancelBut.setFixedWidth(buttonSize+10)
        self.connect(cancelBut, qt.SIGNAL("clicked()"), self.__clearAll)

        removeBut = qt.QPushButton("Remove", toolBar)
        removeBut.setFixedWidth(buttonSize)
        self.connect(removeBut, qt.SIGNAL("clicked()"), self.__remove)

        setupBut  = qt.QPushButton("Setup", toolBar)
        setupBut.setFixedWidth(buttonSize-5)
        self.connect(setupBut, qt.SIGNAL("clicked()"), self.__setup)

        printBut  = qt.QPushButton("Print", toolBar)
        printBut.setFixedWidth(buttonSize-5)
        self.connect(printBut, qt.SIGNAL("clicked()"), self.__print)
        
        zoomPlusBut  = qt.QPushButton("Zoom +", toolBar)
        zoomPlusBut.setFixedWidth(buttonSize-5)
        self.connect(zoomPlusBut, qt.SIGNAL("clicked()"), self.__zoomPlus)
        
        zoomMinusBut  = qt.QPushButton("Zoom -", toolBar)
        zoomMinusBut.setFixedWidth(buttonSize-5)
        self.connect(zoomMinusBut, qt.SIGNAL("clicked()"), self.__zoomMinus)
        
        # now we put widgets in the toolLayout
        toolsLayout.addWidget(hideBut)
        toolsLayout.addWidget(printBut)
        toolsLayout.addWidget(cancelBut)
        toolsLayout.addWidget(removeBut)
        toolsLayout.addWidget(setupBut)
        #toolsLayout.addStretch()
        #toolsLayout.addWidget(marginLabel)
        #toolsLayout.addWidget(self.marginSpin)    
        toolsLayout.addStretch()
        #toolsLayout.addWidget(scaleLabel)
        #toolsLayout.addWidget(scaleCombo)
        toolsLayout.addWidget(zoomPlusBut)
        toolsLayout.addWidget(zoomMinusBut)
        toolsLayout.addStretch()
        self.toolBar = toolBar
        self.mainLayout.addWidget(self.toolBar)

    def _buildStatusBar(self):
        # status bar
        statusBar = qt.QStatusBar(self)
        self.targetLabel = qt.QLabel(statusBar)
        self.__updateTargetLabel()
        statusBar.addWidget(self.targetLabel)            
        self.mainLayout.addWidget(statusBar)

    def __updateTargetLabel(self):
        if self.printer is None:
            self.targetLabel.setText("???")
        elif len(self.printer.outputFileName()):
            self.targetLabel.setText(qt.QString("File:").append(
                self.printer.outputFileName()))
        else:
            self.targetLabel.setText(qt.QString("Printer:").append(
                self.printer.printerName()))

    def __print(self):
        printer = self.printer
        painter = qt.QPainter(printer)
        try:
            self.scene.render(painter, qt.QRectF(0, 0, printer.width(), printer.height()),
                              qt.QRectF(self.page.rect().x(), self.page.rect().y(),
                              self.page.rect().width(),self.page.rect().height()),
                              qt.Qt.KeepAspectRatio)
            painter.end()
            self.__clearAll()
            self.hide()
            self.accept()
        except:
            painter.end()
            qt.QMessageBox.critical(self, sys.exc_info()[1],
                                    'Printing problem:\n %s' % sys.exc_info()[1])
            return

    def __scaleChanged(self, value):
        if DEBUG:print "current scale = ",   self._viewScale
        if value > 2:
            self.view.scale(1.20, 1.20)
        else:
            self.view.scale(0.80, 0.80)
            
    def __zoomPlus(self):
        if DEBUG:print "current scale = ",   self._viewScale
        self._viewScale *= 1.20 
        self.view.scale(1.20, 1.20)
            
    def __zoomMinus(self):
        if DEBUG:print "current scale = ",   self._viewScale
        self._viewScale *= 0.80 
        self.view.scale(0.80, 0.80)

    def addImage(self, image):
        """
        add an image item to the print preview scene
        """
        self.addPixmap(qt.QPixmap.fromImage(image))

    def addPixmap(self, pixmap):
        """
        add a pixmap to the print preview scene
        """
        rectItem = qt.QGraphicsRectItem(self.page, self.scene)
        rectItem.setRect(qt.QRectF(1, 1,
                                         pixmap.width(), pixmap.height()))
        
        pen = rectItem.pen()
        color = qt.QColor(qt.Qt.red)
        color.setAlpha(1)
        pen.setColor(color)
        rectItem.setPen(pen)
        rectItem.setZValue(1)
        rectItem.setFlag(qt.QGraphicsItem.ItemIsSelectable, True)
        rectItem.setFlag(qt.QGraphicsItem.ItemIsMovable, True)
        rectItem.setFlag(qt.QGraphicsItem.ItemIsFocusable, False)
            
        #I add the resize tool
        rectItemResizeRect = GraphicsResizeRectItem(rectItem, self.scene)
        rectItemResizeRect.setZValue(2)

        #I add a pixmap item
        pixmapItem = qt.QGraphicsPixmapItem(rectItem, self.scene)
        pixmapItem.setPixmap(pixmap)
        pixmapItem.moveBy(1, 1)
        pixmapItem.setZValue(0)

    def __setup(self):
        """
        """
        self.printer = qt.QPrinter()
        printDialog = qt.QPrintDialog(self.printer, self)
        if printDialog.exec_():
             self.printer.setFullPage(True)
             self.updatePrinter()

    def updatePrinter(self):
        if DEBUG:print "UPDATE PRINTER"
        self.scene.setSceneRect(qt.QRectF(0,0, self.printer.width(), self.printer.height()))
        self.page.setPos(qt.QPointF(0.0, 0.0))
        self.page.setRect(qt.QRectF(0,0, self.printer.width(), self.printer.height()))
        #self.view.scale(1./self._viewScale, 1./self._viewScale) 
        #self.view.fitInView(self.page.rect(), qt.Qt.KeepAspectRatio)
        #self._viewScale = 1.00
        self.__updateTargetLabel()

    def __cancel(self):
        """
        """
        self.reject()

    def __clearAll(self):
        """
        Clear the print preview window, remove all items
        but and keep the page.
        """
        itemlist = self.scene.items()
        keep = self.page
        while (len(itemlist) != 1):
            if itemlist.index(keep) == 0:
                self.scene.removeItem(itemlist[1])
            else:
                self.scene_.removeItem(itemlist[0])
            itemlist = self.scene.items()
        
    def __remove(self):
        """
        """
        itemlist = self.scene.items()
        i = None

        #this loop is not efficient if there are many items ...
        for item in itemlist:
            if item.isSelected():
                i = itemlist.index(item)
                break

        if i is not None:
            self.scene.removeItem(item)
            #this line is not really needed because the list
            #should be deleted at the end of the method
            del itemlist[i]             

class GraphicsResizeRectItem(qt.QGraphicsRectItem):
    def __init__(self, parent = None, scene = None, keepratio = True):
        qt.QGraphicsRectItem.__init__(self, parent, scene)
        rect = parent.sceneBoundingRect()
        w = rect.width()
        h = rect.height()
        self._newRect = None
        self.keepRatio = keepratio
        self.setRect(qt.QRectF(w-40, h-40, 40, 40))
        if DEBUG:
            self.setBrush(qt.QBrush(qt.Qt.white, qt.Qt.SolidPattern))
        else:
            pen = qt.QPen()
            color = qt.QColor()
            color.setAlpha(0)
            pen.setColor(color)
            self.setPen(pen)
        self.setFlag(self.ItemIsMovable, True)
        self.show()

    def mouseDoubleClickEvent(self, event):
        if DEBUG:print "ResizeRect mouseDoubleClick"


    def mousePressEvent(self, event):
        if DEBUG:print "ResizeRect mousePress"
        if self._newRect is not None:
            self._newRect = None
        self.__point0 = self.pos()
        parent = self.parentItem()
        scene  = self.scene()
        rect = parent.rect()
        self._x = rect.x()
        self._y = rect.x()
        self._w = rect.width()
        self._h = rect.height()
        self._ratio = self._w /self._h
        self._newRect = qt.QGraphicsRectItem(parent, scene)
        self._newRect.setRect(qt.QRectF(self._x,
                                        self._y,
                                        self._w,
                                        self._h))
        qt.QGraphicsRectItem.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if DEBUG:print "ResizeRect mouseMove"
        point1 = self.pos()
        deltax = point1.x() -  self.__point0.x()
        deltay = point1.y() -  self.__point0.y()
        if self.keepRatio:
            r1 = (self._w + deltax) / self._w
            r2 = (self._h + deltay) / self._h
            if r1 < r2:
                self._newRect.setRect(qt.QRectF(self._x,
                                                self._y,
                                                self._w + deltax,
                                                (self._w + deltax)/self._ratio))
            else:
                self._newRect.setRect(qt.QRectF(self._x,
                                                self._y,
                                                (self._h + deltay)* self._ratio,
                                                self._h + deltay))
        else:
            self._newRect.setRect(qt.QRectF(self._x,
                                        self._y,
                                        self._w + deltax,
                                        self._h + deltay))
        qt.QGraphicsRectItem.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if DEBUG:print "ResizeRect mouseRelease"
        point1 = self.pos()
        deltax = point1.x() -  self.__point0.x()
        deltay = point1.y() -  self.__point0.y()
        self.moveBy(-deltax, -deltay)
        parent = self.parentItem()
        if 0:
            #this works if no zoom at the viewport
            rect = parent.sceneBoundingRect()
            w = rect.width()
            h = rect.height()
            scalex = (w + deltax) / w
            scaley = (h + deltay) / h
            if self.keepRatio:
                scalex = min(scalex, scalex)
                parent.scale(scalex, scalex)
            else:
                parent.scale(scalex, scaley)
        else:
            #deduce it from the rect because it always work
            if self.keepRatio:
                scalex = self._newRect.rect().width()/ self._w
                scaley = scalex
            else:
                scalex = self._newRect.rect().width()/ self._w
                scaley = self._newRect.rect().height()/self._h
            parent.scale(scalex, scaley)
        self.scene().removeItem(self._newRect)
        self._newRect = None
        qt.QGraphicsRectItem.mouseReleaseEvent(self, event)


################################################################################
#####################    TEST -- PyMcaPrintPreview  -- TEST   ##################
################################################################################
def testPreview():
    """
    """
    import sys
    import os

    if len(sys.argv) < 2:
        print "give an image file as parameter please."
        sys.exit(1)

    if len(sys.argv) > 2:
        print "only one parameter please."
        sys.exit(1)

    filename = sys.argv[1]

    a = qt.QApplication(sys.argv)
 
    p = qt.QPrinter()
    p.setOutputFileName(os.path.splitext(filename)[0]+".ps")
    p.setColorMode(qt.QPrinter.Color)

    w = PyMcaPrintPreview( parent = None, printer = p, name = 'Print Prev',
                      modal = 0, fl = 0)
    w.resize(400,500)
    w.addPixmap(qt.QPixmap.fromImage(qt.QImage(filename)))
    w.addImage(qt.QImage(filename))
    #w.addImage(qt.QImage(filename))
    w.exec_()




def testSimple():
    import sys
    import os
    filename = sys.argv[1]

    a = qt.QApplication(sys.argv)
    w = qt.QWidget()
    l = qt.QVBoxLayout(w)

    button = qt.QPushButton(w)
    button.setText("Print")

    scene = qt.QGraphicsScene()
    pixmapItem = qt.QGraphicsPixmapItem(qt.QPixmap.fromImage(qt.QImage(filename)))
    pixmapItem.setFlag(pixmapItem.ItemIsMovable, True)


    printer = qt.QPrinter(qt.QPrinter.HighResolution)
    printer.setPageSize(qt.QPrinter.A4)
    printer.setFullPage(True)
    printer.setOutputFileName(os.path.splitext(filename)[0]+".ps")

    page = qt.QGraphicsRectItem(0,0, printer.width(), printer.height())
    scene.setSceneRect(qt.QRectF(0,0, printer.width(), printer.height()))
    scene.addItem(page)
    scene.addItem(pixmapItem)
    view = qt.QGraphicsView(scene)
    view.fitInView(page.rect(), qt.Qt.KeepAspectRatio)
    #view.setSceneRect(
    view.scale(2, 2)
    #page.scale(0.05, 0.05)

    def printFile():
        painter = qt.QPainter(printer)
        scene.render(painter, qt.QRectF(0, 0, printer.width(), printer.height()),
                              qt.QRectF(page.rect().x(), page.rect().y(),
                              page.rect().width(),page.rect().height()),
                              qt.Qt.KeepAspectRatio)
        painter.end()
    l.addWidget(button)
    l.addWidget(view)
    w.resize(300, 600)
    w.show()
    w.connect(button, qt.SIGNAL('clicked()'), printFile)

    a.exec_()    


##  MAIN   
if  __name__ == '__main__':
    testPreview()
    #testSimple()
 
