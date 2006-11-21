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
import PyQt4.Qt as qt
DEBUG = 0
__revision__="$Revision: 1.5 $"

# TODO:
# - automatic picture centering
# - print quality

QTVERSION = qt.qVersion()

class ResizableGraphicsPixmapItem(qt.QGraphicsPixmapItem):
    def __init__(self, parent=None, scene = None):
        qt.QGraphicsPixmapItem.__init__(self, parent, scene)
        self.resizeRect = None
        self.setHandlesChildEvents(False)

    def setPixmap(self, pixmap):
        qt.QGraphicsPixmapItem.setPixmap(self, pixmap)
        self.resizeRect = GraphicsViewItemResizeRect(self, self.scene())

class GraphicsResizeRectItem(qt.QGraphicsRectItem):
    def __init__(self, parent = None, scene = None, keepratio = True):
        qt.QGraphicsRectItem.__init__(self, parent, scene)
        rect = parent.sceneBoundingRect()
        w = rect.width()
        h = rect.height()
        self.keepRatio = keepratio
        self.setRect(qt.QRectF(w-10, h-10, 10, 10))
        if DEBUG:
            self.setBrush(qt.QBrush(qt.Qt.white, qt.Qt.SolidPattern))
        else:
            pen = qt.QPen()
            color = qt.QColor()
            color.setAlpha(0)
            pen.setColor(color)
            self.setPen(pen)
        self.setFlag(self.ItemIsMovable, True)
        #I could not get the hover events :(
        self.setAcceptHoverEvents(True)
        self.show()

    def mouseDoubleClickEvent(self, event):
        if DEBUG:print "ResizeRect mouseDoubleClick"
        qt.QGraphicsRectItem.mouseDoubleClickEvent(self, event)

    def mousePressEvent(self, event):
        if DEBUG:print "ResizeRect mousePress"
        self.__point0 = self.pos()
        qt.QGraphicsRectItem.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if DEBUG:print "ResizeRect mouseMove"
        qt.QGraphicsRectItem.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if DEBUG:print "ResizeRect mouseRelease"
        point1 = self.pos()
        deltax = point1.x() -  self.__point0.x()
        deltay = point1.y() -  self.__point0.y()
        parent = self.parentItem()
        rect = parent.sceneBoundingRect()
        w = rect.width()
        h = rect.height()
        scalex = (w + deltax) / w
        scaley = (h + deltay) / h
        self.moveBy(-deltax, -deltay)
        if self.keepRatio:
            scalex = min(scalex, scalex)
            parent.scale(scalex, scalex)
        else:
            parent.scale(scalex, scaley)
        qt.QGraphicsRectItem.mouseReleaseEvent(self, event)

    def hoverEnterEvent(self, event):
        if DEBUG:print "ResizeRect hoverEnter"

    def hoverMoveEvent(self, event):
        if DEBUG:print "ResizeRect hoverEnter"

    def hoverLeaveEvent(self, event):
        if DEBUG:print "ResizeRect hoverEnter"

    def sceneEvent(self, event):
        if DEBUG:print "ResizeRect hoverEnter"
        qt.QGraphicsRectItem.sceneEvent(self, event)
        

    
class PrintGraphicsView(qt.QGraphicsView):
    """
    """
    def __init__(self, scene, parent =None, name ="PrintCanvas", fl =0):
        """
        """
        #scene is a QGraphicsScene
        qt.QGraphicsView.__init__(self, scene)
        self.scene().setBackgroundBrush(qt.QColor(qt.Qt.lightGray))

        self.setAcceptDrops(1)

        self.marginSize = (0, 0)
        self.marginItem = None

        self.viewScale = 1

        self.activePen = qt.QPen(qt.Qt.red, 1)

        self.__active_item = None
        self.__moving_start = None
        self.__moving_scale = 0

    def setPaperSize(self, width, height):
        """
        Define paper size. Update margin rectangle
        """
        if QTVERSION < '4.0.0':
            self.scene().resize(width, height)
        else:
            self.scene().setSceneRect(0, 0, width, height)
        self.__updateMargin()

    def getPaperSize(self):
        """
        Return paper size as tuple (width, height)
        """
        return (self.scene().width(), self.scene().height())

    def __updateMargin(self):
        """
        Redraw margin rectangle on scene
        """
        return
        if DEBUG: print "updateMargin"
        if QTVERSION < '4.0.0':
            w = self.scene().width()
            h = self.scene().height()
        else:
            w = self.scene().width()
            h = self.scene().height()
        if self.marginSize[0] > w/2 :
            self.marginSize = (int(w/2)-1, self.marginSize[1])
            
        if self.marginSize[1]>h/2 :
            self.marginSize = (self.marginSize[0], int(h/2)-1)
        if QTVERSION < '4.0.0':
            self.marginItem.move(self.marginSize[0], self.marginSize[1])
            self.marginItem.setSize(w-2*self.marginSize[0], h-2*self.marginSize[1])
            self.marginItem.setZ(-1)
            self.resizeAllItems()
            self.scene().update()
        else:
            rect = self.marginItem.rect()
            if not rect.width() or not rect.height():   #initial condition
                self.marginItem.setRect(self.marginSize[0],
                                    self.marginSize[1],
                                    w-2*self.marginSize[0],
                                    h-2*self.marginSize[1])
            else:
                self.marginItem.setPos(qt.QPointF(self.marginSize[0],
                                                  self.marginSize[1]))
                self.marginItem.setRect(qt.QRectF(0,
                                                  0,
                                                  w-2*self.marginSize[0],
                                                  h-2*self.marginSize[1]))
                
            self.marginItem.setZValue(-1)
            #self.resizeAllItems()      #this would not respect image ratio
    
    def setMargin(self, left, top):
        """
        Set left/right and top/bottom margin size
        """
        if (left, top)!=self.marginSize:
            self.marginSize = (left, top)
            self.__updateMargin()

    def getMargin(self):
        """
        Return margin size as tuple (left, right)
        """
        return self.marginSize

    def getScale(self):
        """ Return scale factor
        """
        return self.viewScale

    def setScale(self, scale =1):
        """
        Set scale factor
        """
        if QTVERSION < '4.0.0':
            if scale!=self.viewScale:
                self.viewScale = float(scale)
                matrix = qt.QWMatrix(self.viewScale,0.,0.,self.viewScale,0.,0.)
                self.setWorldMatrix(matrix)
        else:
            if scale!=self.viewScale:
                self.viewScale = float(scale)
                matrix = qt.QMatrix(self.viewScale,0.,0.,self.viewScale,0.,0.)
                self.setMatrix(matrix)
 
    def getPrintItems(self):
        """
        """
        if QTVERSION < '4.0.0':
            items = [ item for item in self.scene().allItems() \
                if item.rtti() in [PC_RTTI_Rect, PC_RTTI_Title] ]
            def itemZsort(item1, item2):
                return cmp(item1.z(), item2.z())
        else:
            items = [ item for item in self.scene().items()]
            def itemZsort(item1, item2):
                return cmp(item1.zValue(), item2.zValue())
            
        items.sort(itemZsort)
        return items

    def removeActiveItem(self):
        """
        """
        if self.__active_item is not None:
            # print self.scene().allItems()
            self.__active_item.remove()
            # print self.scene().allItems()
            self.__active_item = None
            self.scene().update()

    def removeAllItems(self):
        """
        Remove all printable items in scene
        """
        itemToRemove = self.getPrintItems()
        
        for item in itemToRemove:
            item.remove()
            self.__active_item = None

        self.scene().update()
                        
    if QTVERSION < '4.0.0':
        def contentsMouseDoubleClickEvent(self, e):
            """
            """
            if DEBUG: print "contentsMouseDoubleClickEvent"
            point = self.inverseWorldMatrix().map(e.pos())
            ilist = self.scene().collisions(point)

            if len(ilist):
                item = ilist[0]
                if item.rtti() ==PC_RTTI_Title:
                    item = item.getMasterItem()
                if item.rtti() ==PC_RTTI_Rect:
                    item.editTitle(self)
                    self.resizeItem(item)
                    self.scene().update()
    else:
        def mouseDoubleClickEvent(self, e):
            m = self.matrix()
            

    def contentsMousePressEvent(self, e):
        """
        """
        if DEBUG: print "contentsMousePressEvent"
        point = self.inverseWorldMatrix().map(e.pos())
        ilist = self.scene().collisions(point)

        self.__moving_start = None
        self.__moving_scale = 0

        if not len(ilist) or ilist[0].rtti() not in [PC_RTTI_Rect,     \
                                                     PC_RTTI_Colormap]:
            return

        if self.__active_item!=ilist[0]:
            if self.__active_item is not None:
                self.__active_item.setNormal()
            self.__active_item = ilist[0]
            self.__active_item.setActive()
            zmax = 0
            for item in self.scene().allItems():
                zmax = max(zmax, item.z())
            self.__active_item.setZ(zmax+1)
            self.scene().update()

        rect = self.__active_item.getBoundingRect()
        self.__moving_scale = self.__isMovingScale(rect, point)
        self.__moving_start = point

        if self.__moving_scale ==0: self.setCursor(qt.QCursor(qt.Qt.SizeAllCursor))
        elif self.__moving_scale in [11,22]:
            self.setCursor(qt.QCursor(qt.Qt.SizeFDiagCursor))
        elif self.__moving_scale in [12,21]:
            self.setCursor(qt.QCursor(qt.Qt.SizeBDiagCursor))
        elif self.__moving_scale in [1,2]:
            self.setCursor(qt.QCursor(qt.Qt.SizeHorCursor))
        elif self.__moving_scale in [10,20]:
            self.setCursor(qt.QCursor(qt.Qt.SizeVerCursor))

    def __isMovingScale(self, rect, point):
        """
        """
        scaling = 0

        size = max(rect.width(), rect.height())
        size = max(2, size)
        size = min(5, size)

        xl = abs(point.x()-rect.left())
        xr = abs(point.x()-rect.right())
        yt = abs(point.y()-rect.top())
        yb = abs(point.y()-rect.bottom())

        scaling = 0
        if xl<size: scaling+=1
        elif xr<size: scaling+=2

        if yt<size: scaling+=10
        elif yb<size: scaling+=20

        return scaling


    def dragEnterEvent(self, de):
        """
        """
        if DEBUG: print "dragEnterEvent"
        source = de.source()
        if source and hasattr(source, "GetImage"):
            de.accept(1)
        else:    de.accept(0)

    def dropEvent(self, de):
        """
        """
        if DEBUG: print "dropEvent"
        source = de.source()
        try:
            image = source.GetImage()
            pos = de.pos()
            item = PrintCanvasImage(pos.x(), pos.y(), image, self.scene())
            self.resizeItem(item)
            item.show()
            self.scene().update()
        except:
            sys.excepthook(sys.exc_info()[0],
                           sys.exc_info()[1],
                           sys.exc_info()[2])



            
################################################################################
##################             PyMcaPrintPreview               ###################
################################################################################
class PyMcaPrintPreview(qt.QDialog):
    """
    PyMcaPrintPreview is a widget designed to show and manage items to print. It
    is possible to
    - add/remove pixmaps to the PP scene
    - move and resize pixmaps
    - configure the printer
    - define margin for printing
    - zoom the view of the scene
    - define title and legend for items
    - print the scene
    """
    def __init__(self, parent = None, printer = None, name = "PrintPreview", \
                 modal = 0, fl = 0):
        """
        Constructor method:

        """
        if QTVERSION < '4.0.0':
            qt.QDialog.__init__(self, parent, name, modal, fl)
        else:
            qt.QDialog.__init__(self, parent)
            self.setWindowTitle(name)
            self.setModal(modal)

        self.printer    = None

        # main layout
        layout = qt.QVBoxLayout(self)
        layout.setMargin(0)
        layout.setSpacing(0)

        toolBar = qt.QWidget(self)

        # Margin
        marginLabel = qt.QLabel("Margins:", toolBar)
        if QTVERSION < '4.0.0':
            self.marginSpin = qt.QSpinBox(0, 50, 10, toolBar)
        else:
            self.marginSpin = qt.QSpinBox(toolBar)
            self.marginSpin.setRange(0, 50)
            self.marginSpin.setSingleStep(10)
        self.connect(self.marginSpin, qt.SIGNAL("valueChanged(int)"),    \
                 self.__marginChanged)

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


        # --- command buttons
        buttonSize = 65
        
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
        
        # a layout for the toolbar
        toolsLayout = qt.QHBoxLayout(toolBar)
        toolsLayout.setMargin(0)
        toolsLayout.setSpacing(0)

        # now we put widgets in the toolLayout
        toolsLayout.addWidget(hideBut)
        toolsLayout.addWidget(printBut)
        toolsLayout.addWidget(cancelBut)
        toolsLayout.addWidget(removeBut)
        toolsLayout.addWidget(setupBut)
        toolsLayout.addStretch()
        toolsLayout.addWidget(marginLabel)
        toolsLayout.addWidget(self.marginSpin)    
        toolsLayout.addStretch()
        toolsLayout.addWidget(scaleLabel)
        toolsLayout.addWidget(scaleCombo)
        toolsLayout.addStretch()

        # scene to display items to print
        self.scene_     = qt.QGraphicsScene(self)

        self.graphicsView = PrintGraphicsView(self.scene_, self)

        # status bar
        statusBar = qt.QStatusBar(self)

        self.targetLabel = qt.QLabel( "???", statusBar)
        statusBar.addWidget(self.targetLabel)

        # finally, building main widget.
        layout.addWidget(toolBar)
        layout.addWidget(self.graphicsView)
        layout.addWidget(statusBar)
        
        # use user printer or a default QPrinter
        if printer == None:
            printer = qt.QPrinter()
            #qt.QPrinter.PrinterMode.HighResolution)
        self.setPrinter(printer)

    def setPrinter(self, printer):
        """
        define a printer 
        """
        self.printer = printer
        self.updatePrinter()

    def setOutputFileName(self, name):
        """
        define the name of the output file (default is ???)
        """
        if self.printer != None:
            self.printer.setOutputFileName(name)
        else:
            print "error setOutputFileName : a printer must be defined before"

    def setPrintToFile(self, value):
        """
        define if the output is in a file (default True)
        """
        if self.printer != None:
            self.printer.setOutputToFile(value)
        else:
            print "error setOutputToFile : a printer must be defined before"

    def updatePrinter(self):
        """
        """
        # --- set paper size
        if QTVERSION < '4.0.0':
            metrics = qt.QPaintDeviceMetrics(self.printer)
            psize = (metrics.width(), metrics.height())
            self.graphicsView.setPaperSize(psize[0], psize[1]) #this sets the scene size
        else:
            self.printer.setFullPage(True)
            metrics = self.printer
            psize = (metrics.width(), metrics.height())
            self.scene_.setSceneRect(0, 0, psize[0], psize[1])
            if (psize[0] <= 0) or (psize[1] <= 0):
                print "Unable to initialize printer"
                print "Unable to use preview"
                self.printer = None
                return
            if self.graphicsView.marginItem is None:
                self.graphicsView.marginItem = qt.QGraphicsRectItem(0, 0, psize[0], psize[1])
                self.scene_.addItem(self.graphicsView.marginItem)
                self.graphicsView.marginItem.setBrush(qt.QBrush(qt.Qt.white, qt.Qt.SolidPattern))
                self.graphicsView.marginItem.show()
        # --- find correct zoom
        wsize = (self.width(), self.height())
        scale = min(float(wsize[0])/float(psize[0]),
                    float(wsize[1])/float(psize[1]))
        iscale = int(100*scale)
        dscale = [ abs(iscale - val) for val in self.scaleValues ]
        iscale = self.scaleValues[dscale.index(min(dscale))]

        self.graphicsView.setScale(float(iscale)/100.0)
        self.scaleCombo.setCurrentIndex(self.scaleValues.index(iscale))            
        if QTVERSION < '4.0.0':
            # --- possible margin values
            oldv = self.marginSpin.value()
            smax = int(psize[0]/40)*10
            self.marginSpin.setMaximum(smax)
            self.marginSpin.setSingleStep(10)
            margin = self.graphicsView.getMargin()
            if margin[0]>smax: 
                self.marginSpin.setValue(smax)
            else:    
                self.marginSpin.setValue(oldv)

        # update output target
        if QTVERSION < '4.0.0':
            fileoutput = self.printer.outputToFile()
        else:
            try:
                fileoutput = len(self.printer.outputFileName())
            except:
                fileoutput = 0
        if fileoutput:
            self.targetLabel.setText(qt.QString("File:").append(
                self.printer.outputFileName()))
        else:
            self.targetLabel.setText(qt.QString("Printer:").append(
                self.printer.printerName()))            
        self.update()

    def __marginChanged(self, value):
        """
        """
        self.graphicsView.setMargin(value, value)

    def __scaleChanged(self, index):
        """
        """
        self.graphicsView.setScale(float(self.scaleValues[index])/100.0)

    def addImage(self, image):
        """
        add an image item to the print preview scene
        """
        self.addPixmap(qt.QPixmap.fromImage(image))

    def addPixmap(self, pixmap):
        """
        add a pixamap to the print preview scene
        """
        (x,y) = self.graphicsView.getMargin()

        #I create a rectangle item
        if 0:
            rectItem = qt.QGraphicsRectItem(x+1, y+1,
                                             pixmap.width(), pixmap.height())
            self.scene_.addItem(rectItem)
        else:
            rectItem = qt.QGraphicsRectItem(self.graphicsView.marginItem, self.scene_)
            rectItem.setRect(qt.QRectF(x+1, y+1,
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
        rectItemResizeRect = GraphicsResizeRectItem(rectItem, self.scene_)
        rectItemResizeRect.setZValue(2)

        #I add a pixmap item
        pixmapItem = qt.QGraphicsPixmapItem(rectItem, self.scene_)
        pixmapItem.setPixmap(pixmap)
        pixmapItem.moveBy(1, 1)
        pixmapItem.setZValue(0)

    def __setup(self):
        """
        """
        if self.printer is not None:            
            if QTVERSION < '4.0.0':
                if self.printer.setup(): self.updatePrinter()
            else:
                self.printer = qt.QPrinter()
                printDialog = qt.QPrintDialog(self.printer, self)
                if printDialog.exec_():
                     self.updatePrinter()   
    

    def __cancel(self):
        """
        """
        self.reject()

    def __clearAll(self):
        """
        Clear the print preview window, remove all items
        but and keep the page.
        """
        itemlist = self.scene_.items()
        keep = self.graphicsView.marginItem
        while (len(itemlist) != 1):
            if itemlist.index(keep) == 0:
                self.scene_.removeItem(itemlist[1])
            else:
                self.scene_.removeItem(itemlist[0])
            itemlist = self.scene_.items()
        
    def __remove(self):
        """
        """
        itemlist = self.scene_.items()
        i = None

        #this loop is not efficient if there are many items ...
        for item in itemlist:
            if item.isSelected():
                i = itemlist.index(item)
                break
        if i is not None:
            self.scene_.removeItem(item)
            #this line is not really needed because the list
            #should be deleted at the end of the method
            del itemlist[i]             

    def __print(self):
        """
        send all items of the scene to the printer (file or device)
        """
        rectF = self.graphicsView.marginItem.rect()
        rect = qt.QRect(rectF.x(), rectF.y(), rectF.width(), rectF.height())
        printer = self.printer
        prt = qt.QPainter(self.printer)
        self.graphicsView.render(prt,
                    qt.QRectF(0, 0,printer.width(), printer.height()),
                    rect) 
        prt.end()
        self.__clearAll()
        self.accept()
            
class GraphicsResizeRectItem(qt.QGraphicsRectItem):
    def __init__(self, parent = None, scene = None, keepratio = True):
        qt.QGraphicsRectItem.__init__(self, parent, scene)
        rect = parent.sceneBoundingRect()
        w = rect.width()
        h = rect.height()
        self._newRect = None
        self.keepRatio = keepratio
        self.setRect(qt.QRectF(w-22, h-22, 20, 20))
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
        parent = self.parentItem()
        rect = parent.sceneBoundingRect()
        w = rect.width()
        h = rect.height()
        scalex = (w + deltax) / w
        scaley = (h + deltay) / h
        self.moveBy(-deltax, -deltay)
        if self.keepRatio:
            scalex = min(scalex, scalex)
            parent.scale(scalex, scalex)
        else:
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
    import os.path

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

##  MAIN   
if  __name__ == '__main__':
    testPreview()
 
 
