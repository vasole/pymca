#/*##########################################################################
# Copyright (C) 2004-2009 European Synchrotron Radiation Facility
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
import qt
import qtcanvas

paintbrush_xpm = [
"17 19 8 1",
"   c None",
"n  c #000000",
"c  c #444444",
"m  c #AAAAAA",
"*  c #DDDDDD",
"d  c #949494",
"e  c #706c6c",
"j  c #000000",
"           ndecn ",
"          ndeecn ",
"         ndeecn  ",
"         ndeecn  ",
"        ndeecn   ",
"        ndeecn   ",
"       ndeecn    ",
"       ndeecn    ",
"       nnnnn     ",
"      n*mmn      ",
"     n**mcn      ",
"     n*mmcn      ",
"    n*mmcn       ",
"    nmccn        ",
"   nccnn         ",
"  nnnn           ",
"                 ",
" jjjjjjjjjjjjjj  ",
" jjjjjjjjjjjjjj  "]

###############################################################################
##################           QubColorToolButton                   #############
###############################################################################
class QubColorToolButton(qt.QToolButton):
    """
    The QubColorToolButton provides a pushButton usable as a color selector.
    Its icon represents a paint tool with a color sample dynamicly updated
    according to the selected color.
    """
    def __init__(self, parent=None, name="CTB"):
        """
        """
        qt.QToolButton.__init__(self, parent, name)

        self.setAutoRaise(True)
        
        self.simpleColor = [qt.Qt.black,
                            qt.Qt.white,
                            qt.Qt.red,
                            qt.Qt.green,
                            qt.Qt.blue,
                            qt.Qt.yellow]
        
        self.selectedColor = qt.QColor(qt.Qt.black)
        
        self.setIconColor(self.selectedColor)
        
        self.popupMenu = qt.QPopupMenu(self)

        self.setPopup(self.popupMenu)
        self.setPopupDelay(0)
        
        colorBar = qt.QHButtonGroup(self.popupMenu)
        colorBar.setFlat(True)
        colorBar.setInsideMargin(5)
        colorButton = []
        
        for color in self.simpleColor:
            w = qt.QPushButton(colorBar)
            w.setPaletteBackgroundColor(color)
            w.setFixedSize(15, 15)
            colorBar.insert(w)
            colorButton.append(w)
        
        self.connect(colorBar, qt.SIGNAL("clicked(int )"),
                     self.selectSimpleColor)
                    
        otherBar = qt.QHButtonGroup(self.popupMenu)
        otherBar.setFlat(True)
        otherBar.setInsideMargin(5)
        moreButton = qt.QToolButton(otherBar)
        moreButton.setTextLabel("More color ...")
        moreButton.setUsesTextLabel(True)
        moreButton.setAutoRaise(True)
        self.connect(moreButton, qt.SIGNAL("clicked()"), self.selectColor)
                        
        self.popupMenu.insertItem(colorBar)
        self.popupMenu.insertItem(otherBar)
        
    def selectSimpleColor(self, ind):
        """
        
        """
        self.selectedColor = qt.QColor(self.simpleColor[ind])
        self.setIconColor(self.selectedColor)
        self.emit(qt.PYSIGNAL("colorSelected"), (self.selectedColor,))
        self.popupMenu.hide()
        
    def selectColor(self):
        """
        
        """
        color = qt.QColorDialog.getColor(self.selectedColor, self)
        
        if color is not None:
            self.selectedColor = color
            self.setIconColor(self.selectedColor)
            self.emit(qt.PYSIGNAL("colorSelected"), (self.selectedColor,))      
        
        self.popupMenu.hide()
    
    def setColor(self, color):
        try:
            self.setIconColor(color)
        except:
            self.setIconColor(self.selectedColor)
        else:
            self.selectedColor = color
            
    def setIconColor(self, color):
        """
        internal method to change the color of the icon
        """
        r = color.red()
        g = color.green()
        b = color.blue()
        paintbrush_xpm[8] = "j  c #%02x%02x%02x"%(r, g, b)
        self.setPixmap(qt.QPixmap(paintbrush_xpm))

class QubTitleEditor(qt.QDialog):
    """
    This dialog window allow to define top and bottom texts (fonts and
    colors) to comment items put in a PrintPreviewCanvas
    """
    def __init__(self, topText = None, bottomText = None, \
                 parent = None, name = "TextEditor", modal = 1, fl = 0): 
        """
        Constructor method
        topText .. : text appearing in the top of the item (title ?)
        bottomText : text appearing in the bottom of the item (legend ?)
        parent ... : 
        name ..... : name of the dialog widget
        modal .... : is the dialog modal ?
        fl ....... : qt flag
        """
        qt.QDialog.__init__(self, parent, name, modal, fl)
        self.setCaption(name)

        # placing
        layout = qt.QVBoxLayout(self)
        gridg  = qt.QHGroupBox(self)
        
        gridw  = qt.QWidget(gridg)
        grid   = qt.QGridLayout(gridw, 2, 4)

        grid.setColStretch(1, 4)

        # Top Text
        topLabel     = qt.QLabel("Top Text", gridw)   # self. ?
        self.topText = qt.QLineEdit(gridw)
        
        if topText is not None:
            self.topText.setText(topText[0])
            self.topFont  = topText[1]
            self.topColor = topText[2]
        else:
            self.topFont  = self.topText.font()
            self.topColor = self.topText.paletteForegroundColor()

        topFontButton = qt.QPushButton("Font" , gridw)
        self.topColorButton = QubColorToolButton (gridw)
        self.topColorButton.setAutoRaise(False)
        self.topColorButton.setIconColor(self.topColor)
        
        self.connect(topFontButton , qt.SIGNAL("clicked()"), self.__topFont)
        self.connect(self.topColorButton, qt.PYSIGNAL("colorSelected"),
                     self.__setTopColor)
        
        grid.addWidget(topLabel, 0, 0)
        grid.addWidget(self.topText, 0, 1)
        grid.addWidget(topFontButton, 0, 2)
        grid.addWidget(self.topColorButton, 0, 3)
        

        # Bottom Text
        botLabel     = qt.QLabel("Bottom Text", gridw)
        self.botText = qt.QLineEdit(gridw)

        if bottomText is not None:
            self.botText.setText(bottomText[0])
            self.botFont  = bottomText[1]
            self.botColor = bottomText[2]
        else:
            self.botFont  = self.botText.font()
            self.botColor = self.botText.paletteForegroundColor()

        botFontButton = qt.QPushButton("Font", gridw)
        self.botColorButton = QubColorToolButton (gridw)
        self.botColorButton.setAutoRaise(False)
        self.botColorButton.setIconColor(self.botColor)
            
        self.connect(botFontButton,  qt.SIGNAL("clicked()"), self.__botFont)
        self.connect(self.botColorButton, qt.PYSIGNAL("colorSelected"),
                     self.__setBotColor)

        grid.addWidget(botLabel, 1, 0)
        grid.addWidget(self.botText, 1, 1)
        grid.addWidget(botFontButton, 1, 2)
        grid.addWidget(self.botColorButton, 1, 3)

        # dialog buttons
        butw = qt.QHButtonGroup(self)
        cancelBut = qt.QPushButton("Cancel", butw)
        okBut     = qt.QPushButton("OK", butw)
        okBut.setDefault(1)
        self.connect(cancelBut, qt.SIGNAL("clicked()"), self.reject)
        self.connect(okBut,     qt.SIGNAL("clicked()"), self.accept)

        layout.addWidget(gridg)
        layout.addWidget(butw)

    def __topFont(self):
        """
        set the Font of the top label
        """
        font = qt.QFontDialog.getFont(self.topFont)
        if font[1]:
            self.topFont = font[0]

    def __setTopColor(self):
        """
        set the Color of the top label
        """
        self.topColor = self.topColorButton.selectedColor

    def __setBotColor(self):
        """
        set the Color of the bottom label
        """
        self.botColor = self.botColorButton.selectedColor

    def __botFont(self):
        """
        set the Font of the bottom label
        """
        font = qt.QFontDialog.getFont(self.botFont)
        if font[1]:
            self.botFont = font[0]

    def getTopText(self):
        """
        return the text of the Top Label
        """
        text = str(self.topText.text())
        if len(text):
            return (text, self.topFont, self.topColor)
        else:    return (None, None, None)

    def getBottomText(self):
        """
        return the text of the bottom Label
        """
        text = str(self.botText.text())
        if len(text):
            return (text, self.botFont, self.botColor)
        else:    return (None, None, None)


################################################################################
####################             BoundingRect               ####################
################################################################################
class BoundingRect(qt.QRect):
    """
    """
    def __init__(self, x, y, w, h, topRect =None, bottomRect =None):
        """
        """
        qt.QRect.__init__(self, x, y, w, h)
        self.topRect = topRect
        self.bottomRect = bottomRect

    def __getInRect(self, rect):
        """
        """
        if self.topRect is not None:
            rect.setTop(rect.top()+self.topRect.height())
        if self.bottomRect is not None:
            rect.setBottom(rect.bottom()-self.bottomRect.height())
        return rect

    def setMinWidth(self):
        """
        """
        self.setWidth(1)

    def setMinHeight(self):
        """
        """
        self.setHeight(1)

    def moveByIn(self, dx, dy, rect):
        """ Move rectangle by (dx,dy) staying inside (rect)
            Return 1 if rectangle changed, 0 otherwise
        """
        rect = self.__getInRect(rect)
        if self.left() + dx < rect.left():
            dx = self.left() - rect.left()
        if self.right() + dx > rect.right():
            dx = rect.right() - self.right()
        if self.top() + dy < rect.top():
            dy = self.top() - rect.top()
        if self.bottom() + dy > rect.bottom():
            dy = rect.bottom()-self.bottom()
        if (dx,dy) != (0., 0.):
            self.moveBy(dx, dy)
            return 1
        else:
            return 0

    def sizeToIn(self, scaling, point, rect):
        """ Change rectangle size to include point staying inside rect
            scaling: Give which corner/side needs to be moved

             11      10      12
              \      |      /
               +-----------+
               |           |
               |           |
            1 -|           |- 2
               |           |
               |           |
               +-----------+
              /      |      \
             21      20      22
        """

        rect  = self.__getInRect(rect)
        point = self.getPointIn (point, rect)
        
        if scaling in [1,  11, 21]:  self._sizeToLeft(point, rect)
        if scaling in [2,  12, 22]:  self._sizeToRight(point, rect)
        if scaling in [10, 11, 12]:  self._sizeToTop(point, rect)
        if scaling in [20, 21, 22]:  self._sizeToBottom(point, rect)
        return 1

    def resizeIn(self, rect):
        """
        """
        rect = self.__getInRect(rect)
        left = min(  max( self.left(), rect.left() ),  rect.right()   )
        top  = min(  max( self.top(), rect.top( )  ),  rect.bottom()  )
        
        self.moveTopLeft(qt.QPoint(left, top))
        
        if self.right()  > rect.right():  self.setRight(rect.right())
        if self.bottom() > rect.bottom(): self.setBottom(rect.bottom())

    def _sizeToLeft(self, point, rect):
        """
        """
        if point.x()>self.right():
            self.setMinWidth()
        else:
            self.setLeft(max(point.x(), rect.left()))

    def _sizeToRight(self, point, rect):
        """
        """
        if point.x()<self.left():
            self.setMinWidth()
        else:
            self.setRight(min(point.x(), rect.right()))

    def _sizeToTop(self, point, rect):
        """
        """
        if point.y()>self.bottom():
            self.setMinHeight()
        else:
            self.setTop(max(point.y(), rect.top()))

    def _sizeToBottom(self, point, rect):
        """
        """
        if point.y()<self.top():
            self.setMinHeight()
        else:
            self.setBottom(min(point.y(), rect.bottom()))
            
    def getPointIn(self, point, rect):
        """ Change point coordinates to stay inside rect
        """
        pointin = qt.QPoint(point.x(), point.y())
        if point.x() < rect.left():     pointin.setX(rect.left())
        if point.x() > rect.right():    pointin.setX(rect.right())
        if point.y() < rect.top():      pointin.setY(rect.top())
        if point.y() > rect.bottom():   pointin.setY(rect.bottom())
        return pointin


class BoundingFixedScaleRect(BoundingRect):
    """
    """
    def __init__(self, x, y, w, h, scale, topRect =None, bottomRect =None):
        """ scale is w/h
        """
        BoundingRect.__init__(self, x, y, w, h, topRect, bottomRect)
        self.scale = scale

    def setMinWidth(self):
        """
        """
        if self.width()<self.height():
            self.setWidth(1)
        else:
            self.setWidth(int(self.scale))

    def setMinHeight(self):
        """
        """
        if self.height()<self.width():
            self.setHeight(1)
        else:
            self.setHeight(int(1/self.scale))

    def setWidth(self, w):
        """
        """
        BoundingRect.setWidth(self, w)
        BoundingRect.setHeight(self, int(w/self.scale))

    def setHeight(self, h):
        """
        """
        BoundingRect.setHeight(self, h)
        BoundingRect.setWidth(self, int(h*self.scale))

    def setLeft(self, l):
        """
        """
        BoundingRect.setLeft(self, l)
        BoundingRect.setHeight(self, int(self.width()/self.scale))

    def setRight(self, r):
        """
        """
        BoundingRect.setRight(self, r)
        BoundingRect.setHeight(self, int(self.width()/self.scale))

    def setTop(self, t):
        """
        """
        BoundingRect.setTop(self, t)
        BoundingRect.setWidth(self, int(self.height()*self.scale))

    def setBottom(self, b):
        """
        """
        BoundingRect.setBottom(self, b)
        BoundingRect.setWidth(self, int(self.height()*self.scale))

    def _sizeToLeft(self, point, rect):
        """
        """
        if point.x()>self.right(): self.setMinWidth()
        else:
            self.setLeft(max(point.x(), rect.left()))
            if self.bottom()>rect.bottom():
                self.setBottom(rect.bottom())
        
    def _sizeToRight(self, point, rect):
        """
        """
        if point.x()<self.left():
            self.setMinWidth()
        else: 
            self.setRight(min(point.x(), rect.right()))
            if self.bottom()>rect.bottom():
                self.setBottom(rect.bottom())

    def _sizeToTop(self, point, rect):
        """
        """
        if point.y()>self.bottom():
            self.setMinHeight()
        else: 
            self.setTop(max(point.y(), rect.top()))
            if self.right()>rect.right():
                self.setRight(rect.right())

    def _sizeToBottom(self, point, rect):
        """
        """
        if point.y()<self.top():
            self.setMinHeight()
        else: 
            self.setBottom(min(point.y(), rect.bottom()))
            if self.right()>rect.right():
                self.setRight(rect.right())

PC_RTTI_Title = 3500
class PrintCanvasTitle(qtcanvas.QCanvasText):
    """
    """
    def __init__(self, text, master_item, position = "top"):
        """
        """
        self.masterItem = master_item
        self.position   = position

        qtcanvas.QCanvasText.__init__(self, text, master_item.canvas())

        if position == "top":
            self.setTextFlags(qt.Qt.AlignHCenter | qt.Qt.AlignBottom)
        else:
            self.setTextFlags(qt.Qt.AlignHCenter | qt.Qt.AlignTop)
            
    def getMasterItem(self):
        """
        """
        return self.masterItem

    def updatePosition(self, rect):
        """
        """
        if self.position == "top":
            self.move(rect.center().x(), rect.top() )
        else:
            self.move(rect.center().x(), rect.bottom() )

    def getParameters(self):
        """
        """
        return (self.text(), self.font(), self.color())

    def setParameters(self, text = None, font = None, color = None):
        """
        """
        if text is not None:
            self.setText(text)
        if font is not None:
            self.setFont(font)
        if color is not None:
            self.setColor(color)

    def printTo(self, painter):
        """
        """
        self.draw(painter)

    def rtti(self):
        """
        """
        return PC_RTTI_Title

PC_RTTI_Rect = 3501
class PrintCanvasRectangle(qtcanvas.QCanvasRectangle):
    """
    """
    def __init__(self, x, y, w, h, canvas, keepscale =0):
        """
        """

        qtcanvas.QCanvasRectangle.__init__(self, x, y, w, h, canvas)

        self.__pen = qt.QPen(qt.Qt.black, 1)
        self.setPen(self.__pen)
        
        if keepscale:
            self.scale = float(w)/float(h)
        else:
            self.scale = 0

        self.topItem    = None
        self.bottomItem = None

    def setTopTitle(self, text = None, font = None, color = None):
        """
        """
        if text is None:
            if self.topItem is not None:
                self.topItem.hide()
                self.topItem = None
        else:
            if self.topItem is None:
                self.topItem = PrintCanvasTitle(text, self, position ="top")
                self.topItem.updatePosition(self.rect())
                self.topItem.show()
            self.topItem.setParameters(text, font, color)
            self.topItem.update()

    def setBottomTitle(self, text = None, font = None, color = None):
        """
        """
        if text is None:
            if self.bottomItem is not None:
                self.bottomItem.hide()
                self.bottomItem = None
        else:
            if self.bottomItem is None:
                self.bottomItem = PrintCanvasTitle(text, self,             \
                                                   position ="bottom")
                self.bottomItem.updatePosition(self.rect())
                self.bottomItem.show()
            self.bottomItem.setParameters(text, font, color)
            self.bottomItem.update()

    def setActive(self):
        """
        """
        self.__pen = self.pen()
        self.setPen(qt.QPen(qt.QColor(qt.Qt.red), 1))

    def setNormal(self):
        """
        """
        self.setPen(self.__pen)

    def getBoundingRect(self):
        """
        """
        rect = self.rect()

        if self.topItem is not None:
            topRect = self.topItem.boundingRect()
        else:
            topRect = None

        if self.bottomItem is not None:
            bottomRect = self.bottomItem.boundingRect()
        else:
            bottomRect = None

        if self.scale:
            return BoundingFixedScaleRect(rect.x(), rect.y(),         \
                                          rect.width(), rect.height(), \
                                          self.scale, topRect, bottomRect)
        else:    
            return BoundingRect(rect.x(), rect.y(),        \
                                rect.width(), rect.height(),      \
                                topRect, bottomRect)

    def setBoundingRect(self, rect):
        """
        """
        self.move(rect.x(), rect.y())
        if self.scale: 
            self.setSize(rect.width(), int(rect.width()/self.scale))
        else:     self.setSize(rect.width(), rect.height())
        if self.topItem is not None:
            self.topItem.updatePosition(rect)
        if self.bottomItem is not None:
            self.bottomItem.updatePosition(rect)

    def rtti(self):
        """
        """
        return PC_RTTI_Rect

    def fullRedraw(self):
        """
        """
        return 0

    def printTo(self, painter):
        """
        """
        painter.drawRect(self.rect())

    def editTitle(self, master):
        """
        """
        if 0:
            text = "Not implemented (yet)\n"
            text += "Please use the methods setTopTitle,"
            text += "setBottomTitle and self.update()"
            qt.QMessageBox.information(master, "Open", text )
            return
        if self.topItem is not None:
            topText = self.topItem.getParameters()
        else:
            topText = None
        if self.bottomItem is not None:
            bottomText = self.bottomItem.getParameters()
        else:
            bottomText = None

        dlg = QubTitleEditor(topText, bottomText, master)
        dlg.exec_loop()
        
        if dlg.result() == qt.QDialog.Accepted:
            topText = dlg.getTopText()
            self.setTopTitle(topText[0], topText[1], topText[2])
            bottomText = dlg.getBottomText()
            self.setBottomTitle(bottomText[0], bottomText[1], bottomText[2])
            self.update()

    def remove(self):
        """
        """
        if self.topItem is not None:
            self.topItem.setCanvas(None)
            self.topItem = None
        if self.bottomItem is not None:
            self.bottomItem.setCanvas(None)
            self.bottomItem = None
        self.setCanvas(None)


class PrintCanvasImage(PrintCanvasRectangle):
    """
    """
    def __init__(self, x, y, img, canvas):
        """
        """
        PrintCanvasRectangle.__init__(self, x, y,         \
                                      img.width(), img.height(), canvas, 1)
        self.image = img
        self.pixmap = qt.QPixmap(img)
        self.pixmap.setOptimization(qt.QPixmap.BestOptim)
        self.imageWidth = float(img.width())
        self.fullRedrawFlag = 1

    def draw(self, p):
        """
        """
        if self.fullRedrawFlag:
            scale = float(self.width())/self.imageWidth
            wm = p.worldMatrix()
            nwm = qt.QWMatrix(wm.m11(),wm.m12(),wm.m21(),wm.m22(),wm.dx(),wm.dy())
            nwm = nwm.scale(scale, scale)
            np = qt.QPainter(p.device())
            np.setWorldMatrix(nwm)
            np.drawPixmap(int(self.x()/scale)+1, int(self.y()/scale)+1,    \
                          self.pixmap)
            
        PrintCanvasRectangle.draw(self, p)

    def setBoundingRect(self, rect):
        """
        """
        self.fullRedrawFlag = 0
        PrintCanvasRectangle.setBoundingRect(self, rect)

    def fullRedraw(self):
        """
        """
        self.fullRedrawFlag = 1
        self.update()
        return 1

    def printTo(self, painter):
        """
        """
        img = self.image.scaleWidth(self.width())
        painter.drawImage(self.x(), self.y(), img)


class PrintCanvasPixmap(PrintCanvasImage):
    """
    """
    def printTo(self, painter):
        img = self.image.convertToImage()
        img = img.scaleWidth(self.width())
        painter.drawImage(self.x(), self.y(), img)


class PrintCanvasView(qtcanvas.QCanvasView):
    """
    """
    def __init__(self, canvas, parent =None, name ="PrintCanvas", fl =0):
        """
        """
        qtcanvas.QCanvasView.__init__(self, canvas, parent, name, fl)

        self.setAcceptDrops(1)

        self.canvas().setBackgroundColor(qt.QColor(qt.Qt.lightGray))
        self.marginSize = (0, 0)
        self.marginItem = qtcanvas.QCanvasRectangle(0, 0, 0, 0, self.canvas())
        self.marginItem.setBrush(qt.QBrush(qt.Qt.white, qt.Qt.SolidPattern))
        self.marginItem.show()

        self.viewScale = 1

        self.activePen = qt.QPen(qt.Qt.red, 1)

        self.__active_item = None
        self.__moving_start = None
        self.__moving_scale = 0

    def setPaperSize(self, width, height):
        """
        Define paper size. Update margin rectangle
        """
        self.canvas().resize(width, height)
        self.__updateMargin()

    def getPaperSize(self):
        """
        Return paper size as tuple (width, height)
        """
        return (self.canvas().width(), self.canvas().height())

    def __updateMargin(self):
        """
        Redraw margin rectangle on canvas
        """
        w = self.canvas().width()
        h = self.canvas().height()
        if self.marginSize[0] > w/2 :
            self.marginSize = (int(w/2)-1, self.marginSize[1])
            
        if self.marginSize[1]>h/2 :
            self.marginSize = (self.marginSize[0], int(h/2)-1)
            
        self.marginItem.move(self.marginSize[0], self.marginSize[1])
        self.marginItem.setSize(w-2*self.marginSize[0], h-2*self.marginSize[1])
        self.marginItem.setZ(-1)
        self.resizeAllItems()
        self.canvas().update()
    
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
        if scale!=self.viewScale:
            self.viewScale = float(scale)
            matrix = qt.QWMatrix(self.viewScale,0.,0.,self.viewScale,0.,0.)
            self.setWorldMatrix(matrix)

    def resizeAllItems(self):
        """
        """
        for item in self.canvas().allItems():
            self.resizeItem(item)

    def resizeItem(self, item):
        """
        """
        if item.rtti() in [PC_RTTI_Rect]:
            rect = item.getBoundingRect()
            rect.resizeIn(self.marginItem.rect())
            item.setBoundingRect(rect)
            item.fullRedraw()

    def getPrintItems(self):
        """
        """
        items = [ item for item in self.canvas().allItems() \
                if item.rtti() in [PC_RTTI_Rect, PC_RTTI_Title] ]
        def itemZsort(item1, item2):
            return cmp(item1.z(), item2.z())
        items.sort(itemZsort)
        return items

    def removeActiveItem(self):
        """
        """
        if self.__active_item is not None:
            # print self.canvas().allItems()
            self.__active_item.remove()
            # print self.canvas().allItems()
            self.__active_item = None
            self.canvas().update()

    def removeAllItems(self):
        """
        Remove all printable items in canvas
        """
        itemToRemove = self.getPrintItems()
        
        for item in itemToRemove:
            item.remove()
            self.__active_item = None

        self.canvas().update()
                        
    def contentsMouseDoubleClickEvent(self, e):
        """
        """
        point = self.inverseWorldMatrix().map(e.pos())
        ilist = self.canvas().collisions(point)

        if len(ilist):
            item = ilist[0]
            if item.rtti() ==PC_RTTI_Title:
                item = item.getMasterItem()
            if item.rtti() ==PC_RTTI_Rect:
                item.editTitle(self)
                self.resizeItem(item)
                self.canvas().update()

    def contentsMousePressEvent(self, e):
        """
        """
        point = self.inverseWorldMatrix().map(e.pos())
        ilist = self.canvas().collisions(point)

        self.__moving_start = None
        self.__moving_scale = 0

        if not len(ilist) or ilist[0].rtti() not in [PC_RTTI_Rect]:
            return

        if self.__active_item!=ilist[0]:
            if self.__active_item is not None:
                self.__active_item.setNormal()
            self.__active_item = ilist[0]
            self.__active_item.setActive()
            zmax = 0
            for item in self.canvas().allItems():
                zmax = max(zmax, item.z())
            self.__active_item.setZ(zmax+1)
            self.canvas().update()

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

    def contentsMouseMoveEvent(self, e):
        """
        """
        if self.__moving_start is None: return
        point = self.inverseWorldMatrix().map(e.pos())
        rect = self.__active_item.getBoundingRect()
        margin = self.marginItem.rect()

        if self.__moving_scale:
            redraw = rect.sizeToIn(self.__moving_scale, point, margin)
        else:
            redraw = rect.moveByIn(point.x()-self.__moving_start.x(),
                      point.y()-self.__moving_start.y(), margin)

        if redraw:
            self.__active_item.setBoundingRect(rect)
            self.canvas().update()

        self.__moving_start = point

    def contentsMouseReleaseEvent(self, e):
        """
        """
        if self.__active_item is not None:
            if self.__active_item.fullRedraw(): self.canvas().update()
        self.setCursor(qt.QCursor(qt.Qt.ArrowCursor))
        self.__moving_start = None
        self.__moving_scale = 0

    def dragEnterEvent(self, de):
        """
        """
        source = de.source()
        if source and hasattr(source, "GetImage"):
            de.accept(1)
        else:    de.accept(0)

    def dropEvent(self, de):
        """
        """
        source = de.source()
        try:
            image = source.GetImage()
            pos = de.pos()
            item = PrintCanvasImage(pos.x(), pos.y(), image, self.canvas())
            self.resizeItem(item)
            item.show()
            self.canvas().update()
        except:
            sys.excepthook(sys.exc_info()[0],
                           sys.exc_info()[1],
                           sys.exc_info()[2])



            
################################################################################
##################             PrintPreview               ###################
################################################################################
class PrintPreview(qt.QDialog):
    """
    PrintPreview is a widget designed to show and manage items to print. It
    is possible to
    - add/remove pixmaps to the PP canvas
    - move and resize pixmaps
    - configure the printer
    - define margin for printing
    - zoom the view of the canvas
    - define title and legend for items
    - print the canvas
    """
    def __init__(self, parent = None, printer = None, name = "PrintPreview", \
                 modal = 0, fl = 0):
        """
        Constructor method:

        """
        qt.QDialog.__init__(self, parent, name, modal, fl)

        self.printer    = None

        # main layout 
        layout = qt.QVBoxLayout(self, 0, -1, "PrintPreview global layout")

        toolBar = qt.QWidget(self)

        # Margin
        marginLabel = qt.QLabel("Margins:", toolBar)    
        self.marginSpin = qt.QSpinBox(0, 50, 10, toolBar)
        self.connect(self.marginSpin, qt.SIGNAL("valueChanged(int)"),    \
                     self.__marginChanged)

        # Scale / Zoom
        scaleLabel = qt.QLabel("Zoom:", toolBar)
        scaleCombo = qt.QComboBox(toolBar)
        self.scaleValues = [20, 40, 60, 80, 100, 150, 200]

        for scale in self.scaleValues:
            scaleCombo.insertItem("%3d %%"%scale)
            
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
        toolsLayout = qt.QHBoxLayout(toolBar, 0, -1, "Tools Layout")

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

        # canvas to display items to print
        self.canvas     = qtcanvas.QCanvas(self)
        self.canvasView = PrintCanvasView(self.canvas, self)

        # status bar
        statusBar = qt.QStatusBar(self)

        self.targetLabel = qt.QLabel( "???", statusBar, "targetLabel")
        statusBar.addWidget(self.targetLabel)

        # finally, building main widget.
        layout.addWidget(toolBar)
        layout.addWidget(self.canvasView)
        layout.addWidget(statusBar)
        
        # use user printer or a default QPrinter
        if printer == None:
            printer = qt.QPrinter()
            
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
        metrics = qt.QPaintDeviceMetrics(self.printer)
        psize = (metrics.width(), metrics.height())
        self.canvasView.setPaperSize(psize[0], psize[1])

        # --- find correct zoom
        wsize = (self.width(), self.height())
        scale = min(float(wsize[0])/float(psize[0]),
                    float(wsize[1])/float(psize[1]))
        iscale = int(100*scale)
        dscale = [ abs(iscale - val) for val in self.scaleValues ]
        iscale = self.scaleValues[dscale.index(min(dscale))]

        self.canvasView.setScale(float(iscale)/100.0)
        self.scaleCombo.setCurrentItem(self.scaleValues.index(iscale))

        # --- possible margin values
        oldv = self.marginSpin.value()
        smax = int(psize[0]/40)*10
        self.marginSpin.setMaxValue(smax)
        self.marginSpin.setLineStep(10)
        margin = self.canvasView.getMargin()
        if margin[0]>smax: 
            self.marginSpin.setValue(smax)
        else:    
            self.marginSpin.setValue(oldv)

        # update output target
        if self.printer.outputToFile():
            self.targetLabel.setText(qt.QString("File:").append(
                self.printer.outputFileName()))
        else:
            self.targetLabel.setText(qt.QString("Printer:").append(
                self.printer.printerName()))
            
        self.update()

    def __marginChanged(self, value):
        """
        """
        self.canvasView.setMargin(value, value)

    def __scaleChanged(self, index):
        """
        """
        self.canvasView.setScale(float(self.scaleValues[index])/100.0)

    def addImage(self, image):
        """
        add an image item to the print preview canvas
        """
        (x,y) = self.canvasView.getMargin()
        self.__addItem(PrintCanvasImage(x+1, y+1, image, self.canvas))

    def addPixmap(self, pixmap):
        """
        add a pixamap to the print preview canvas
        """
        (x,y) = self.canvasView.getMargin()
        self.__addItem(PrintCanvasPixmap(x+1, y+1, pixmap, self.canvas))

    def __addItem(self, item):
        """
        """
        self.canvasView.resizeItem(item)
        item.show()
        self.canvas.update()

    def __setup(self):
        """
        """
        if self.printer is not None:
            if self.printer.setup(): self.updatePrinter()

    def __cancel(self):
        """
        """
        self.reject()

    def __clearAll(self):
        """
        Clear the print preview window, remove all items
        """
        self.canvasView.removeAllItems()
        
    def __remove(self):
        """
        """
        self.canvasView.removeActiveItem()

    def __print(self):
        """
        send all items of the canvas to the printer (file or device)
        """
        prt = qt.QPainter()
        prt.begin(self.printer)
        for item in self.canvasView.getPrintItems():
            item.printTo(prt)
        prt.end()
        self.accept()


################################################################################
#####################    TEST -- PrintPreview  -- TEST   ####################
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
    #p.setPrintToFile(1)
    p.setOutputToFile(1)
    p.setOutputFileName(os.path.splitext(filename)[0]+".ps")
    p.setColorMode(qt.QPrinter.Color)

    w = PrintPreview( parent = None, printer = p, name = 'Print Prev',
                      modal = 0, fl = 0)
    w.resize(400,500)

    w.addPixmap(qt.QPixmap(qt.QImage(filename)))
    #w.addImage(qt.QImage(filename))
    w.addImage(qt.QImage(filename))

    w.exec_loop()

              
##  MAIN   
if  __name__ == '__main__':
    testPreview()
    # testCanvas()
    # testTitle()

 
 
