#/*##########################################################################
# Copyright (C) 2004-2013 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/

__author__ = "T. Rueter - ESRF Data Analysis"
from PyMca import PyMcaQt as qt
from PyMca.PyMca_Icons import IconDict

if hasattr(qt, "QString"):
    print('qt has QString')
    QString = QString
    QStringList = QStringList
else:
    print('qt does not have QString')
    QString = str
    QStringList = list

DEBUG = 1

# Build all symbols
# Curtesy of the pyqtgraph project
Symbols = dict([(name, qt.QPainterPath()) for name in ['o', 's', 't', 'd', '+', 'x']])
Symbols['o'].addEllipse(qt.QRectF(-0.5, -0.5, 1, 1))
Symbols['s'].addRect(qt.QRectF(-0.5, -0.5, 1, 1))
coords = {
    't': [(-0.5, -0.5), (0, 0.5), (0.5, -0.5)],
    'd': [(0., -0.5), (-0.4, 0.), (0, 0.5), (0.4, 0)],
    '+': [
        (-0.5, -0.05), (-0.5, 0.05), (-0.05, 0.05), (-0.05, 0.5),
        (0.05, 0.5), (0.05, 0.05), (0.5, 0.05), (0.5, -0.05), 
        (0.05, -0.05), (0.05, -0.5), (-0.05, -0.5), (-0.05, -0.05)
    ],
}
for k, c in coords.items():
    Symbols[k].moveTo(*c[0])
    for x,y in c[1:]:
        Symbols[k].lineTo(x, y)
    Symbols[k].closeSubpath()
tr = qt.QTransform()
tr.rotate(45)
Symbols['x'] = tr.map(Symbols['+'])

class LegendIcon(qt.QWidget):

    symbols = ['triangle',
               'utriangle',
               'square',
               'cross',
               'circle',
               'ellipse',
               'diamond']

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        # Assign default values
        self.symbol    = None
        self.lineWidth = 4.
        self.color     = qt.Qt.green
        
        # Pen draws outlines of a shape
        self.pen = qt.QPen(self.color,
                           self.lineWidth,
                           qt.Qt.SolidLine,
                           qt.Qt.FlatCap,
                           qt.Qt.RoundJoin)
        # Brush draws interior as a shape
        self.brush = qt.QBrush(qt.Qt.SolidPattern)

        # TODO: Only draws lines
        self.path = qt.QPainterPath()
        self.path.moveTo(0.,50.)
        self.path.lineTo(100.,50.)
        # Control widget size: sizeHint "is the only acceptable
        # alternative, so the widget can never grow or shrink"
        # (c.f. Qt Doc, enum QSizePolicy::Policy)
        self.setSizePolicy(qt.QSizePolicy.Fixed,
                           qt.QSizePolicy.Fixed)
    
    def paintEvent(self, event):
        '''
        :param event: event
        :type event: QPaintEvent
        '''
        painter = qt.QPainter(self)
        self.paint(painter, event.rect(), self.palette())

    def paint(self, painter, rect, palette):
        painter.save() # -> pushes painter state onto a stack
        # Boundary line will be rendered symmetrically
        # around the mathematical shape of an area (rect)
        # when using QPainter.Antialiasing
        painter.setRenderHint(qt.QPainter.Antialiasing);
        painter.setPen(self.pen)
        painter.setBrush(self.brush)
        if self.symbol == 'triangle':
            spath = self._trianglePainterPath()
        elif self.symbol == 'utriangle':
            spath = self._upsideTrianglePainterPath()
        elif self.symbol == 'square':
            spath = self._squarePainterPath()
        elif self.symbol == 'cross':
            spath = self._crossPainterPath()
        elif self.symbol == 'diamond':
            spath = self._diamondPainterPath()
        elif self.symbol == 'circle':
            spath = self._circlePainterPath()
        elif self.symbol == 'ellipse':
            spath = self._ellipsePainterPath()
        lpath = self._linePainterPath()
        offset = rect.topLeft()
        for path in [spath, lpath]:
            path.translate(offset)
            painter.drawPath(path)
        painter.restore() # -> unwinds painter stack

    def sizeHint(self):
        return qt.QSize(50,20)

    def getSymbolRectangle(self, square=True):
        '''
        :param square: Both sides have the same length
        :type square: Bool

        Returns the rectangle in which the symbol is drawn
        '''
        w, h = float(self.width()), float(self.height())
        # Remove width of outline from rectangle
        lineWidth = self.lineWidth / 2.
        # TODO: Check how widget is directed (broader/higher)
        if square:
            rectW = w
        else:
            rectW = 1.3 * w 
        '''
        if square:
            rectW = rectH
            size = qt.QSizeF(rectH, rectW)
        elif rectH < rectW:
            rectW = 1.3 * rectH
            size = qt.QSizeF(rectH, rectW)
        else:
            rectW = 1.3 * rectH
            size = qt.QSizeF(rectW, rectH)
        x, y = (w - rectW)/2., self.lineWidth
        if rectH < rectW:
            # Typical case: broad rectangle..
            topLeft = qt.QPointF(x, y)
        else:
            # ..but maybe the rectangle is rather
            # higher than broad
            topLeft = qt.QPointF(y, x)
        '''
        # Top Left Corner
        topLeftX = (rectW - h)/2. + lineWidth
        topLeftY = lineWidth
        topLeft = qt.QPointF(topLeftX, topLeftY)
        # Bottom Right Corner
        bottomRightX = (rectW + h)/2. - lineWidth
        bottomRightY = h - lineWidth
        bottomRight = qt.QPointF(bottomRightX, bottomRightY)
        return qt.QRectF(topLeft, bottomRight)

    #def drawSymbol(painter, symbol, )

    # Line painter paths
    def _linePainterPath(self):
        path = qt.QPainterPath()
        w, h = float(self.width()), float(self.height())
        path.moveTo(0., h/2.)
        path.lineTo(w, h/2.)
        return path

    # Symbol painter paths
    def _crossPainterPath(self):
        path = qt.QPainterPath()
        rect = self.getSymbolRectangle()
        path.moveTo(rect.topLeft())
        path.lineTo(rect.bottomRight())
        path.moveTo(rect.bottomLeft())
        path.lineTo(rect.topRight())
        return path

    def _trianglePainterPath(self):
        path = qt.QPainterPath()
        rect = self.getSymbolRectangle()
        points = [rect.bottomLeft(),
                  rect.bottomRight(),
                  rect.topLeft() + qt.QPointF(rect.width()/2., 0.)]
        triangle = qt.QPolygonF(points)
        path.addPolygon(triangle)
        return path

    def _upsideTrianglePainterPath(self):
        path = qt.QPainterPath()
        rect = self.getSymbolRectangle()
        points = [rect.topLeft(),
                  rect.topRight(),
                  rect.bottomLeft() + qt.QPointF(rect.width()/2., 0.)]
        triangle = qt.QPolygonF(points)
        path.addPolygon(triangle)
        return path

    def _diamondPainterPath(self):
        path = qt.QPainterPath()
        rect = self.getSymbolRectangle()
        halfWidth  = qt.QPointF(rect.width()/2., 0.)
        halfHeigth = qt.QPointF(0., rect.height()/2.)
        points = [(rect.topLeft() + halfWidth),
                  (rect.topRight() + halfHeigth),
                  (rect.bottomLeft() + halfWidth),
                  (rect.topLeft() + halfHeigth)]
        triangle = qt.QPolygonF(points)
        path.addPolygon(triangle)
        return path

    def _squarePainterPath(self):
        path = qt.QPainterPath()
        rect = self.getSymbolRectangle()
        path.addRect(rect)
        return path

    def _circlePainterPath(self):
        path = qt.QPainterPath()
        rect = self.getSymbolRectangle()
        path.addEllipse(rect)
        return path

    def _ellipsePainterPath(self):
        # TODO: Check getSymbolRectangle(square=False)
        path = qt.QPainterPath()
        rect = self.getSymbolRectangle(square=False)
        path.addEllipse(rect)
        return path

    def setPen(self, pen):
        # Use copy contructor
        self.pen = qt.QPen(pen)
        self.update()

    def setBrush(self, brush):
        # Use copy contructor
        self.brush = qt.QBrush(brush)
        self.update()

    # Modify icon
    def setColor(self, color):
        self.pen.setColor(color)
        self.brush.setColor(color)
        self.update()

    def setSymbol(self, symbol):
        if symbol not in self.symbols:
            raise ValueError('Unknown symbol: %s'%symbol)
        self.symbol = symbol
        self.update()

    # Modify brush
    def setBrushColor(self, color):
        self.brush.setColor(color)
        self.update()

    # Modify pen
    def setPenColor(self, color):
        self.pen.setColor(color)
        self.update()

    def setLineWidth(self, width):
        self.pen.setWidth(width)
        self.update()

class LegendModel(qt.QAbstractListModel):
    iconColorRole     = qt.Qt.UserRole + 0
    iconLineWidthRole = qt.Qt.UserRole + 1
    iconSymbolRole    = qt.Qt.UserRole + 2
    
    def __init__(self, llist=[], parent=None):
        qt.QAbstractListModel.__init__(self, parent)
        self.legendList = []
        for (legend, icon) in llist:
            checkState = LegendListItemWidget
            curveType  = 0
            # Add Item Delegate here?
            item = [legend,
                    icon,
                    qt.Qt.Checked,
                    curveType]
            self.legendList.append(item)
        print('LegendModel Constructor finished..')

    def __getitem__(self, idx):
        if idx >= len(self.legendList):
            raise IndexError('list index out of range')
        return self.legendList[idx]

    def rowCount(self, modelIndex=None):
        return len(self.legendList)

    def flags(self, index):
        return qt.Qt.ItemIsEditable | qt.Qt.ItemIsEnabled

    def data(self, modelIndex, role):
        if modelIndex.isValid:
            idx = modelIndex.row()
        else:
            return qt.QVariant()
        if idx >= len(self.legendList):
            #raise IndexError('list index out of range')
            print('data -- List index out of range, idx: %d'%idx)
            return qt.QVariant()
        
        item = self.legendList[idx]
        if role == qt.Qt.DisplayRole:
            # Data to be rendered in the form of text
            legend = QString(item[0])
            #return qt.QVariant(legend)
            return legend
        elif role == qt.Qt.SizeHintRole:
            #size = qt.QSize(200,50)
            print('LegendModel -- size hint role not implemented')
            return qt.QSize()
        elif role == qt.Qt.TextAlignmentRole:
            alignment = qt.Qt.AlignVCenter | qt.Qt.AlignLeft
            return alignment
        elif role == qt.Qt.BackgroundRole:
            # Background color, must be QBrush
            if idx%2:
                brush = qt.QBrush(qt.QColor(240,240,240))
            else:
                brush = qt.QBrush(qt.Qt.white)
            return brush
        elif role == qt.Qt.ForegroundRole:
            # ForegroundRole color, must be QBrush
            brush = qt.QBrush(qt.Qt.blue)
            return brush
        elif role == qt.Qt.CheckStateRole:
            return item[2] == qt.Qt.Checked
        elif role == qt.Qt.ToolTipRole or role == qt.Qt.StatusTipRole:
            return ''
        elif role == self.iconColorRole:
            return item[1]['color']
        elif role == self.iconLineWidthRole:
            return item[1]['linewidth']
        elif role == self.iconSymbolRole:
            return item[1]['symbol']        
        else:
            print('Unkown role requested: %s',str(role))
            return None

    def setData(self, modelIndex, value, role):
        if modelIndex.isValid:
            idx = modelIndex.row()
        else:
            return None
        if idx >= len(self.legendList):
            #raise IndexError('list index out of range')
            print('setData -- List index out of range, idx: %d'%idx)
            return None

        item = self.legendList[idx]
        try:
            if role == qt.Qt.DisplayRole:
                # Set legend
                item[0] = str(value)
            elif role == self.iconColorRole:
                item[1]['color'] = qt.QColor(value)
            elif role == self.iconLineWidthRole:
                item[1]['linewidth'] = int(value)
            elif role == self.iconSymbolRole:
                item[1]['symbol'] = str(value)
            elif role == qt.Qt.CheckStateRole:
                item[2] = value
        except ValueError:
            if DEBUG == 1:
                print('Conversion failed:'
                     +'\n\tvalue:',value
                     +'\n\trole:',role)
        # Can that be right? Read docs again..
        self.dataChanged.emit(modelIndex, modelIndex)
        return True

    def insertRows(self, row, count, modelIndex):
        '''        
        :param row: After which row comes the insert 
        :type row: int
        :param count: How many items are inserted
        :type count: int
        :param modelIndex: Check for children
        '''
        print('insertRows')
        length = len(self.legendList)
        if row < 0 or row >= length:
            self.endInsertingRows()
            raise IndexError('Index out of range -- '
                            +'idx: %d, len: %d'%(idx, length))
        qt.QAbstractListModel.beginInsertingRows(self,
                                                 modelIndex,
                                                 row,
                                                 row+count)
        head = self.legendList[0:row]
        tail = self.legendList[row:]
        new  = []
        for child in modelIndex.children():
            legend, icon = child
            item = (legend,
                    icon,
                    checkState,
                    curveType)
            new.append(item)
        self.legendList = head + new + tail
        qt.QAbstractListModel.endInsertRows(self)
        return True

    def removeRows(self, row, count, modelIndex):
        print('removeRows')

    def setEditor(self, event, editor):
        '''
        :param event: String that identifies the editor
        :type event: str
        :param editor: Widget used to change data in the underlying model
        :type editor: QWidget
        '''
        if event not in self.eventList:
            raise ValueError('setEditor -- Event must be in'
                            +'%s'%(str(self.eventList)))
        self.editorDict[event] = editor

#class LegendListItemWidget(qt.QStyledItemDelegate):
class LegendListItemWidget(qt.QItemDelegate):
        

    # TODO: Add Icon handling, align icons on the right
    # Notice: LegendListItem does NOT inherit
    # from QObject, it cannot emit signals!

    curveType = 0
    imageType = 1

    def __init__(self, parent=None, itemType=0):
        #qt.QWidget.__init__(self, parent)
        #qt.QStyledItemDelegate.__init__(self, parent)
        qt.QItemDelegate.__init__(self, parent)

        # Keep checkbox and legend to get sizeHint
        self.checkbox = qt.QCheckBox()
        self.checkbox.setCheckState(qt.Qt.Checked)
        
        self.legend = qt.QLabel()
        self.legend.setAlignment(qt.Qt.AlignVCenter |
                                 qt.Qt.AlignLeft)
        self.icon = LegendIcon()
        self.__currentEditor = None
        #self.color = qt.QColor('darkyellow')
        '''
        itemLayout = qt.QHBoxLayout()
        itemLayout.addWidget(self.checkbox)
        itemLayout.addWidget(self.legend)
        itemLayout.addWidget(qt.HorizontalSpacer())
        itemLayout.addWidget(self.icon)
        #self.setLayout(itemLayout)
        '''

        '''
        self.itemType = 1000 + itemType
        self.
        self.legend = legend
        self.currentCheckState = qt.Qt.Checked
        self.lastCheckState    = qt.Qt.Checked
        self.pen = qt.QPen()
        self.textColor = qt.QColor()
        '''
        print('LegendListItemWidget.',type(self.itemEditorFactory()))

    def updateItem(self, ddict):
        keys = ddict.keys()
        label     = ddict['label'] if 'label' in keys else None
        color     = ddict['color'] if 'color' in keys else None
        linewidth = ddict['linewidth'] if 'linewidth' in keys else None
        if 0:
            linewidth = ddict['linewidth'] if 'linewidth' in keys else None
            brush     = ddict['brush'] if 'brush' in keys else None
            style     = ddict['style'] if 'style' in keys else None
            symbol    = ddict['symbol'] if 'symbol' in keys else None
        # Set new legend
        if label:
            self.setText(label)
        # Set 
        if color:
            color = qt.QColor(0, 0, 128)
        else:
            color = qt.QColor(0, 0, 0)
        # Set text color
        self.textColor = color

    def paint(self, painter, option, idx):
        '''
        :param painter:
        :type painter: QPainter
        :param option:
        :type option: QStyleOptionViewItem
        :param idx:
        :type idx: QModelIndex

        Here be docs..
        '''
        rect = option.rect
        # Calculate the checkbox rectangle
        topLeft  = rect.topLeft()
        botRight = qt.QPoint(topLeft.x() + 30,
                             topLeft.y() + rect.height())
        chBoxRect = qt.QRect(topLeft, botRight)
        # Calculate the icon rectangle
        iconSize = self.icon.sizeHint()
        topRight = rect.topRight()
        x = topRight.x() - iconSize.width()
        y = topRight.y() + (rect.height()-iconSize.height()) / 2.
        iconRect = qt.QRect(qt.QPoint(x,y), iconSize)
        # Calculate the label rectangle
        y  = rect.topLeft().y()
        topLeft   = qt.QPoint(rect.topLeft().x() + 31, y)
        botRight  = qt.QPoint(iconRect.bottomRight().x() - 1, y + rect.height())
        labelRect = qt.QRect(topLeft, botRight)

        # Draw background first!
        if option.state & qt.QStyle.State_MouseOver:
            painter.setOpacity(.5) # Control opacity
            painter.fillRect(rect, option.palette.highlight())
            painter.setOpacity(1.) # Reset opacity
        else:
            backgoundBrush = idx.data(qt.Qt.BackgroundRole)
            painter.fillRect(rect, backgoundBrush)

        # Draw the checkbox
        if idx.data(qt.Qt.CheckStateRole):
            checkState = qt.Qt.Checked
        else:
            checkState = qt.Qt.Unchecked
        itemStyle  = qt.QStyleOptionViewItem()
        #itemStyl.
        self.drawCheck(painter, itemStyle, chBoxRect, checkState)

        # Draw label
        legendText = idx.data(qt.Qt.DisplayRole)
        textBrush  = idx.data(qt.Qt.ForegroundRole)
        textAlign  = idx.data(qt.Qt.TextAlignmentRole)
        painter.setBrush(textBrush)
        painter.setFont(self.legend.font())
        painter.drawText(labelRect, textAlign, legendText)

        # Draw icon
        iconColor = idx.data(LegendModel.iconColorRole)
        iconLineWidth = idx.data(LegendModel.iconLineWidthRole)
        iconSymbol = idx.data(LegendModel.iconSymbolRole)
        self.icon.setColor(iconColor)
        self.icon.setLineWidth(iconLineWidth)
        self.icon.setSymbol(iconSymbol)
        self.icon.paint(painter, iconRect, option.palette)
        self.icon.resize(iconRect.size())
        self.icon.move(iconRect.topRight())

    def editorEvent(self, event, model, option, modelIndex):
        # self.createEditor is called first
        if event.button() == qt.Qt.RightButton:
            menu = LegendListContextMenu()
            menu.exec_(event.globalPos())
            return True
        elif event.button() == qt.Qt.LeftButton:
            # Check if checkbox was clicked
            xpos = event.pos().x()
            cbClicked = (xpos >= 10) & (xpos <= 20)
            if cbClicked:
                # Edit checkbox
                print('CB clicked!')
                currentState = modelIndex.data(qt.Qt.CheckStateRole)
                print(str(currentState))
                if currentState:
                    newState = qt.Qt.Unchecked
                else:
                    newState = qt.Qt.Checked
                model.setData(modelIndex, newState, qt.Qt.CheckStateRole)
            event.accept()
            return True
        return qt.QItemDelegate.editorEvent(self, event, model, option, modelIndex)

    def createEditor(self, parent, option, idx):
        return False
        # QColorDialog::QColorDialog(const QColor & initial, QWidget * parent = 0)
        # TODO: Set editor to the items color
        print('createEditor -- type(self.__currentEditor):',
              self.__currentEditor)
        if self.__currentEditor:
            editor = self.__currentEditor(parent=parent)
        else:
            editor = False
        '''
        editor = qt.QColorDialog()
        editor.colorSelected.connect(self.commitColor)
        '''
        return editor

    def commitColor(self, color):
        print('commitColor -- Received color: %s'%str(color))
        # set modelItem to color using self.lastModelItemIdx..
        self.commitData.emit(self.sender())
        self.closeEditor.emit(self.sender(),
                              qt.QAbstractItemDelegate.NoHint)

    def setEditorData(self, editor, idx):
        if not idx.isValid():
            raise IndexError('setEditorData -- invalid index')
        '''
        iconColor = idx.data(LegendModel.iconColorRole)
        print('setEditorData -- Set editor to color: %s'%str(iconColor))
        #editor.blockSignals(True)
        qt.QColorDialog.setCurrentColor(editor, iconColor)
        #editor.blockSignals(False)
        '''
        return True

    def setModelData(self, editor, model, idx):
        if isinstance(editor, qt.QColorDialog):
            value = qt.QColorDialog.currentColor(editor)
            role  = LegendModel.iconColorRole
        res = model.setData(idx, value, role)
        print('setModelData -- change accepted? %s!'%str(res))

    def sizeHint(self, option, idx):
        #return qt.QSize(68,24)
        iconSize = self.icon.sizeHint()
        legendSize = self.legend.sizeHint()
        checkboxSize = self.checkbox.sizeHint()

        height = max([iconSize.height(), legendSize.height(), checkboxSize.height()]) + 4
        width = iconSize.width() + legendSize.width() + checkboxSize.width()

        #print('Delegate.sizeHint -- height: %d, width: %d'%(height, width)) # height: 20, width: 68
        return qt.QSize(height, width)

class LegendListView(qt.QListView):

    sigMouseClicked = qt.pyqtSignal(object)
    rightClicked = qt.pyqtSignal(qt.QModelIndex, qt.QEvent)
    leftClicked = qt.pyqtSignal(qt.QModelIndex, qt.QEvent)
    __mouseClickedEvent  = 'mouseClicked'
    __legendClickedEvent = 'legendClicked'
    eventList = ['rightClick', 'legendClick', 'checkboxClick']
    
    def __init__(self, parent=None):
        qt.QListWidget.__init__(self, parent)
        self.__lastButton   = None
        self.__lastClickPos = None
        self.__lastModelIdx = None
        # Set default delegate
        self.setItemDelegate(LegendListItemWidget())
        # Set default editors
        self.editorDict = {
                'rightClick': None,
                'checkboxClick': None,
                'legendClick': None
        }
        # Connects
        self.clicked.connect(self._handleMouseClick)
        
        self.setSizePolicy(qt.QSizePolicy.MinimumExpanding,
                           qt.QSizePolicy.MinimumExpanding)
        # Set edit triggers by hand using self.edit(QModelIndex)
        # in mousePressEvent (better to control than signals)
        #self.setEditTriggers(
        #     qt.QAbstractItemView.NoEditTriggers
        #)
        # Control layout
        #self.setBatchSize(2)
        #self.setLayoutMode(qt.QListView.Batched)
        #self.setFlow(qt.QListView.LeftToRight)
        self.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)

    def sizeHint(self):
        return qt.QSize(300,500)

    def __getitem__(self, idx):
        model = self.model()
        try:
            item = model[idx]
        except ValueError:
            item = None
        return item

    def setEditor(self, event, editor):
        '''
        :param event: String that identifies the event
        :type event: str
        :param editor: Widget used to change data in the underlying model
        :type editor: QWidget

        Assigns editor to the delegate to be used in case of event.
        '''
        delegate = self.itemDelegate()
        if not hasattr(delegate, 'setEditor'):
            NotImplementedError('triggerEditor -- Delegate needs setEditor function')
        if event not in self.eventList:
            raise ValueError('setEditor -- Event must be in'
                            +'%s'%(str(self.eventList)))
        delegate.setEditor(event, editor)

    def mousePressEvent(self, event):
        self.__lastButton = event.button()
        self.__lastPosition = event.pos()
        idx = self.indexAt(self.__lastPosition)
        #self.edit(idx, qt.QAbstractItemView.NoEditTriggers, event)
        #print('self.mousePressEvent called')
        qt.QListView.mousePressEvent(self, event)

    def _handleMouseClick(self, modelIndex):
        '''
        :param modelIndex: index of the clicked item
        :type modelIndex: QModelIndex

        Distinguish between mouse click on Legend
        and mouse click on CheckBox by setting the
        currentCheckState attribute in LegendListItem.

        Emits signal sigMouseClicked(ddict)
        '''
        print('self._handleMouseClick called')
        if self.__lastButton not in [qt.Qt.LeftButton,
                                     qt.Qt.RightButton]:
            return
        if not modelIndex.isValid():
            print('_handleMouseClick -- Invalid QModelIndex')
            return
        model = self.model()
        idx   = modelIndex.row()
        xpos  = self.__lastPosition.x()
        # TODO: make cbClicked geometry independent
        cbClicked = (xpos >= 10) & (xpos <= 20)
        # item is tupel: (legend, icon, checkState, curveType)
        item  = model[idx]
        ddict = {
            'legend'   : str(item[0]),
            'icon'     : item[1],
            'selected' : item[2] == qt.Qt.Checked,
            'type'     : item[3]
        }
        if self.__lastButton == qt.Qt.RightButton:
            if DEBUG == 1:
                print('Right clicked')
            ddict['button'] = qt.Qt.RightButton
            ddict['event']  = self.__mouseClickedEvent
        elif cbClicked:
            if DEBUG == 1:
                print('CheckBox clicked')
            ddict['button'] = qt.Qt.LeftButton
            ddict['event']  = self.__mouseClickedEvent
        else:
            if DEBUG == 1:
                print('Legend clicked')
            ddict['button'] = qt.Qt.LeftButton
            ddict['event']  = self.__legendClickedEvent
        if DEBUG == 1:
            print('  idx: %d\n  ddict: %s'%(idx, str(ddict)))
        self.sigMouseClicked.emit(ddict)

class LegendListContextMenu(qt.QMenu):
    def __init__(self, parent=None):
        qt.QMenu.__init__(self, parent)
        actionList = ['Set Active',
                      'Toggle points',
                      'Toggle lines',
                      'Remove curve']
        for elem in actionList:
            self.addAction(elem)

class Notifier(qt.QObject):
    def __init__(self):
        qt.QObject.__init__(self)
        self.chk = True

    def signalReceived(self, **kw):
        obj = self.sender()
        print('NOTIFIER -- signal received\n\tsender:',str(obj))

if __name__ == '__main__':
    notifier = Notifier()
    legends = ['Legend0', 'Legend1', 'Long Legend 2', 'Foo Legend 3', 'Even Longer Legend 4', 'Short Leg 5']
    colors  = [qt.Qt.darkRed, qt.Qt.green, qt.Qt.yellow, qt.Qt.darkCyan, qt.Qt.blue, qt.Qt.darkBlue, qt.Qt.red]
    symbols = ['circle', 'triangle', 'utriangle', 'diamond', 'square', 'cross']
    app = qt.QApplication([])
    win = LegendListView()
    #win = LegendListContextMenu()
    #win = qt.QWidget()
    #layout = qt.QVBoxLayout()
    #layout.setContentsMargins(0,0,0,0)
    llist = []
    for idx, (l, c, s) in enumerate(zip(legends, colors, symbols)):
        ddict = {
            'color': qt.QColor(c),
            'linewidth': 4,
            'symbol': s,
        }
        legend = l
        llist.append((legend, ddict))
        #item = qt.QListWidgetItem(win)
        #legendWidget = LegendListItemWidget(l)
        #legendWidget.icon.setSymbol(s)
        #legendWidget.icon.setColor(qt.QColor(c))
        #layout.addWidget(legendWidget)
        #win.setItemWidget(item, legendWidget)
    #win = LegendListItemWidget('Some Legend 1')
    #print(llist)
    model = LegendModel(llist=llist)
    win.setModel(model)
    #print('Edit triggers: %d'%win.editTriggers())
    
    #win = LegendListWidget(None, legends)
    #win[0].updateItem(ddict)
    #win.setLayout(layout)
    win.sigMouseClicked.connect(notifier.signalReceived)
    win.show()
    
    app.exec_()
            
'''
GRAVEYARD

    def _triggerEditor(self, ddict):
        delegate = self.itemDelegate()
        if isinstance(delegate, qt.QStyledItemDelegate) or\
           isinstance(delegate, qt.QItemDelegate):
            NotImplementedError('triggerEditor -- Use nontrivial delegate')
        button   = ddict['button']
        event    = ddict['event']
        selected = ddict['selected']
        if button == qt.Qt.RightButton:
            if DEBUG == 1:
                print('triggerEditor -- Trigger rightClick Editor')
            editor = self.editorDict['rightClick']
        elif not selected: # TODO: Switch off editor rather than cbClicked
            if DEBUG == 1:
                print('triggerEditor -- Trigger checkboxClick Editor')
            editor = self.editorDict['checkboxClick']
        elif event == self.__legendClickedEvent:
            if DEBUG == 1:
                print('triggerEditor -- Trigger legendClick Editor')
            editor = self.editorDict['legendClick']
        else:
            if DEBUG == 1:
                print('triggerEditor -- No case applies')
            return
        if editor:
            print('triggerEditor -- Create Editor..')
        else:
            #raise ValueError('triggerEditor -- Signal not connected')
            print('triggerEditor -- Signal connction is faulty')
'''
