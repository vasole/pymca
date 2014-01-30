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
Symbols['o'].addEllipse(qt.QRectF(.1, .1, .8, .8))
Symbols['s'].addRect(qt.QRectF(.1, .1, .8, .8))

coords = {
    't': [(0.5, 0.), (.1,.8), (.9, .8)],
    'd': [(0.1, 0.5), (0.5, 0.), (0.9, 0.5), (0.5, 1.)],
    '+': [(0.0, 0.40), (0.40, 0.40), (0.40, 0.), (0.60, 0.),
          (0.60, 0.40), (1., 0.40), (1., 0.60), (0.60, 0.60),
          (0.60, 1.), (0.40, 1.), (0.40, 0.60), (0., 0.60)],
    'x': [(0.0, 0.40), (0.40, 0.40), (0.40, 0.), (0.60, 0.),
          (0.60, 0.40), (1., 0.40), (1., 0.60), (0.60, 0.60),
          (0.60, 1.), (0.40, 1.), (0.40, 0.60), (0., 0.60)]
}
for s, c in coords.items():
    Symbols[s].moveTo(*c[0])
    for x,y in c[1:]:
        Symbols[s].lineTo(x, y)
    Symbols[s].closeSubpath()
tr = qt.QTransform()
tr.rotate(45)
Symbols['x'].translate(qt.QPointF(-0.5,-0.5))
Symbols['x'] = tr.map(Symbols['x'])
Symbols['x'].translate(qt.QPointF(0.5,0.5))

class LegendIcon(qt.QWidget):

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        # Visibilities
        self.showLine   = True
        self.showSymbol = True

        # Line attributes
        self.lineStyle = qt.Qt.SolidLine
        self.lineWidth = 1.
        self.lineColor = qt.Qt.green

        self.symbol     = ''
        # Symbol attributes
        self.symbolStyle = qt.Qt.SolidPattern
        self.symbolColor = qt.Qt.green
        self.symbolOutlineBrush = qt.QBrush(qt.Qt.white)

        # Control widget size: sizeHint "is the only acceptable
        # alternative, so the widget can never grow or shrink"
        # (c.f. Qt Doc, enum QSizePolicy::Policy)
        self.setSizePolicy(qt.QSizePolicy.Fixed,
                           qt.QSizePolicy.Fixed)

    def sizeHint(self):
        return qt.QSize(50,20)

    # Modify Symbol
    def setSymbol(self, symbol):
        if symbol not in Symbols:
            raise ValueError('Unknown symbol: \'%s\''%symbol)
        self.symbol = symbol
        # self.update() after set...?
        # Does not seem necessary

    def setSymbolColor(self, color):
        '''
        :param color: determines the symbol color
        :type style: qt.QColor
        '''
        self.symbolColor = qt.QColor(color)

    def setSymbolStyle(self, style):
        '''
        :param style: Must be in Qt.BrushStyle
        :type style: int

        Possible joices are:
          Qt.NoBrush
          Qt.SolidPattern
          Qt.Dense1Pattern
          Qt.Dense2Pattern
          Qt.Dense3Pattern
          Qt.Dense4Pattern
          Qt.Dense5Pattern
          Qt.Dense6Pattern
          Qt.Dense7Pattern
          Qt.HorPattern
          Qt.VerPattern
          Qt.CrossPattern
          Qt.BDiagPattern
          Qt.FDiagPattern
          Qt.DiagCrossPattern
          Qt.LinearGradientPattern
          Qt.ConicalGradientPattern
          Qt.RadialGradientPattern
        '''
        if style not in list(range(18)):
            raise ValueError('Unknown style: %d')
        self.symbolStyle = int(style)

    # Modify Line
    def setLineColor(self, color):
        self.lineColor = qt.QColor(color)

    def setLineWidth(self, width):
        self.lineWidth = float(width)

    def setLineStyle(self, style):
        '''
        :param style: Must be in Qt.PenStyle
        :type style: int

        Possible joices are:
          Qt.NoPen
          Qt.SolidLine
          Qt.DashLine
          Qt.DotLine
          Qt.DashDotLine
          Qt.DashDotDotLine
          Qt.CustomDashLine
        '''
        if style not in list(range(7)):
            raise ValueError('Unknown style: %d')
        self.lineStyle = int(style)

    # Paint
    def paintEvent(self, event):
        '''
        :param event: event
        :type event: QPaintEvent
        '''
        painter = qt.QPainter(self)
        self.paint(painter, event.rect(), self.palette())

    def paint(self, painter, rect, palette):
        painter.save()
        painter.setRenderHint(qt.QPainter.Antialiasing)
        # Scale painter to the icon height
        # current -> width = 2.5, height = 1.0
        scale  = float(self.height())
        ratio  = float(self.width()) / scale
        painter.scale(scale,
                      scale)
        symbolOffset = qt.QPointF(.5*(ratio-1.), 0.)
        # Determine and scale offset
        offset = qt.QPointF(
                    float(rect.left())/scale,
                    float(rect.top())/scale)
        # Draw BG rectangle (for debugging)
        #bottomRight = qt.QPointF(
        #    float(rect.right())/scale,
        #    float(rect.bottom())/scale)               
        #painter.fillRect(qt.QRectF(offset, bottomRight),
        #                 qt.QBrush(qt.Qt.green))
        llist = []
        if self.showLine:
            linePath = qt.QPainterPath()
            linePath.moveTo(0.,0.5)
            linePath.lineTo(ratio,0.5)
            #linePath.lineTo(2.5,0.5)
            linePen = qt.QPen(
                qt.QBrush(self.lineColor),
                (self.lineWidth / self.height()),
                self.lineStyle,
                qt.Qt.FlatCap
            )
            llist.append((linePath,
                          linePen,
                          qt.QBrush(self.lineColor)))
        if self.showSymbol and len(self.symbol):
            # PITFALL ahead: Let this be a warning to other
            #symbolPath = Symbols[self.symbol]
            # Copy before translate! Dict is a mutable type
            symbolPath = qt.QPainterPath(Symbols[self.symbol])
            symbolPath.translate(symbolOffset)
            symbolBrush = qt.QBrush(
                self.symbolColor,
                self.symbolStyle
            )
            symbolPen = qt.QPen(
                self.symbolOutlineBrush, # Brush
                1./self.height(),        # Width
                qt.Qt.SolidLine          # Style
            )
            llist.append((symbolPath,
                          symbolPen,
                          symbolBrush))
        # Draw
        for path, pen, brush in llist:
            path.translate(offset)
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawPath(path)
        painter.restore()

class LegendModel(qt.QAbstractListModel):
    iconColorRole     = qt.Qt.UserRole + 0
    iconLineWidthRole = qt.Qt.UserRole + 1
    showLineRole      = qt.Qt.UserRole + 2
    iconSymbolRole    = qt.Qt.UserRole + 3
    showSymbolRole    = qt.Qt.UserRole + 4
    legendTypeRole    = qt.Qt.UserRole + 5
    
    def __init__(self, llist=[], parent=None):
        qt.QAbstractListModel.__init__(self, parent)
        self.legendList = []
        for (legend, icon) in llist:
            checkState = LegendListItemWidget
            showLine = True
            showSymbol = True
            curveType  = 0
            item = [legend,
                    icon,
                    qt.Qt.Checked,
                    showLine,
                    showSymbol,
                    curveType]
            self.legendList.append(item)

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
            raise IndexError('list index out of range')
        
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
        elif role == self.showLineRole:
            return item[3]
        elif role == self.showSymbolRole:
            return item[4]
        elif role == self.legendTypeRole:
            return 0 # item[4] ..curveType..
        elif role == qt.Qt.EditRole:
            print('### EDIT ROLE REQUESTED ###')
            return qt.QString('What now?')
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
            elif role == self.showLineRole:
                item[3] = value
            elif role == self.showSymbolRole:
                item[4] = value
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
            showLine = True
            showSymbol = True
            curveType  = 0
            item = [legend,
                    icon,
                    qt.Qt.Checked,
                    showLine,
                    showSymbol,
                    curveType]
            new.append(item)
        self.legendList = head + new + tail
        qt.QAbstractListModel.endInsertRows(self)
        return True

    def removeRows(self, row, count, modelIndex):
        print('removeRows')
        length = len(self.legendList)
        if row < 0 or row >= length:
            self.endInsertingRows()
            raise IndexError('Index out of range -- '
                            +'idx: %d, len: %d'%(idx, length))
        if count == 0:
            return False
        qt.QAbstractListModel.beginRemoveRows(self,
                                              modelIndex,
                                              row,
                                              row+count)
        del(self.legendList[row:row+count])
        qt.QAbstractListModel.endRemoveRows(self)
        return True

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

class LegendListItemWidget(qt.QAbstractItemDelegate):
        
    # Notice: LegendListItem does NOT inherit
    # from QObject, it cannot emit signals!

    curveType = 0
    imageType = 1

    def __init__(self, parent=None, itemType=0):
        qt.QItemDelegate.__init__(self, parent)

        # Dictionary to render checkboxes
        self.cbDict = {}
        self.labelDict = {}
        self.iconDict = {}

        # Keep checkbox and legend to get sizeHint
        self.checkbox = qt.QCheckBox()
        self.legend = qt.QLabel()
        self.icon = LegendIcon()

        # Context Menu and Editors
        self.contextMenu = None

    def paint(self, painter, option, modelIndex):
        '''
        :param painter:
        :type painter: QPainter
        :param option:
        :type option: QStyleOptionViewItem
        :param modelIndex:
        :type modelIndex: QModelIndex

        Here be docs..
        '''
        painter.save()
        # Rect geometry
        width  = option.rect.width()
        height = option.rect.height()
        left   = option.rect.left()
        top    = option.rect.top()
        rect = qt.QRect(qt.QPoint(left, top),
                        qt.QSize(width, height))
        rect = option.rect

        # Calculate the icon rectangle
        iconSize = self.icon.sizeHint()
        # Calculate icon position
        x = rect.left() + 2
        y = rect.top() + int(.5*(rect.height()-iconSize.height()))
        iconRect = qt.QRect(qt.QPoint(x,y), iconSize)

        # Calculate label rectangle
        legendSize = qt.QSize(
                        rect.width() - iconSize.width() - 30,
                        rect.height())
        # Calculate label position
        x = rect.left() + iconRect.width()
        y = rect.top()
        labelRect = qt.QRect(qt.QPoint(x, y),
                             legendSize)
        labelRect.translate(qt.QPoint(10, 0))

        # Calculate the checkbox rectangle
        x = rect.right() - 30
        y = rect.top()
        chBoxRect = qt.QRect(qt.QPoint(x, y),
                             rect.bottomRight())

        # Remember the rectangles
        idx = modelIndex.row()
        self.iconDict[idx] = iconRect
        self.labelDict[idx] = labelRect

        # Draw background first!
        if option.state & qt.QStyle.State_MouseOver:
            backgoundBrush = option.palette.highlight()
        else:
            backgoundBrush = modelIndex.data(qt.Qt.BackgroundRole)
        painter.fillRect(rect, backgoundBrush)

        # Draw label
        legendText = modelIndex.data(qt.Qt.DisplayRole)
        textBrush  = modelIndex.data(qt.Qt.ForegroundRole)
        textAlign  = modelIndex.data(qt.Qt.TextAlignmentRole)
        painter.setBrush(textBrush)
        painter.setFont(self.legend.font())
        painter.drawText(labelRect, textAlign, legendText)

        # Draw icon
        iconColor = modelIndex.data(LegendModel.iconColorRole)
        iconLineWidth = modelIndex.data(LegendModel.iconLineWidthRole)
        iconSymbol = modelIndex.data(LegendModel.iconSymbolRole)
        icon = LegendIcon()
        icon.resize(iconRect.size())
        icon.move(iconRect.topRight())
        icon.showSymbol = modelIndex.data(LegendModel.showSymbolRole)
        icon.showLine = modelIndex.data(LegendModel.showLineRole)
        icon.setSymbolColor(iconColor)
        icon.setLineColor(iconColor)
        icon.setLineWidth(iconLineWidth)
        icon.setSymbol(iconSymbol)
        icon.symbolOutlineBrush = backgoundBrush
        icon.paint(painter, iconRect, option.palette)
        
        # Draw the checkbox
        if modelIndex.data(qt.Qt.CheckStateRole):
            checkState = qt.Qt.Checked
        else:
            checkState = qt.Qt.Unchecked
        # Remember the paint device
        originalPaintDevice = painter.device()
        # Painter needs to end before
        painter.end()
        try:
            cb = self.cbDict[idx]
        except KeyError:
            cb = qt.QCheckBox()
            self.cbDict[idx]= cb
        cb.setCheckState(checkState)
        cb.setGeometry(chBoxRect)
        cb.render(painter.device(),
                             chBoxRect.topLeft(),
                             qt.QRegion(),
                             qt.QWidget.DrawChildren)

        # Reset painter
        painter.begin(originalPaintDevice)
        painter.restore()
        return

    def editorEvent(self, event, model, option, modelIndex):
        # From the docs:
        # Mouse events are sent to editorEvent()
        # even if they don't start editing of the item.
        if event.button() == qt.Qt.RightButton and self.contextMenu:
            self.contextMenu.exec_(event.globalPos(), modelIndex)
            return True
        elif event.button() == qt.Qt.LeftButton:
            # Check if checkbox was clicked
            for cb in self.cbDict.values():
                cbRect = cb.geometry()
                cbClicked = cbRect.contains(event.pos())
                if cbClicked:
                    break
            if cbClicked:
                # Edit checkbox
                currentState = modelIndex.data(qt.Qt.CheckStateRole)
                if currentState:
                    newState = qt.Qt.Unchecked
                else:
                    newState = qt.Qt.Checked
                #self.checkbox.setCheckState(newState)
                idx = modelIndex.row()
                self.cbDict[idx].setCheckState(newState)
                model.setData(modelIndex, newState, qt.Qt.CheckStateRole)
            event.ignore()
            return True
        else:
            return qt.QAbstractItemDelegate.editorEvent(self, event, model, option, modelIndex)

    def createEditor(self, parent, option, idx):
        print('### Editor request ###')

    def sizeHint(self, option, idx):
        #return qt.QSize(68,24)
        iconSize = self.icon.sizeHint()
        legendSize = self.legend.sizeHint()
        checkboxSize = self.checkbox.sizeHint()
        height = max([iconSize.height(), legendSize.height(), checkboxSize.height()]) + 4
        width = iconSize.width() + legendSize.width() + checkboxSize.width()
        return qt.QSize(width, height)

class LegendListView(qt.QListView):

    sigMouseClicked = qt.pyqtSignal(object)
    __mouseClickedEvent  = 'mouseClicked'
    __legendClickedEvent = 'legendClicked'
    
    def __init__(self, parent=None):
        qt.QListWidget.__init__(self, parent)
        self.__lastButton   = None
        self.__lastClickPos = None
        self.__lastModelIdx = None
        # Set default delegate
        self.setItemDelegate(LegendListItemWidget())
        # Set default editors
        self.setSizePolicy(qt.QSizePolicy.MinimumExpanding,
                           qt.QSizePolicy.MinimumExpanding)
        # Set edit triggers by hand using self.edit(QModelIndex)
        # in mousePressEvent (better to control than signals)
        self.setEditTriggers(
             qt.QAbstractItemView.NoEditTriggers
        )
        # Control layout
        #self.setBatchSize(2)
        #self.setLayoutMode(qt.QListView.Batched)
        #self.setFlow(qt.QListView.LeftToRight)
        self.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)

    def sizeHint(self):
        print('ListView.sizeHint called')
        return qt.QSize(300,500)

    def minimumWidth(self):
        print('ListView.minimumSize called')
        return 500

    def minimumSize(self):
        print('ListView.minimumSize called')
        return qt.QSize(300,500)

    def minimumSizeHint(self):
        print('ListView.minimumSizeHint called')
        return qt.QSize(300,500)

    def setContextMenu(self):#, actionList):
        delegate = self.itemDelegate()
        if isinstance(delegate, LegendListItemWidget) and self.model():
            delegate.contextMenu = LegendListContextMenu(self.model())

    def __getitem__(self, idx):
        model = self.model()
        try:
            item = model[idx]
        except ValueError:
            item = None
        return item

    def mousePressEvent(self, event):
        self.__lastButton = event.button()
        self.__lastPosition = event.pos()
        qt.QListView.mousePressEvent(self, event)
        # call _handleMouseClick after editing was handled
        # If right click (context menu) is aborted, no
        # signal is emitted..
        self._handleMouseClick(
                self.indexAt(self.__lastPosition))

    def mouseReleaseEvent(self, event):
        if DEBUG == 1:
            print('LegendListView.mouseReleaseEvent -- '
                 +'is overwritten to subpress unwanted '
                 +'behavior in the delegate.')

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

        delegate = self.itemDelegate()
        if isinstance(delegate, LegendListItemWidget):
            cbClicked = False
            for cb in delegate.cbDict.values():
                cbRect = cb.geometry()
                if cbRect.contains(self.__lastPosition):
                    cbClicked = True
                    break

        # TODO: Check for doubleclicks on legend/icon and spawn editors
                
        # item is tupel: (legend, icon, checkState, curveType)
        item  = model[idx]
        ddict = {
            'legend'   : modelIndex.data(qt.Qt.DisplayRole),
            'icon'     : {
                'linewidth' : modelIndex.data(LegendModel.iconLineWidthRole),
                'symbol'    : modelIndex.data(LegendModel.iconSymbolRole),
                'color'     : modelIndex.data(LegendModel.legendTypeRole)
            },
            'selected' : modelIndex.data(qt.Qt.CheckStateRole),
            'type'     : modelIndex.data()
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

class BaseContextMenu(qt.QMenu):
    def __init__(self, model):
        qt.QMenu.__init__(self, parent=None)
        self.model = model
        print('Model type:',type(model))
        
    def exec_(self, pos, idx):
        self.__currentIdx = idx
        qt.QMenu.exec_(self, pos)

    def currentIdx(self):
        return self.__currentIdx

class LegendListContextMenu(BaseContextMenu):

    def __init__(self, model):
        BaseContextMenu.__init__(self, model)
        actionList = [('Set Active', self.setActiveAction),
                      ('Toggle points', self.togglePointsAction),
                      ('Toggle lines', self.toggleLinesAction),
                      ('Remove curve', self.removeCurveAction)]
        for name, action in actionList:
            self.addAction(name, action)

    def removeCurveAction(self):
        idx = self.currentIdx()
        self.model.removeRow(idx.row())
        if DEBUG == 1:
            print('LegendListContextMenu.removeCurveAction called')

    def toggleLinesAction(self):
        idx = self.currentIdx()
        if idx.data(LegendModel.showLineRole):
            model.setData(idx, False, LegendModel.showLineRole)
            if DEBUG == 1:
                print('togglePointsAction -- lines turned off')
        else:
            model.setData(idx, True, LegendModel.showLineRole)
            if DEBUG == 1:
                print('togglePointsAction -- lines turned on')
            
    def togglePointsAction(self):
        idx = self.currentIdx()
        if idx.data(LegendModel.showSymbolRole):
            model.setData(idx, False, LegendModel.showSymbolRole)
            if DEBUG == 1:
                print('togglePointsAction -- Symbols turned off')
        else:
            model.setData(idx, True, LegendModel.showSymbolRole)
            if DEBUG == 1:
                print('togglePointsAction -- Symbols turned on')

    def setActiveAction(self):
        idx = self.currentIdx()
        legend = idx.data(qt.Qt.DisplayRole)
        print('setActiveAction -- active curve:',legend)

class Notifier(qt.QObject):
    def __init__(self):
        qt.QObject.__init__(self)
        self.chk = True

    def signalReceived(self, **kw):
        obj = self.sender()
        print('NOTIFIER -- signal received\n\tsender:',str(obj))

if __name__ == '__main__':
    notifier = Notifier()
    legends = 1*['Legend0', 'Legend1', 'Long Legend 2', 'Foo Legend 3', 'Even Longer Legend 4', 'Short Leg 5']
    colors  = 1*[qt.Qt.darkRed, qt.Qt.green, qt.Qt.yellow, qt.Qt.darkCyan, qt.Qt.blue, qt.Qt.darkBlue, qt.Qt.red]
    #symbols = ['circle', 'triangle', 'utriangle', 'diamond', 'square', 'cross']
    symbols = 1*['o', 't', '+', 'x', 's', 'd']
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
    win.setContextMenu()
    #print('Edit triggers: %d'%win.editTriggers())
    
    #win = LegendListWidget(None, legends)
    #win[0].updateItem(ddict)
    #win.setLayout(layout)
    win.sigMouseClicked.connect(notifier.signalReceived)
    win.show()
    
    app.exec_()
