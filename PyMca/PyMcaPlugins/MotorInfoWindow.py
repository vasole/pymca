__author__ = 'Tonn Rueter'
from PyMca import PyMcaQt as qt
from PyMca.PyMca_Icons import IconDict

DEBUG = 0
class MotorInfoComboBox(qt.QComboBox):
    
    loadColumnSignal = qt.pyqtSignal(object)
    
    def __init__(self, parent, mlist, nCol):
        qt.QComboBox.__init__(self,  parent)
        self.motorNamesList = [""] + mlist
        self.nColumn = nCol
        self.addItems([qt.QString(elem) for elem in self.motorNamesList])
        self.activated.connect(self.emitLoadColumnSignal)

    def emitLoadColumnSignal(self):
        ddict = {}
        ddict['column'] = self.nColumn
        ddict['motor'] = str(self.currentText())
        ddict['event'] = "activated"
        self.loadColumnSignal.emit(ddict)

    def currentMotor(self):
        return str(self.currentText())
        
    def updateMotorNamesList(self, newMotorNamesList):
        currentMotorName = self.currentMotor()
        self.clear()
        newMotorNamesList = [''] + newMotorNamesList
        self.motorNamesList = newMotorNamesList 
        if currentMotorName in self.motorNamesList:
            newIndex = newMotorNamesList.index(currentMotorName)
        else:
            newIndex = 0
        self.addItems( [ qt.QString(elem) for elem in self.motorNamesList ] )
        self.setCurrentIndex(newIndex)

class MotorInfoHeader(qt.QHeaderView):
    
    xOffsetLeft = 5
    xOffsetRight = -35
    yOffset = -1
    
    def __init__(self, parent):
        qt.QHeaderView.__init__(self, qt.Qt.Horizontal, parent)
        self.boxes = []
        self.sectionResized.connect( self.handleSectionResized )
        self.setClickable(True)
        self.setDefaultSectionSize(120)
        self.setMinimumSectionSize(120)

    def showEvent(self, event):
        if self.boxes == []:
            self.boxes = [None] * self.count()
        for i in range(1, self.count()):
            if not self.boxes[i]:
                newBox = MotorInfoComboBox(self, self.parent().motorNamesList, i)
                newBox.loadColumnSignal.connect(self.parent().loadColumn)
                newBox.resize(self.sectionSize(i) - 30, self.height())
                self.boxes[i] = newBox
            self.boxes[i].setGeometry(self.sectionViewportPosition(i) + self.xOffsetLeft, 
                                      self.yOffset, 
                                      self.sectionSize(i) +  self.xOffsetRight, 
                                      self.height())
            self.boxes[i].show()
        qt.QHeaderView.showEvent(self, event)

    def handleSectionResized(self, index):
        for i in range(self.visualIndex(index), len(self.boxes)):
            if i > 0:
                logical = self.logicalIndex (i)
                self.boxes[i].setGeometry(self.sectionViewportPosition(logical) + self.xOffsetLeft, 
                                          self.yOffset,
                                          self.sectionSize(logical) +  self.xOffsetRight,
                                          self.height())

    def deleteLastSection(self):
        self.boxes[-1].close()
        del( self.boxes[-1] )

    def addLastSection(self):
        i = self.count()-1
        newBox = MotorInfoComboBox(self, self.parent().motorNamesList, i)
        newBox.loadColumnSignal.connect(self.parent().loadColumn)
        newBox.setGeometry(self.sectionViewportPosition(i) + self.xOffsetLeft, 
                           self.yOffset, 
                           self.sectionSize(i) +  self.xOffsetRight, 
                           self.height() )
        newBox.show()
        self.boxes += [newBox]

    def fixComboPositions(self):
        for i in range(1, self.count()):
            self.boxes[i].setGeometry(self.sectionViewportPosition(i) + self.xOffsetLeft, 
                                      self.yOffset, 
                                      self.sectionSize(i) +  self.xOffsetRight, 
                                      self.height())

class MotorInfoTable(qt.QTableWidget):
    def __init__(self, parent, numRows, numColumns, legList, motList):
        qt.QTableWidget.__init__(self, 0, numColumns, parent)
        self.currentComboBox = 1
        self.legendsList = legList
        self.motorsList  = motList
        self.motorNamesList = self.getAllMotorNames()
        self.motorNamesList.sort()
        self.infoDict = dict( zip( self.legendsList, self.motorsList ) )
        self.header = MotorInfoHeader(self)
        self.setHorizontalHeader(self.header)
        self.setHorizontalHeaderItem(0, qt.QTableWidgetItem('Legend'))
        self.setSortingEnabled(True)
        self.verticalHeader().hide()
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setShowGrid(False)
        for i in range(len(self.legendsList)):
            curveLegend = self.legendsList[i]
            self.insertRow(i)
            self.setItem(i, 0, curveLegend )
            for j in range(1, self.columnCount()):
                self.setItem(0, j, '')
        self.sortByColumn ( 0, qt.Qt.AscendingOrder )

    def addColumn(self):
        currentColumn = self.columnCount()
        self.insertColumn(currentColumn)
        self.header.addLastSection()

    def delColumn(self):
        if self.columnCount() > 1:
            self.removeColumn(self.columnCount()-1)
            self.header.deleteLastSection()

    def fillRow(self, currentRow):
        legend = self.legendsList[currentRow]
        self.setItem(currentRow, 0, legend )

    def updateTable(self, legList, motList):
        if DEBUG:
            print("updateTable received lengths = ", len(legList), len(motList))
            print("updateTable received legList = ", legList)
            print("updateTable received motList = ", motList)
        if legList is None:
            nItems = 0
        else:
            nItems = len(legList)
        if self.legendsList == legList and self.motorsList == motList:
            if DEBUG:
                print("Ignoring update, no changes")
        else:
            nRows = self.rowCount()
            if nRows != nItems:
                self.setRowCount(nItems)
            self.infoDict = dict(zip(legList, motList))
            self.legendsList = legList
            self.motorsList = motList
            motorNamesList = self.getAllMotorNames()
            motorNamesList.sort()
            for i in range(0, self.columnCount()):
                cBox = self.header.boxes[i]
                if cBox is not None:
                    cBox.updateMotorNamesList(motorNamesList)
            self.motorNamesList = motorNamesList
            for i in range(len(legList)):
                self.fillRow(i)
            for i in range(0, self.columnCount()):
                cBox = self.header.boxes[i]
                if cBox is not None:
                    cBox.emitLoadColumnSignal()

    def loadColumn(self, ddict):
        for key in ddict.keys():
            if str(key) == str("motor"):
                motorName = ddict[key]
            elif str(key) == str("column"):
                column = ddict[key]
        if len(motorName) > 0:
            for i in range(self.rowCount()):
                legend = str( self.item(i, 0).text() )
                curveInfo = self.infoDict.get(legend, None)
                if curveInfo is not None:
                    motorValue = curveInfo.get(motorName, '---')
                else:
                    motorValue = '---'
                self.setItem(i, column, str(motorValue))
        else:
            for i in range(0, self.rowCount()):
                self.setItem(i, column, '')
        self.resizeColumnToContents(column)

    def getAllMotorNames(self):
        nameSet = []
        for dic in self.motorsList:
            for key in dic.keys():
                if key not in nameSet:
                    nameSet.append(key)
        return nameSet

    def setItem(self, row, column, text=''):
        item = self.item(row, column)
        if item is None:
            item = qt.QTableWidgetItem(text)
            item.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
            qt.QTableWidget.setItem(self, row, column, item)
        else:
            item.setText(text)

    def scrollContentsBy(self, dx, dy):
        qt.QTableWidget.scrollContentsBy(self, dx, dy )
        if (dx != 0):
            self.horizontalHeader().fixComboPositions()

class MotorInfoDialog(qt.QWidget):
    def __init__(self, parent, legends, motorValues):
        """
        legends         List contains Plotnames
        motorValues     List contains names and values of the motors
        """
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("Motor Info Plugin")
        if len(legends) != len(motorValues):
            print('Consistency error: legends and motorValues do not have same length!')
        self.numCurves = len(legends)
        # Buttons
        self.buttonAddColumn = qt.QPushButton("Add", self)
        self.buttonDeleteColumn = qt.QPushButton("Del", self)
        self.buttonUpdate = qt.QPushButton(qt.QIcon(qt.QPixmap(IconDict["reload"])), '', self)
        # Table
        self.table = MotorInfoTable(self, self.numCurves, 4, legends, motorValues)
        # Layout
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(1, 1, 1, 1)
        self.mainLayout.setSpacing(2)
        self.buttonLayout = qt.QHBoxLayout(None)
        self.buttonLayout .setSpacing(1)
        # Add widgets to layour
        self.mainLayout.addWidget(self.table, 0, 0)
        self.mainLayout.addLayout(self.buttonLayout, 1, 0)
        self.buttonLayout.addWidget(self.buttonUpdate)
        self.buttonLayout.addWidget(self.buttonAddColumn)
        self.buttonLayout.addWidget(self.buttonDeleteColumn)
        self.buttonLayout.addWidget(qt.HorizontalSpacer(self))
        self.resize(700, 400)
        # Make connections
        self.buttonAddColumn.clicked.connect(self.table.addColumn)
        self.buttonDeleteColumn.clicked.connect(self.table.delColumn)

    def keyPressEvent(self, event):
        if (event.key() == qt.Qt.Key_Escape):
            self.close()

def main():
    import sys, random
    legends = ['Curve0', 'Curve1', 'Curve2', 'Curve3']
    motors = [{'Motor12': 0.5283546103038855, 'Motor11': 0.8692713996985609, 'Motor10': 0.2198364185388587,
               'Motor 8': 0.19806882661182112, 'Motor 9': 0.4844754557916431, 'Motor 4': 0.3502522172639875},
               {'Motor18': 0.4707468826876532, 'Motor17': 0.6958160702991127, 'Motor16': 0.8257808117546283,
               'Motor13': 0.09084289261899736, 'Motor12': 0.5190253643331453, 'Motor11': 0.21344565983311958},
               {'Motor12': 0.6504890336783156, 'Motor11': 0.44400576643956124, 'Motor10': 0.613870067851634, 
               'Motor 8': 0.901968648110583, 'Motor 9': 0.3197687710845185, 'Motor 4': 0.5714322786278168}, 
               {'Motor13': 0.6491598094029021, 'Motor12': 0.2975843286841311, 'Motor11': 0.006312468992195397,
               'Motor 9': 0.014325738753558803, 'Motor 4': 0.8185362197656616, 'Motor 5': 0.6643614796103005}]
    app = qt.QApplication(sys.argv)
    w = MotorInfoDialog(None, legends, motors)
    w.show()
    app.exec_()
    
if __name__ == '__main__':
    main()
