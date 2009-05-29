import os
import HDF5Widget
qt = HDF5Widget.qt
try:
    from PyMca import SpecFileCntTable
except:
    import SpecFileCntTable

class Buttons(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        self.buttonGroup = qt.QButtonGroup(self)
        self.buttonList = []
        i = 0
        optionList = ['SCAN', 'MCA', 'IMAGE']
        optionList = ['SCAN', 'MCA']
        actionList = ['ADD', 'REMOVE', 'REPLACE']
        for option in optionList:
            row = optionList.index(option)
            for action in actionList:
                col = actionList.index(action)
                button = qt.QPushButton(self)
                button.setText(action + " " + option)
                self.mainLayout.addWidget(button, row, col)
                self.buttonGroup.addButton(button)
                self.buttonList.append(button)
        self.connect(self.buttonGroup,
                     qt.SIGNAL('buttonClicked(QAbstractButton *)'),
                     self.emitSignal)

    def emitSignal(self, button):
        ddict={}
        ddict['event'] = 'buttonClicked'
        ddict['action'] = str(button.text())
        self.emit(qt.SIGNAL('ButtonsSignal'), ddict)

class QNexusWidget(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.data = None
        self._dataSourceList = []
        self._oldCntSelection = None
        self._cntList = []
        self._widgetList = []
        self.build()

    def build(self):
        self.mainLayout = qt.QVBoxLayout(self)
        self.splitter = qt.QSplitter(self)
        self.splitter.setOrientation(qt.Qt.Vertical)
        self._model = HDF5Widget.FileModel()
        self.hdf5Widget = HDF5Widget.HDF5Widget(self._model, self.splitter)
        self.hdf5Widget.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        self.cntTable = SpecFileCntTable.SpecFileCntTable(self.splitter)
        self.mainLayout.addWidget(self.splitter)
        self.buttons = Buttons(self)
        self.mainLayout.addWidget(self.buttons)
        self.connect(self.hdf5Widget,
                     qt.SIGNAL('HDF5WidgetSignal'),
                     self.hdf5Slot)
        self.connect(self.buttons,
                     qt.SIGNAL('ButtonsSignal'),
                     self.buttonsSlot)

    def setDataSource(self, dataSource):
        self.data = dataSource
        if self.data is None:
            self.hdf5Widget.collapseAll()
            return
        for source in self.data._sourceObjectList:
            self.hdf5Widget.model().appendPhynxFile(source, weakreference=True)  

    def setFile(self, filename):
        self._data = self.hdf5Widget.model().openFile(filename, weakreference=True)

    def hdf5Slot(self, ddict):
        if ddict['event'] == 'itemClicked':
            if ddict['mouse'] == "right":
                if ddict['type'] == 'Dataset':
                    if ddict['dtype'].startswith('|S'):
                        #print "string"
                        pass
                    else:
                        #print "dataset"
                        pass
                
        if ddict['event'] == "itemDoubleClicked":
            if ddict['type'] == 'Dataset':
                if ddict['dtype'].startswith('|S'):
                    print "string"
                else:
                    root = ddict['path'].split('/')
                    root = "/" + root[1]
                    cnt  = ddict['path'].split(root)[-1]
                    if cnt not in self._cntList:
                        self._cntList.append(cnt)
                        self.cntTable.build(self._cntList)
            if ddict['type'] == 'Entry':
                print "I should apply latest selection"

    def buttonsSlot(self, ddict):
        if self.data is None:
            return
        action, selectionType = ddict['action'].split()
        entryList = self.getEntryList()
        if not len(entryList):
            return
        cntSelection = self.cntTable.getCounterSelection()
        selectionList = []
        for entry, filename in entryList:
            if not len(cntSelection['cntlist']):
                continue
            if not len(cntSelection['y']):
                #nothing to plot
                continue
            for yCnt in cntSelection['y']:
                sel = {}
                sel['SourceName'] = [filename]
                sel['SourceType'] = "HDF5"
                fileIndex = self.data.sourceName.index(filename)
                phynxFile  = self.data._sourceObjectList[fileIndex]
                entryIndex = phynxFile["/"].listnames().index(entry[1:])
                sel['Key']        = "%d.%d" % (fileIndex+1, entryIndex+1)
                sel['legend']     = os.path.basename(sel['SourceName'][0])+\
                                    " " + sel['Key']
                sel['selection'] = {}
                sel['selection']['sourcename'] = filename
                sel['selection']['entry'] = entry
                sel['selection']['key'] = "%d.%d" % (fileIndex+1, entryIndex+1)
                sel['selection']['x'] = cntSelection['x']
                sel['selection']['y'] = [yCnt]
                sel['selection']['m'] = cntSelection['m']
                sel['selection']['cntlist'] = cntSelection['cntlist']
                sel['selection']['selectiontype'] = selectionType
                if selectionType.upper() == "SCAN":
                    sel['scanselection'] = True
                    sel['mcaselection']  = False
                elif selectionType.upper() == "MCA":
                    sel['scanselection'] = False
                    sel['mcaselection']  = True
                else:
                    sel['scanselection'] = False
                    sel['mcaselection']  = False
                selectionList.append(sel)
        if len(selectionList):
            if action.upper() == "ADD":
                self.emit(qt.SIGNAL("addSelection"), selectionList)
            if action.upper() == "REMOVE":
                self.emit(qt.SIGNAL("removeSelection"), selectionList)
            if action.upper() == "REPLACE":
                self.emit(qt.SIGNAL("replaceSelection"), selectionList)

    def getEntryList(self):
        return self.hdf5Widget.getSelectedEntries()
                

if __name__ == "__main__":
    import sys
    app = qt.QApplication(sys.argv)
    w = QNexusWidget()
    if 0:
        w.setFile(sys.argv[1])
    else:
        import NexusDataSource
        dataSource = NexusDataSource.NexusDataSource(sys.argv[1])
        w.setDataSource(dataSource)
    def addSelection(sel):
        print sel
    def removeSelection(sel):
        print sel
    def replaceSelection(sel):
        print sel
    w.show()
    qt.QObject.connect(w, qt.SIGNAL("addSelection"),     addSelection)
    qt.QObject.connect(w, qt.SIGNAL("removeSelection"),  removeSelection)
    qt.QObject.connect(w, qt.SIGNAL("replaceSelection"), replaceSelection)
    sys.exit(app.exec_())

    
    fileModel = FileModel()
    fileView = HDF5Widget(fileModel)
    #fileModel.openFile('/home/darren/temp/PSI.hdf')
    phynxFile = fileModel.openFile(sys.argv[1])
    def mySlot(ddict):
        print ddict
        if ddict['type'].lower() in ['dataset']:
            print phynxFile[ddict['path']].dtype, phynxFile[ddict['path']].shape 
    qt.QObject.connect(fileView, qt.SIGNAL("HDF5WidgetSignal"), mySlot)
    fileView.show()
    sys.exit(app.exec_())
