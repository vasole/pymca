#/*##########################################################################
# Copyright (C) 2004-2011 European Synchrotron Radiation Facility
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
try:
    from PyMca.QNexusWidget import *
    from PyMca import QStackWidget
    from PyMca import HDF5Stack1D
except:
    from QNexusWidget import *
    import QStackWidget
    import HDF5Stack1D


DEBUG=0

class PyMcaNexusWidget(QNexusWidget):
    def __init__(self, *var, **kw):
        QNexusWidget.__init__(self, *var, **kw)
        
    def itemRightClickedSlot(self, ddict):
        filename = ddict['file']
        name = ddict['name']
        if ddict['dtype'].startswith('|S'):
            #handle a right click on a dataset of string type
            return self.showInfoWidget(filename, name, False)
            pass
        elif ddict['dtype'] == '':
            #handle a right click on a group
            return self.showInfoWidget(filename, name, False)
        else:
            #handle a right click on a numeric dataset
            _hdf5WidgetDatasetMenu = QtGui.QMenu(self)
            _hdf5WidgetDatasetMenu.addAction(QtCore.QString("Add to selection table"),
                                        self._addToSelectionTable)

            _hdf5WidgetDatasetMenu.addAction(QtCore.QString("Show Information"),
                                    self._showInfoWidgetSlot)
            fileIndex = self.data.sourceName.index(filename)
            phynxFile  = self.data._sourceObjectList[fileIndex]
            info = self.getInfo(phynxFile, name)
            interpretation = info.get('interpretation', "")
            stack1D = False
            stack2D = False
            nDim = len(ddict['shape'].split('x'))
            if nDim > 1:
                stack1D = True
            if nDim == 3:
                stack2D = True
            if interpretation.lower() in ['image']:
                stack1D = False
            if interpretation.lower() in ['spectrum']:
                stack2D = False
                
            if stack1D:
                _hdf5WidgetDatasetMenu.addAction(QtCore.QString("Show as 1D Stack"),
                                    self._stack1DSignal)
            if stack2D:
                _hdf5WidgetDatasetMenu.addAction(QtCore.QString("Show as 2D Stack"),
                                    self._stack2DSignal)
            self._lastDatasetDict= ddict
            _hdf5WidgetDatasetMenu.exec_(QtGui.QCursor.pos())
            self._lastDatasetDict= None
            return

    def _stack1DSignal(self):
        if DEBUG:
            print("_stack1DSignal")
        self._stackSignal(index=-1)

    def _stack2DSignal(self):
        if DEBUG:
            print("_stack2DSignal")
        self._stackSignal(index=0)

    def _stackSignal(self, index=-1):
        ddict = self._lastDatasetDict
        filename = ddict['file']
        name = ddict['name']
        sel = {}
        sel['SourceName'] = self.data.sourceName * 1
        sel['SourceType'] = "HDF5"
        fileIndex = self.data.sourceName.index(filename)
        phynxFile  = self.data._sourceObjectList[fileIndex]
        title     = filename + " " + name
        sel['selection'] = {}
        sel['selection']['sourcename'] = filename
        #single dataset selection
        scanlist = None
        sel['selection']['x'] = []
        sel['selection']['y'] = [name]
        sel['selection']['m'] = []
        sel['selection']['index'] = index
        self._checkWidgetDict()
        if 1:
            #this does not crash because
            #the same phynx instance is shared
            stack = HDF5Stack1D.HDF5Stack1D(phynxFile, sel['selection'],
                                scanlist=scanlist,
                                dtype=None)
        else:
            #this crashes
            stack = HDF5Stack1D.HDF5Stack1D([filename], sel['selection'],
                                scanlist=scanlist,
                                dtype=None)
        widget = QStackWidget.QStackWidget()
        widget.setWindowTitle(title)
        widget.notifyCloseEventToWidget(self)
        widget.setStack(stack)
        wid = id(widget)
        self._lastWidgetId = wid
        self._widgetDict[wid] = widget
        widget.show()

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    try:
        import Object3D
    except:
        pass
    w = PyMcaNexusWidget()
    if 0:
        w.setFile(sys.argv[1])
    else:
        import NexusDataSource
        dataSource = NexusDataSource.NexusDataSource(sys.argv[1:])
        w.setDataSource(dataSource)
    def addSelection(sel):
        print(sel)
    def removeSelection(sel):
        print(sel)
    def replaceSelection(sel):
        print(sel)
    w.show()
    QtCore.QObject.connect(w, QtCore.SIGNAL("addSelection"),     addSelection)
    QtCore.QObject.connect(w, QtCore.SIGNAL("removeSelection"),  removeSelection)
    QtCore.QObject.connect(w, QtCore.SIGNAL("replaceSelection"), replaceSelection)
    sys.exit(app.exec_())
