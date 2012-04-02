#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
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
import sys
if 'PyQt4.QtGui' in sys.modules:
    raise ImportError("PyMcaNexusWidget only works under PyQt4")
try:
    from PyMca.QNexusWidget import *
    from PyMca import QStackWidget
    from PyMca import HDF5Stack1D
except:
    print("PyMcaNexusWidget importing from directory")
    from QNexusWidget import *
    import QStackWidget
    import HDF5Stack1D
import h5py

DEBUG=0

class PyMcaNexusWidget(QNexusWidget):
    def __init__(self, *var, **kw):
        QNexusWidget.__init__(self, *var, **kw)
        
    def itemRightClickedSlot(self, ddict):
        filename = ddict['file']
        name = ddict['name']
        if ddict['dtype'].startswith('|S') or\
           ddict['dtype'].startswith('|O'):
            #handle a right click on a dataset of string type
            return self.showInfoWidget(filename, name, False)
            pass
        elif ddict['dtype'] == '':
            #handle a right click on a group
            return self.showInfoWidget(filename, name, False)
        else:
            #handle a right click on a numeric dataset
            _hdf5WidgetDatasetMenu = QtGui.QMenu(self)
            _hdf5WidgetDatasetMenu.addAction(QString("Add to selection table"),
                                        self._addToSelectionTable)

            _hdf5WidgetDatasetMenu.addAction(QString("Show Information"),
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
                _hdf5WidgetDatasetMenu.addAction(QString("Show as 1D Stack"),
                                    self._stack1DSignal)
            if stack2D:
                _hdf5WidgetDatasetMenu.addAction(QString("Show as 2D Stack"),
                                    self._stack2DSignal)
                _hdf5WidgetDatasetMenu.addAction(QString("Load and show as 2D Stack"),
                                    self._loadStack2DSignal)
            self._lastDatasetDict= ddict
            _hdf5WidgetDatasetMenu.exec_(QtGui.QCursor.pos())
            self._lastDatasetDict= None
            return

    def _stack1DSignal(self):
        if DEBUG:
            print("_stack1DSignal")
        self._stackSignal(index=-1)

    def _loadStack2DSignal(self):
        if DEBUG:
            print("_loadStack2DSignal")
        self._stackSignal(index=0, load=True)

    def _stack2DSignal(self, load=False):
        if DEBUG:
            print("_stack2DSignal")
        self._stackSignal(index=0, load=False)

    def _stackSignal(self, index=-1, load=False):
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

        widget = QStackWidget.QStackWidget()
        widget.setWindowTitle(title)
        widget.notifyCloseEventToWidget(self)

        #different ways to fill the stack
        if h5py.version.version < '2.0':
            useInstance = True
        else:
            useInstance = False
        if useInstance:
            #this crashes with h5py 1.x
            #this way it is not loaded into memory unless requested
            #and cannot crash because same instance is used
            stack = phynxFile[name]
        else:
            #create a new instance
            stack = h5py.File(filename, 'r')[name]

        #the only problem is that, if the shape is not of type (a, b, c),
        #it will not be possible to reshape it. In that case I have to
        #actually read the values
        nDim = len(stack.shape)
        if (load) or (nDim != 3):
            stack = stack.value
            shape = stack.shape
            if index == 0:
                #Stack of images
                n = 1
                for dim in shape[:-2]:
                    n = n * dim
                stack.shape = n, shape[-2], shape[-1]
            else:
                #stack of mca
                n = 1
                for dim in shape[:-1]:
                    n = n * dim
                stack.shape = 1, n, shape[-1]
                #index equal -1 should be able to handle it
                #if not, one would have to uncomment next line
                #index = 2
        widget.setStack(stack, mcaindex=index)
        wid = id(widget)
        self._lastWidgetId = wid
        self._widgetDict[wid] = widget
        widget.show()

if __name__ == "__main__":
    try:
        #this is to add the 3D buttons ...
        from PyMca import Object3D
    except:
        pass
    app = QtGui.QApplication(sys.argv)
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
