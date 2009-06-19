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
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui

import copy
import posixpath

class HDFInfoCustomEvent(QtCore.QEvent):
    def __init__(self, ddict):
        if ddict is None:
            ddict = {}
        self.dict = ddict
        QtCore.QEvent.__init__(self, QtCore.QEvent.User)

class VerticalSpacer(QtGui.QWidget):
    def __init__(self, *args):
        QtGui.QWidget.__init__(self, *args)
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,
                                          QtGui.QSizePolicy.Expanding))

class SimpleInfoGroupBox(QtGui.QGroupBox):
    def __init__(self, parent, title=None, keys=None):
        QtGui.QGroupBox.__init__(self, parent)
        self.mainLayout = QtGui.QGridLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        if title is not None:
            self.setTitle(title)
        if keys is None:
           keys = [] 
        self.keyList = keys
        self.keyDict = {}
        if len(self.keyList):
            self._build()

    def _build(self):
        i = 0
        for key in self.keyList:
            label = QtGui.QLabel(self)
            label.setText(key)
            line = QtGui.QLineEdit(self)
            line.setReadOnly(True)
            self.mainLayout.addWidget(label, i, 0)
            self.mainLayout.addWidget(line, i, 1)
            self.keyDict[key] = (label, line)
            i += 1

    def setInfoDict(self, ddict):
        if not len(self.keyList):
            self.keyList = ddict.keys()
            self._build()            
        self._fillInfo(ddict)

    def _fillInfo(self, ddict0):
        ddict = self._getMappedDict(ddict0)
        actualKeys = ddict.keys()
        dictKeys = []
        for key in actualKeys:
            l = key.lower()
            if l not in dictKeys:
                dictKeys.append(l)
        for key in self.keyList:
            l = key.lower()
            if l in dictKeys:
                self._fillKey(key,
                              ddict[actualKeys[dictKeys.index(l)]])

    def _getMappedDict(self, ddict):
        #Default implementation returns a copy of the input dictionnary
        return copy.deepcopy(ddict)

    def _fillKey(self, key, value):
        #This can be overwritten
        if type(value) == type(""):
            self.keyDict[key][1].setText(value)
        else:
            self.keyDict[key][1].setText(str(value))

class NameGroupBox(SimpleInfoGroupBox):
    def __init__(self, parent, title=None, keys=["Name", "Path", "Type"]):
        SimpleInfoGroupBox.__init__(self, parent, title=title, keys=keys)

class DimensionGroupBox(SimpleInfoGroupBox):
    def __init__(self, parent, title=None, keys=None):
        keys = ["No. of Dimension(s)",
                "Dimension Size(s)",
                "Data Type"]
        SimpleInfoGroupBox.__init__(self, parent, title=title, keys=keys)

    def _getMappedDict(self, ddict):
        return copy.deepcopy(ddict)

class MembersGroupBox(QtGui.QGroupBox):
    def __init__(self, parent):
        QtGui.QGroupBox.__init__(self, parent)
        self.mainLayout = QtGui.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        self.setTitle("Group Members")
        self.label = QtGui.QLabel(self)
        self.label.setText("Number of members: 0")
        self.table = QtGui.QTableWidget(self)
        self.table.setColumnCount(2)
        labels = ["Name", "Type"]
        for i in [0, 1]:
            item = self.table.horizontalHeaderItem(i)
            if item is None:
                item = QtGui.QTableWidgetItem(labels[i],
                                           QtGui.QTableWidgetItem.Type)
            item.setText(labels[i])
            self.table.setHorizontalHeaderItem(i, item)        
        self.mainLayout.addWidget(self.label)
        self.mainLayout.addWidget(self.table)

    def setInfoDict(self, ddict):
        keylist = ddict.keys()
        if "members" not in keylist:
            self.hide()
            return
        keylist = ddict['members']
        self.label.setText("Number of members: %d" % len(keylist))
        nrows = len(keylist)
        if not nrows:
            self.table.setRowCount(nrows)
            self.hide()
            return
        self.table.setRowCount(nrows)        
        keylist.sort()
        row = 0
        for key in keylist:
            item = self.table.item(row, 0)
            if item is None:
                item = QtGui.QTableWidgetItem(key, QtGui.QTableWidgetItem.Type)
                item.setFlags(QtCore.Qt.ItemIsSelectable|
                                  QtCore.Qt.ItemIsEnabled)
                self.table.setItem(row, 0, item)
            else:
                item.setText(key)
            info = ddict[key]['Type']
            item = self.table.item(row, 1)
            if item is None:
                item = QtGui.QTableWidgetItem(info, QtGui.QTableWidgetItem.Type)
                item.setFlags(QtCore.Qt.ItemIsSelectable|
                                  QtCore.Qt.ItemIsEnabled)
                self.table.setItem(row, 1, item)
            else:
                item.setText(info)
            row += 1
        #self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(1)
        #self.show()
        
class HDF5GeneralInfoWidget(QtGui.QWidget):
    def __init__(self, parent=None, ddict=None):
        QtGui.QWidget.__init__(self, parent)
        self.mainLayout = QtGui.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        self.nameWidget = NameGroupBox(self)
        self.membersWidget = MembersGroupBox(self)
        self.dimensionWidget = DimensionGroupBox(self)
        self.mainLayout.addWidget(self.nameWidget)
        self.mainLayout.addWidget(self.membersWidget)
        self.mainLayout.addWidget(VerticalSpacer(self))
        self.mainLayout.addWidget(self.dimensionWidget)
        self._notifyCloseEventToWidget = None
        
        if ddict is not None:
            self.setInfoDict(ddict)

    def setInfoDict(self, ddict):
        if ddict.has_key('general'):
            self._setInfoDict(ddict['general'])
        else:
            self._setInfoDict(ddict)

    def _setInfoDict(self, ddict):
        self.nameWidget.setInfoDict(ddict)
        self.membersWidget.setInfoDict(ddict)
        self.dimensionWidget.setInfoDict(ddict)
        if ddict.has_key('members'):
            if len(ddict['members']):
                #it is a datagroup
                self.dimensionWidget.hide()  
        self.dimensionWidget.hide()


class HDF5AttributesInfoWidget(QtGui.QWidget):
    def __init__(self, parent):
        QtGui.QGroupBox.__init__(self, parent)
        self.mainLayout = QtGui.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        self.label = QtGui.QLabel(self)
        self.label.setText("Number of members: 0")
        self.table = QtGui.QTableWidget(self)
        labels = ["Name", "Value", "Type", "Size"]
        self.table.setColumnCount(len(labels))
        for i in range(len(labels)):
            item = self.table.horizontalHeaderItem(i)
            if item is None:
                item = QtGui.QTableWidgetItem(labels[i],
                                           QtGui.QTableWidgetItem.Type)
            item.setText(labels[i])
            self.table.setHorizontalHeaderItem(i, item)
        self._tableLabels = labels
        self.mainLayout.addWidget(self.label)
        self.mainLayout.addWidget(self.table)

    def setInfoDict(self, ddict):
        if ddict.has_key('attributes'):
            self._setInfoDict(ddict['attributes'])
        else:
            self._setInfoDict(ddict)


    def _setInfoDict(self, ddict):
        keylist = ddict['names']
        self.label.setText("Number of attributes: %d" % len(keylist))
        nrows = len(keylist)
        if not nrows:
            self.table.setRowCount(nrows)
            self.hide()
            return
        self.table.setRowCount(nrows)        
        keylist.sort()
        row = 0
        for key in keylist:
            for label in self._tableLabels:
                if not ddict[key].has_key(label):
                    continue
                else:
                    text = ddict[key][label]
                col = self._tableLabels.index(label)
                item = self.table.item(row, col)
                if item is None:
                    item = QtGui.QTableWidgetItem(text, QtGui.QTableWidgetItem.Type)
                    item.setFlags(QtCore.Qt.ItemIsSelectable|
                                      QtCore.Qt.ItemIsEnabled)
                    self.table.setItem(row, col, item)
                else:
                    item.setText(text)
            row += 1

        #self.table.resizeColumnToContents(0)
        #self.table.resizeColumnToContents(1)
        #self.table.resizeColumnToContents(2)
        #self.table.resizeColumnToContents(3)
        #self.show()

class HDF5InfoWidget(QtGui.QTabWidget):
    def __init__(self, parent=None, info=None):
        QtGui.QTabWidget.__init__(self, parent)
        self._notifyCloseEventToWidget = None
        self._build()
        if info is not None:
            self.setInfoDict(info)

    def sizeHint(self):
        return QtCore.QSize(2 * QtGui.QTabWidget.sizeHint(self).width(),
                        QtGui.QTabWidget.sizeHint(self).height())
                        
    def _build(self):
        self.generalInfoWidget = HDF5GeneralInfoWidget(self)
        self.attributesInfoWidget = HDF5AttributesInfoWidget(self)
        self.addTab(self.generalInfoWidget, 'General')
        self.addTab(self.attributesInfoWidget, 'Attributes')

    def setInfoDict(self, ddict):
        self.generalInfoWidget.setInfoDict(ddict)
        self.attributesInfoWidget.setInfoDict(ddict)

    def notifyCloseEventToWidget(self, widget):
        self._notifyCloseEventToWidget = widget

    def closeEvent(self, event):
        if self._notifyCloseEventToWidget is not None:
            ddict={}
            ddict['event'] = 'closeEventSignal'
            ddict['id']    = id(self)
            newEvent = HDFInfoCustomEvent(ddict)
            QtGui.QApplication.postEvent(self._notifyCloseEventToWidget,
                                      newEvent)
            self._notifyCloseEventToWidget = None
        return QtGui.QWidget.closeEvent(self, event)

def getInfo(hdf5File, node):
    data = hdf5File[node]
    ddict = {}
    ddict['general'] = {}
    ddict['attributes'] = {}
    ddict['general']['Name'] = posixpath.basename(data.name)
    ddict['general']['Path'] = data.name
    ddict['general']['Type'] = str(data)
    if hasattr(data, "keys"):
        ddict['general']['members'] = data.keys()
    elif hasattr(data, "listnames"):
        ddict['general']['members'] = data.listnames()
    else:
        ddict['general']['members'] = []
    for member in ddict['general']['members']:
        ddict['general'][member] = {}
        ddict['general'][member]['Name'] = str(member)
        ddict['general'][member]['Type'] = str(hdf5File[node+"/"+member])
    if hasattr(data.attrs, "keys"):
        ddict['attributes']['names'] = data.attrs.keys()
    elif hasattr(data.attrs, "listnames"):
        ddict['attributes']['names'] = data.attrs.listnames()
    else:
        ddict['attributes']['names'] = []
    ddict['attributes']['names'].sort()
    for key in ddict['attributes']['names']:
        ddict['attributes'][key] = {}
        Name = key
        Value = data.attrs[key]
        Type =  str(type(Value))
        if type(Value) == type(""):
            Size = "%d" % len(Value)
        elif type(Value) in [type(1), type(0.0)]:
            Value = str(Value)
            Size = "1"
        else:
            Value = str(Value)
            Size = "Unknown"
        ddict['attributes'][key]['Name']  = Name
        ddict['attributes'][key]['Value'] = Value
        ddict['attributes'][key]['Type']  = Type
        ddict['attributes'][key]['Size']  = Size
    return ddict
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print "Usage:"
        print "python HDF5Info.py hdf5File node"
        sys.exit(0)
    if 1:
        import h5py
        h=h5py.File(sys.argv[1])
    else:
        from PyMca import phynx
        h = phynx.File(sys.argv[1])
    node = sys.argv[2]
    info = getInfo(h, node)
    app = QtGui.QApplication([])
    w = HDF5InfoWidget()
    w.setInfoDict(info)
    w.show()
    sys.exit(app.exec_())
