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
try:
    from PyMca.PyMcaQt import qt
except ImportError:
    import PyMcaQt as qt

if qt.qVersion() < '4.0.0':
    raise ImportError, "This module requires PyQt4"

import copy
import posixpath

class VerticalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed,
                                          qt.QSizePolicy.Expanding))

class SimpleInfoGroupBox(qt.QGroupBox):
    def __init__(self, parent, title=None, keys=None):
        qt.QGroupBox.__init__(self, parent)
        self.mainLayout = qt.QGridLayout(self)
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
            label = qt.QLabel(self)
            label.setText(key)
            line = qt.QLineEdit(self)
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

class MembersGroupBox(qt.QGroupBox):
    def __init__(self, parent):
        qt.QGroupBox.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        self.setTitle("Group Members")
        self.label = qt.QLabel(self)
        self.label.setText("Number of members: 0")
        self.table = qt.QTableWidget(self)
        self.table.setColumnCount(2)
        labels = ["Name", "Type"]
        for i in [0, 1]:
            item = self.table.horizontalHeaderItem(i)
            if item is None:
                item = qt.QTableWidgetItem(labels[i],
                                           qt.QTableWidgetItem.Type)
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
                item = qt.QTableWidgetItem(key, qt.QTableWidgetItem.Type)
                item.setFlags(qt.Qt.ItemIsSelectable|
                                  qt.Qt.ItemIsEnabled)
                self.table.setItem(row, 0, item)
            else:
                item.setText(key)
            info = ddict[key]['Type']
            item = self.table.item(row, 1)
            if item is None:
                item = qt.QTableWidgetItem(info, qt.QTableWidgetItem.Type)
                item.setFlags(qt.Qt.ItemIsSelectable|
                                  qt.Qt.ItemIsEnabled)
                self.table.setItem(row, 1, item)
            else:
                item.setText(info)
            row += 1
        #self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(1)
        self.show()
        
class HDF5GeneralInfoWidget(qt.QWidget):
    def __init__(self, parent=None, ddict=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        self.nameWidget = NameGroupBox(self)
        self.membersWidget = MembersGroupBox(self)
        self.dimensionWidget = DimensionGroupBox(self)
        self.mainLayout.addWidget(self.nameWidget)
        self.mainLayout.addWidget(self.membersWidget)
        self.mainLayout.addWidget(VerticalSpacer(self))
        self.mainLayout.addWidget(self.dimensionWidget)
        
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

"""
class HDF5InfoWidget(qt.QTabWidget):
    def __init__(self, parent=None, info=None):
        qt.QTabWidget.__init__(self, parent)
        self.info = {"general":{},
                    "attributes":{}}
        self._build()
        if info is not None:
            self.setInfoDict(info)

    def _build(self):
        pass

    def setInfoDict(self, info):
        print info

"""

def getInfo(hdf5File, node):
    data = hdf5File[node]
    ddict = {}
    ddict['general'] = {}
    ddict['attributes'] = {}
    ddict['general']['Name'] = posixpath.basename(data.name)
    ddict['general']['Path'] = data.name
    ddict['general']['Type'] = str(data)
    if hasattr(data, "listnames"):
        ddict['general']['members'] = data.listnames()
    else:
        ddict['general']['members'] = []
    for member in ddict['general']['members']:
        ddict['general'][member] = {}
        ddict['general'][member]['Name'] = str(member)
        ddict['general'][member]['Type'] = str(hdf5File[node+"/"+member])
    for att in data.attrs:
        ddict['attributes'][att] = data.attrs[att]
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
    app = qt.QApplication([])
    w = HDF5GeneralInfoWidget()
    w.setInfoDict(info)
    w.show()
    sys.exit(app.exec_())
