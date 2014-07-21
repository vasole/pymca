#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys
import weakref
from . import Object3DQt as qt
from .Object3DIcons import IconDict
from . import ObjectTree

DEBUG = 0

class ObjectTreeWidget(qt.QTreeWidget):
    def __init__(self, parent=None, tree=None, labels=None):
        qt.QTreeWidget.__init__(self, parent)
        if labels is None:
            labels = ['Name', 'Type'] #, 'Vertices']
        if tree is None:
            self.tree = ObjectTree.ObjectTree('__Scene__', 'Scene')
        else:
            self.tree = tree
        ncols = len(labels)
        self.setColumnCount(ncols)
        self.setHeaderLabels(labels)

    def focusInEvent(self, event):
        event.accept()

    def addObject(self, item, name=None, parent=None, update=True):
        if name is None:
            name = item.name()
        if parent is None:
            self.tree.addChild(item, name)
        else:
            self[parent].addChild(item, name)
        if update:
            self.updateView()

    def removeObject(self, name):
        #An object has to correspond to an entry in the tree
        treeObject = self.tree.find(name)
        if treeObject is None:
            # do nothing?
            return
        treeObject.erase()
        self.updateView()

    def updateView(self):
        self.clear()
        self.showInView(self.tree)

    def showInView(self, tree, parent=None):
        """
        Represent a tree in the QTreeWidget
        """
        if parent is None:
            widgetItem = Object3DTreeWidgetItem(0, tree)
            self.addTopLevelItem(widgetItem)
        else:
            #find the parent item
            itemList = self.findItems(parent.name(),
              qt.Qt.MatchExactly|qt.Qt.MatchCaseSensitive|qt.Qt.MatchRecursive,
                        0)
            if len(itemList):
                widgetItemParent = itemList[0]
                widgetItem = Object3DTreeWidgetItem(1, tree)
                widgetItemParent.addChild(widgetItem)
            else:
                return
        ob = tree.root[0]
        if hasattr(ob,'selected'):
            if ob.selected():
                widgetItem.setSelected(True)
                self.scrollToItem(widgetItem,
                    qt.QAbstractItemView.EnsureVisible)

        for subTree in tree.childList():
            self.showInView(subTree, tree.parent(subTree.name()))

    def setSelected(self, name):
        itemList = self.findItems(name,
              qt.Qt.MatchExactly|qt.Qt.MatchCaseSensitive|qt.Qt.MatchRecursive,
                        0)
        if len(itemList):
            itemList[0].setSelected(True)

class Object3DTreeWidgetItem(qt.QTreeWidgetItem):
    def __init__(self, wtype, object3D):
        if type(wtype) != type(1):
            raise TypeError("First argument must be an integer")
        #if (wtype != 0) and (wtype < qt.QTreeWidgetItem.UserType):
        #    raise TypeError("First argument must be 0 or an integer >= 1000")
        actualType = wtype
        qt.QTreeWidgetItem.__init__(self, wtype + qt.QTreeWidgetItem.UserType)
        self.__object3D = object3D
        self.setText(0, object3D.name())
        if wtype == 0:
            text = "Scene"
        else:
            text = "3D Object"
        self.setText(1, text)

class Object3DObjectTree(qt.QGroupBox):
    sigObjectTreeSignal = qt.pyqtSignal(object)

    def __init__(self, parent = None, tree=None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle('Objects Tree')
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.treeWidget = ObjectTreeWidget(self, tree=tree)
        self.tree = weakref.proxy(self.treeWidget.tree)
        self.actions = ObjectActions(self)
        self.mainLayout.addWidget(self.treeWidget)
        self.mainLayout.addWidget(self.actions)
        self.addObject = self.treeWidget.addObject
        self.__current = 'Scene'
        self.__previous= None
        self.__cutObject = None
        self.__replacing = False
        self.actions.cutButton.clicked.connect(self.cutObject)
        self.actions.pasteButton.clicked.connect(self.pasteObject)
        self.actions.deleteButton.clicked.connect(self.deleteObject)
        self.actions.replaceButton.clicked.connect(self.replaceWithObject)

        self.treeWidget.currentItemChanged.connect(self.itemChanged)

    def updateView(self, expand=False):
        self.treeWidget.updateView()
        if expand:
            if qt.qVersion() >= '4.2.0':
                self.treeWidget.expandAll()
        objectList = self.getSelectedObjectList()
        if len(objectList):
            self.__current = objectList[0]

    def getSelectedObjectList(self):
        selected = []
        for item in self.tree.childList():
            ob = item.root[0]
            if hasattr(ob, 'selected'):
                if ob.selected():
                    selected.append(item.name())
        return selected

    def setSelectedObject(self, name=None):
        if name is None:
            name = self.__current
        else:
            self.__current = name

        #reset all the children
        for item in self.tree.childList():
            ob = item.root[0]
            if hasattr(ob, 'selected'):
                ob.setSelected(False)

        #but do not forget the scene itself
        if hasattr(self.tree.root[0], "setSelected"):
            self.tree.root[0].setSelected(False)

        #now select the proper one
        if self.tree.name() == name:
            self.tree.root[0].setSelected(True)
        else:
            child = self.tree.find(name)
            if child is not None:
                ob = child.root[0]
                if hasattr(ob, 'selected'):
                    ob.setSelected(True)
                    self.treeWidget.setSelected(child.name())

    def cutObject(self):
        if self.__current == 'Scene':
            self.__cutObject = None
            qt.QMessageBox.critical(self, "Error on cut",
                "You cannot cut the Scene itself.",
                qt.QMessageBox.Ok | qt.QMessageBox.Default,
                            qt.QMessageBox.NoButton)
        elif self.__current is None:
            qt.QMessageBox.critical(self, "Error on cut",
                "Please select an object.",
                qt.QMessageBox.Ok | qt.QMessageBox.Default,
                            qt.QMessageBox.NoButton)
        else:
            self.__cutObject = self.__current

    def pasteObject(self):
        if self.__cutObject is None:
            qt.QMessageBox.critical(self, "Error on paste",
                "Please cut an object first.",
                qt.QMessageBox.Ok | qt.QMessageBox.Default,
                            qt.QMessageBox.NoButton)
            return

        if self.__cutObject == self.__current:
            #do nothing
            if DEBUG:
                print("Doing nothing")
            self.__cutObject = None
            self.treeWidget.resizeColumnToContents(0)
            return

        child = self.tree.find(self.__cutObject)
        self.tree.delChild(self.__cutObject)
        destination = self.tree.find(self.__current)
        destination.addChildTree(child)
        if DEBUG:
            print("TREE after addition = ", self.tree)
        self.updateView()

        if 1:
            #this works
            itemList = self.treeWidget.findItems(self.__cutObject,
                      qt.Qt.MatchExactly|qt.Qt.MatchCaseSensitive|qt.Qt.MatchRecursive,
                        0)
            if len(itemList):
                self.treeWidget.scrollToItem(itemList[0],
                    qt.QAbstractItemView.EnsureVisible)
            else:
                if DEBUG:
                    print("Is this a problem?")

        else:
            #this too
            name = self.__cutObject
            while name != 'Scene':
                name = self.tree.parent(name).name()
                itemList = self.treeWidget.findItems(name,
                      qt.Qt.MatchExactly|qt.Qt.MatchCaseSensitive|qt.Qt.MatchRecursive,
                        0)
                if len(itemList):
                    self.treeWidget.expandItem(itemList[0])
                else:
                    if DEBUG:
                        print("Is this a problem?")
                    break
        self.treeWidget.resizeColumnToContents(0)
        self.__cutObject = None
        self.emitSignal('treeChanged')

    def deleteObject(self):
        if self.__current == 'Scene':
            qt.QMessageBox.critical(self, "Error on deletion",
                "You cannot delete the Scene itself.",
                qt.QMessageBox.Ok | qt.QMessageBox.Default,
                            qt.QMessageBox.NoButton)
        self.tree.delChild(self.__current)
        self.__previous = str(self.__current)
        self.setSelectedObject(self.tree.name())
        self.emitSignal('objectDeleted')
        self.updateView()

    def replaceWithObject(self):
        if self.__current in [None, 'None']:
            return
        self.__replacing = True

        if self.__current == 'Scene':
            itemList = self.tree.childList()
            for item in itemList:
                self.tree.delChild(item.name())
            self.updateView()
            self.__cutObject = self.__current * 1
        else:
            self.__cutObject = self.__current * 1
            self.__current = 'Scene'
            child = self.tree.find(self.__cutObject)
            self.tree.delChild(self.__cutObject)
            itemList = self.tree.childList()
            for item in itemList:
                self.tree.delChild(item.name())
            self.tree.addChildTree(child)
            self.updateView()

        itemList = self.treeWidget.findItems(self.__cutObject,
                  qt.Qt.MatchExactly|qt.Qt.MatchCaseSensitive|qt.Qt.MatchRecursive,
                    0)
        if len(itemList):
            self.treeWidget.scrollToItem(itemList[0],
                qt.QAbstractItemView.EnsureVisible)
        self.treeWidget.resizeColumnToContents(0)
        self.__current = self.__cutObject * 1
        self.__cutObject = None
        self.__replacing = False
        self.emitSignal('objectReplaced')

    def itemChanged(self, current, previous):
        if current is None:
            #This happens when updating because I clear the tree
            return
            #This was giving a lot of problems:
            self.__current = 'Scene'
        else:
            self.__current = current.text(0)
        if previous is None:
            self.__previous = None
        else:
            self.__previous = previous.text(0)
        if DEBUG:
            print("current = ", self.__current)
            print("previous = ", self.__previous)
        if self.__current != self.__previous:
            self.setSelectedObject(str(self.__current))
            self.emitSignal('objectSelected')

    def emitSignal(self, event):
        if self.__replacing:
            if DEBUG:
                print("EVENT = ", event, "NOT SENT")
        ddict = {}
        ddict['event'] = event
        ddict['current'] = str(self.__current)
        ddict['previous'] = str(self.__previous)
        self.sigObjectTreeSignal.emit(ddict)


class ObjectActions(qt.QGroupBox):
    def __init__(self, parent = None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle('Object Actions')
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setSpacing(0)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.cutButtonIcon = qt.QIcon(qt.QPixmap(IconDict['cut']))
        self.cutButton = qt.QPushButton(self)
        self.cutButton.setIcon(self.cutButtonIcon)
        self.cutButton.setText('Cut')
        self.pasteButtonIcon = qt.QIcon(qt.QPixmap(IconDict['paste']))
        self.pasteButton = qt.QPushButton(self)
        self.pasteButton.setIcon(self.pasteButtonIcon)
        self.pasteButton.setText('Paste')
        self.deleteButtonIcon = qt.QIcon(qt.QPixmap(IconDict['delete']))
        self.deleteButton = qt.QPushButton(self)
        self.deleteButton.setIcon(self.deleteButtonIcon)
        self.deleteButton.setText('Delete')
        self.replaceButton = qt.QPushButton(self)
        self.replaceButton.setText('Replace')

        self.mainLayout.addWidget(self.cutButton)
        self.mainLayout.addWidget(self.pasteButton)
        self.mainLayout.addWidget(self.deleteButton)
        self.mainLayout.addWidget(self.replaceButton)


if __name__ == "__main__":
    import Object3DBase
    app = qt.QApplication([])
    app.lastWindowClosed.connect(app.quit)
    o0 = Object3DBase.Object3D("DummyObject0")
    o1 = Object3DBase.Object3D("DummyObject1")
    o01 = Object3DBase.Object3D("DummyObject01")
    w = Object3DObjectTree()
    w.addObject(o0, update=False)
    w.addObject(o1, update=False)
    w.addObject(o01, update=True)
    tree = w.tree.find("DummyObject0")
    w.tree.delChild("DummyObject01")
    tree.addChild(o01)
    w.updateView()

    w.show()
    app.exec_()
