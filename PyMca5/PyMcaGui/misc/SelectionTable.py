#/*##########################################################################
# Copyright (C) 2004-2018 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """

Concenience widget to generate a selection  table in which each column can
contain one type of widget.

For the time being only text, check boxes or radio buttons are supported.

Each time one of the contained widgets changes, a sigSelectionTableSignal
is emitted indicating the current selection and the triggering cell.
"""
from PyMca5.PyMcaGui import PyMcaQt as qt


class SelectionTable(qt.QTableWidget):
    sigSelectionTableSignal = qt.pyqtSignal(object)

    LABELS = ["Legend", "X", "Y"]
    TYPES = ["Text", "RadioButton", "CheckBox"]

    def __init__(self, parent=None, labels=None, types=None):
        qt.QTableWidget.__init__(self, parent)
        if labels is None:
            if types is None:
                labels = self.LABELS
                types = self.TYPES
            else:
                labels = []
                i = 0
                for item in types:
                    if item.lower() not in ["text", "checkbox", "radiobutton"]:
                        text = "Only Text, CheckBox or RadioButton accepted"
                        raise TypeError(text)
                    labels.append("Column %02d" % i)
                    i += 1
        elif types is None:
            types = []
            for item in labels:
                try:
                    if len(item) == 1:
                        types.append("CheckBox")
                    else:
                        types.append("Text")
                except:
                    types.append("Text")

        self.setColumnCount(len(labels))
        for i in range(len(labels)):
            item = self.horizontalHeaderItem(i)
            if item is None:
                item = qt.QTableWidgetItem(labels[i],
                                           qt.QTableWidgetItem.Type)
                self.setHorizontalHeaderItem(i, item)
            item.setText(labels[i])
        rheight = self.horizontalHeader().sizeHint().height()

        self._labels = labels
        self._types = types
        self._buttonGroups = [None] * len(self._labels)
        # set a minimum of 5 rows
        self.setMinimumHeight(5*rheight)

    def fillTable(self, entries):
        """
        Fill the table with the given entries.

        :param entries: List in which each item is a list of strings.
        The list of strings has to match the length of the table top header.
        :param type: list
        """
        nEntries = len(entries)
        self.setRowCount(nEntries)

        for i in range(nEntries):
            self.fillLine(i, entries[i])

        # adjust column width
        for i in range(self.columnCount()):
            self.resizeColumnToContents(i)

    def fillLine(self, row, entry):
        for column in range(len(self._types)):
            content = entry[column]
            if self._types[column].lower() == "checkbox":
                widget = self.cellWidget(row, column)
                if widget is None:
                    widget = CheckBoxItem(self, row, column)
                    self.setCellWidget(row, column, widget)
                    widget.sigCheckBoxItemSignal.connect(self._checkBoxSlot)
                widget.setText(content)
            elif self._types[column].lower() == "radiobutton":
                widget = self.cellWidget(row, column)
                if widget is None:
                    widget = RadioButtonItem(self, row, column)
                    self.setCellWidget(row, column, widget)
                    widget.sigRadioButtonItemSignal.connect( \
                                        self._radioButtonSlot)
                    if self._buttonGroups[column] is None:
                        self._buttonGroups[column] = qt.QButtonGroup()
                        widget.setChecked(True)
                    self._buttonGroups[column].addButton(widget)
                widget.setText(content)
            else:
                # text
                item = self.item(row, column)
                if item is None:
                    item = qt.QTableWidgetItem(content,
                                               qt.QTableWidgetItem.Type)
                    item.setTextAlignment(qt.Qt.AlignHCenter | qt.Qt.AlignVCenter)
                    self.setItem(row, column, item)
                    # item is enabled and selectable
                    item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
                else:
                    item.setText(content)

    def _checkBoxSlot(self, ddict):
        row = ddict["row"]
        column = ddict["column"]
        self.emitSelectionChangedSignal(cell=(row, column))

    def _radioButtonSlot(self, ddict):
        # I handle them in the same way
        return self._checkBoxSlot(ddict)

    def getSelection(self):
        ddict = {}
        for column in range(self.columnCount()):
            label = self._labels[column].lower()
            ddict[label] = []
            if self._types[column].lower() in ["checkbox", "radiobutton"]:
                for row in range(self.rowCount()):
                    if self.cellWidget(row, column).isChecked():
                        ddict[label].append(row)
            else:
                for row in range(self.rowCount()):
                    ddict[label].append(self.item(row, column).text())
        return ddict

    def setSelection(self, ddict, qtCall=None):
        if qtCall is not None:
            return super(SelectionTable, self).setSelection(ddict, qtCall)
        for key in ddict:
            for column in range(self.columnCount()):
                if key.lower() == self._labels[column].lower():
                    if self._types[column].lower() in ["checkbox", "radiobutton"]:
                        for row in range(self.rowCount()):
                            widget = self.cellWidget(row, column)
                            if row in ddict[key]:
                                widget.setChecked(True)
                            else:
                                widget.setChecked(False)
                    else:
                        # text
                        if self.rowCount() < len(ddict[key]):
                            self.setRowCount(len(ddict[key]))
                        for row in range(len(ddict[key])):
                            content = ddict[key][row]
                            item = self.item(row, column)
                            item.setText(content)

    def emitSelectionChangedSignal(self, cell=None):
        ddict = self.getSelection()
        ddict["event"] = "selectionChanged"
        ddict["cell"] = cell
        self.sigSelectionTableSignal.emit(ddict)

    def setColumnEnabled(self, index, enabled):
        if index < self.columnCount():
            for row in range(self.rowCount()):
                self.cellWidget(row, index).setEnabled(enabled)


class CheckBoxItem(qt.QCheckBox):
    sigCheckBoxItemSignal = qt.pyqtSignal(object)

    def __init__(self, parent, row, col):
        super(CheckBoxItem, self).__init__(parent)
        self.__row = row
        self.__col = col
        self.clicked[bool].connect(self._mySignal)

    def _mySignal(self, value):
        ddict = {}
        ddict["event"] = "clicked"
        ddict["state"] = value
        ddict["row"] = self.__row * 1
        ddict["column"] = self.__col * 1
        # for compatibility ...
        ddict["col"] = ddict["column"]
        self.sigCheckBoxItemSignal.emit(ddict)


class RadioButtonItem(qt.QRadioButton):
    sigRadioButtonItemSignal = qt.pyqtSignal(object)

    def __init__(self, parent, row, col):
        super(RadioButtonItem, self).__init__(parent)
        self.__row = row
        self.__col = col
        self.clicked[bool].connect(self._mySignal)

    def _mySignal(self, value):
        ddict = {}
        ddict["event"] = "clicked"
        ddict["state"] = value
        ddict["row"] = self.__row * 1
        ddict["column"] = self.__col * 1
        # for compatibility ...
        ddict["col"] = ddict["column"]
        self.sigRadioButtonItemSignal.emit(ddict)


if __name__ == "__main__":
    app = qt.QApplication([])
    def slot(ddict):
        print("received dict = ", ddict)
    tab = SelectionTable(labels=["Legend", "X", "Y"],
                         types=["Text", "RadioButton", "CheckBox"])
    tab.sigSelectionTableSignal.connect(slot)
    tab.fillTable([["Cnt1", "", ""],
                   ["Cnt2", "", ""],
                   ["Cnt3", "", ""],
                   ["Cnt4", "", ""],
                   ["Cnt5", "", ""]])
    tab.setSelection({'x': [1],
                      'y': [4],
                      'legend': ["dummy", "Ca K", "Fe K", "Pb M", "U l"]})
    tab.show()
    app.exec()
