#/*##########################################################################
# Copyright (C) 2004-2020 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys
from PyMca5.PyMcaGui import PyMcaQt as qt

try:
    from silx.gui.widgets.TableWidget import TableWidget
except ImportError:
    from PyMca5.PyMcaGui.misc.TableWidget import TableWidget

QTVERSION = qt.qVersion()

class QTable(TableWidget):
    def setText(self, row, col, text):
        if qt.qVersion() < "4.0.0":
            QTable.setText(self, row, col, text)
        else:
            item = self.item(row, col)
            if item is None:
                item = qt.QTableWidgetItem(text,
                                           qt.QTableWidgetItem.Type)
                self.setItem(row, col, item)
            else:
                item.setText(text)

class SpecFileDataInfoCustomEvent(qt.QEvent):
    def __init__(self, ddict):
        if ddict is None:
            ddict = {}
        self.dict = ddict
        qt.QEvent.__init__(self, qt.QEvent.User)


class SpecFileDataInfo(qt.QTabWidget):
    InfoTableItems= [
        ("SourceType", "Type"),
        ("SourceName", "Filename"),
        ("Date", "Date"),
        ("Command", "Command"),
        ("Key", "Scan Number"),
        ("Lines", "Nb Lines"),
        ("NbMca", "Nb Mca Spectrum"),
        ("NbMcaDet", "Nb Mca Detectors"),
        ("McaCalib", "Mca Calibration"),
        ("McaPresetTime", "Mca Preset Time"),
        ("McaLiveTime", "Mca Live Time"),
        ("McaRealTime", "Mca Real Time"),
        #EDF Related
        ("HeaderID", "HeaderID"),
        ("Image", "Image"),
        ("DataType", "Data Type"),
        ("ByteOrder", "Byte Order"),
        ("Dim_1", "1st Dimension"),
        ("Dim_2", "2nd Dimension"),
        ("Dim_3", "3rd Dimension"),
        ("Size", "File Data Size"),
    ]

    def __init__(self, info, parent=None, name="DataSpecFileInfo", fl=0):
        if QTVERSION < '4.0.0':
            qt.QTabWidget.__init__(self, parent, name, fl)
            self.setContentsMargins(5, 5, 5, 5)
        else:
            qt.QTabWidget.__init__(self, parent)
            if name is not None:self.setWindowTitle(name)
            self._notifyCloseEventToWidget = []
        self.info= info
        self.__createInfoTable()
        self.__createMotorTable()
        self.__createCounterTable()
        self.__createHeaderText()
        self.__createEDFHeaderText()
        self.__createFileHeaderText()

    if QTVERSION > '4.0.0':
        def sizeHint(self):
            return qt.QSize(2 * qt.QTabWidget.sizeHint(self).width(),
                            3 * qt.QTabWidget.sizeHint(self).height())

        def notifyCloseEventToWidget(self, widget):
            if widget not in self._notifyCloseEventToWidget:
                self._notifyCloseEventToWidget.append(widget)

    def __createInfoTable(self):
        pars= [ par for par in self.InfoTableItems if par[0] in self.info.keys() ]
        num= len(pars)
        if num:
            table= self.__createTable(num, "Parameter", "Value")
            for idx in range(num):
                table.setText(idx, 0, str(pars[idx][1]))
                table.setText(idx, 1, str(self.info.get(pars[idx][0], "-")))
            self.__adjustTable(table)
            self.addTab(table, "Info")

    def __createTable(self, rows, head_par, head_val):
        if qt.qVersion() < '4.0.0':
            table= QTable(self)
        else:
            table= QTable()
        table.setColumnCount(2)
        table.setRowCount(rows)
        if qt.qVersion() < '4.0.0':
            table.setReadOnly(1)
            table.setSelectionMode(QTable.SingleRow)
        else:
            table.setSelectionMode(qt.QTableWidget.NoSelection)
        table.verticalHeader().hide()
        if qt.qVersion() < '4.0.0':
            table.setLeftMargin(0)
            table.horizontalHeader().setLabel(0, head_par)
            table.horizontalHeader().setLabel(1, head_val)
        else:
            labels = [head_par, head_val]
            for i in range(len(labels)):
                item = table.horizontalHeaderItem(i)
                if item is None:
                    item = qt.QTableWidgetItem(labels[i],
                                               qt.QTableWidgetItem.Type)
                item.setText(labels[i])
                table.setHorizontalHeaderItem(i,item)
        return table

    def __adjustTable(self, table):
        for col in range(table.columnCount()):
            table.resizeColumnToContents(col)
        if qt.qVersion() > '4.0.0':
            rheight = table.horizontalHeader().sizeHint().height()
            for row in range(table.rowCount()):
                table.setRowHeight(row, rheight)

    def __createMotorTable(self):
        nameKeys = ["MotorNames", "motor_mne"]
        for key in nameKeys:
            names= self.info.get(key, None)
            if names is not None:
                if key != nameKeys[0]:
                    #EDF like ...
                    tmpString = names.replace('"','')
                    #ID01 specific
                    tmpKey = key + '~1'
                    if tmpKey in self.info:
                        tmpString += self.info[tmpKey].replace('"','')
                    names = tmpString.split()
                break
        valKeys = ["MotorValues", "motor_pos"]
        for key in valKeys:
            pos= self.info.get(key, None)
            if pos is not None:
                if key != valKeys[0]:
                    #EDF like ...
                    tmpString = pos.replace('"', '')
                    #ID01 specific
                    tmpKey = key + '~1'
                    if tmpKey in self.info:
                        tmpString += self.info[tmpKey].replace('"','')
                    pos = tmpString.split()
                break
        if names is not None and pos is not None:
            num= len(names)
            if num != len(pos):
                print("Incorrent number of labels or values")
                return
            if num:
                table= self.__createTable(num, "Motor", "Position")
                if sys.version_info > (3, 3):
                    sorted_list = sorted(names, key=str.casefold)
                else:
                    sorted_list = sorted(names)
                for i in range(num):
                    idx = names.index(sorted_list[i])
                    table.setText(i, 0, str(names[idx]))
                    table.setText(i, 1, str(pos[idx]))
                self.__adjustTable(table)
                self.addTab(table, "Motors")

    def __createCounterTable(self):
        nameKeys = ["LabelNames", "counter_mne"]
        for key in nameKeys:
            cnts= self.info.get(key, None)
            if cnts is not None:
                if key != nameKeys[0]:
                    #EDF like ...
                    cnts = cnts.split()
                break
        valKeys = ["LabelValues", "counter_pos"]
        for key in valKeys:
            vals= self.info.get(key, None)
            if vals is not None:
                if key != valKeys[0]:
                    #EDF like ...
                    vals = vals.split()
                break
        if cnts is not None and vals is not None:
            num= len(cnts)
            if num != len(vals):
                print("Incorrent number of labels or values")
                return
            if num:
                table= self.__createTable(num, "Counter", "Value")
                if sys.version_info > (3, 3):
                    sorted_list = sorted(cnts, key=str.casefold)
                else:
                    sorted_list = sorted(cnts)
                for i in range(num):
                    idx = cnts.index(sorted_list[i])
                    table.setText(i, 0, str(cnts[idx]))
                    table.setText(i, 1, str(vals[idx]))
                self.__adjustTable(table)
                self.addTab(table, "Counters")

    def __createHeaderText(self):
        text = self.info.get("SourceType","")
        if text.upper() in ['EDFFILE', 'EDFFILESTACK']:
            return
        text= self.info.get("Header", None)
        if text is not None:
            if qt.qVersion() < '4.0.0':
                wid = qt.QTextEdit(self)
                wid.setText("\n".join(text))
            else:
                wid = qt.QTextEdit()
                wid.insertHtml("<BR>".join(text))
            wid.setReadOnly(1)
            self.addTab(wid, "Scan Header")

    def __createEDFHeaderText(self):
        text = self.info.get("SourceType","")
        if text.upper() not in ['EDFFILE', 'EDFFILESTACK']:
            return
        keys = self.info.keys()
        nameKeys = []
        vals = []
        for key in keys:
            if key in ['SourceName', 'SourceType']:
                continue
            nameKeys.append(key)
            vals.append(self.info.get(key," --- "))
        num = len(nameKeys)
        if num:
            table= self.__createTable(num, "Keyword", "Value")
            for idx in range(num):
                table.setText(idx, 0, str(nameKeys[idx]))
                table.setText(idx, 1, str(vals[idx]))
            self.__adjustTable(table)
            self.addTab(table, "Header")

    def __createFileHeaderText(self):
        text= self.info.get("FileHeader", None)
        if text not in [None, []]:
            if qt.qVersion() < '4.0.0':
                wid = qt.QTextEdit(self)
                wid.setText("\n".join(text))
            else:
                wid = qt.QTextEdit()
                wid.insertHtml("<BR>".join(text))
            wid.setReadOnly(1)
            self.addTab(wid, "File Header")

    def closeEvent(self, event):
        ddict = {}
        ddict['event'] = "SpecFileDataInfoClosed"
        ddict['id'] = id(self)
        #self.sigSpecFileDataInfoSignal.emit(ddict)
        if len(self._notifyCloseEventToWidget):
            for widget in self._notifyCloseEventToWidget:
                newEvent = SpecFileDataInfoCustomEvent(ddict)
                qt.QApplication.postEvent(widget,
                                      newEvent)
            self._notifyCloseEventToWidget = []
        return qt.QTabWidget.closeEvent(self, event)

def test():
    from PyMca5.PyMcaCore import SpecFileLayer

    if len(sys.argv) < 3:
        print("USAGE: %s <filename> <key>" % sys.argv[0])
        sys.exit(0)

    d = SpecFileLayer.SpecFileLayer()

    d.SetSource(sys.argv[1])
    info, data = d.LoadSource(sys.argv[2])

    app= qt.QApplication([])
    wid= SpecFileDataInfo(info)
    wid.show()
    app.exec()

if __name__ == "__main__":
    test()
