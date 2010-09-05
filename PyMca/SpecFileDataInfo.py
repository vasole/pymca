#/*##########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
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
import sys
import PyMcaQt as qt

QTVERSION = qt.qVersion()    
if 0 and qt.qVersion() < '3.0.0':
    import Myqttable as qttable
elif qt.qVersion() < '4.0.0':
    import qttable
if QTVERSION < '4.0.0':
    class QTable(qttable.QTable):
        def __init__(self, parent=None, name=""):
            qttable.QTable.__init__(self, parent, name)
            self.rowCount    = self.numRows
            self.columnCount = self.numCols
            self.setRowCount = self.setNumRows
            self.setColumnCount = self.setNumCols
            self.resizeColumnToContents = self.adjustColumn
else:
    class QTable(qt.QTableWidget):
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

    
import string

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
            self.setMargin(5)
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
        if qt.qVersion() > '3.0.0':
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
                    names = names.split()
                break
        valKeys = ["MotorValues", "motor_pos"] 
        for key in valKeys:
            pos= self.info.get(key, None)
            if pos is not None:
                if key != valKeys[0]:
                    #EDF like ...
                    pos = pos.split()
                break
        if names is not None and pos is not None:
            num= len(names)
            if num != len(pos):
                print "Incorrent number of labels or values"
                return
            if num:
                table= self.__createTable(num, "Motor", "Position")
                for idx in range(num):
                    table.setText(idx, 0, str(names[idx]))
                    table.setText(idx, 1, str(pos[idx]))
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
                print "Incorrent number of labels or values"
                return                
            if num:
                table= self.__createTable(num, "Counter", "Value")
                for idx in range(num):
                    table.setText(idx, 0, str(cnts[idx]))
                    table.setText(idx, 1, str(vals[idx]))
                self.__adjustTable(table)
                self.addTab(table, "Counters")

    def __createHeaderText(self):
        text = self.info.get("SourceType","")
        if text.upper() in ['EDFFILE', 'EDFFILESTACK']:
            return
        text= self.info.get("Header", None)
        if text is not None:
            if qt.qVersion() < '3.0.0':
                wid = qt.QTextView(self)
            elif qt.qVersion() < '4.0.0':
                wid = qt.QTextEdit(self)
                wid.setReadOnly(1)
            else:
                wid = qt.QTextEdit()
                wid.setReadOnly(1)
            if qt.qVersion() < '4.0.0':
                wid.setText(string.join(text, "\n"))
                self.addTab(wid, "Scan Header")
            else:
                wid.insertHtml(string.join(text, "<BR>"))
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
            if qt.qVersion() < '3.0.0':
                wid = qt.QTextView(self)
            elif qt.qVersion() < '4.0.0':
                wid = qt.QTextEdit(self)
                wid.setReadOnly(1)
            else:
                wid = qt.QTextEdit()
                wid.setReadOnly(1)
            if qt.qVersion() < '4.0.0':
                wid.setText(string.join(text, "\n"))
                self.addTab(wid, "File Header")
            else:
                wid.insertHtml(string.join(text, "<BR>"))
                self.addTab(wid, "File Header")

    def closeEvent(self, event):
        ddict = {}
        ddict['event'] = "SpecFileDataInfoClosed"
        ddict['id'] = id(self)
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL("SpecFileDataInfoSignal"), (ddict,))
        else:
            #self.emit(qt.SIGNAL("SpecFileDataInfoSignal"),ddict)
            if len(self._notifyCloseEventToWidget):
                for widget in self._notifyCloseEventToWidget:
                    newEvent = SpecFileDataInfoCustomEvent(ddict)
                    qt.QApplication.postEvent(widget,
                                          newEvent)
                self._notifyCloseEventToWidget = []
        return qt.QTabWidget.closeEvent(self, event)

def test():
    import SpecFileLayer

    if len(sys.argv) < 3:
        print "USAGE: %s <filename> <key>"%sys.argv[0]
        sys.exit(0)

    #d= SpecFileData()
    d = SpecFileLayer.SpecFileLayer()

    d.SetSource(sys.argv[1])
    info,data = d.LoadSource(sys.argv[2])
    #info= d.GetPageInfo({"SourceName":sys.argv[1], "Key":sys.argv[2]})

    if qt.qVersion() < '4.0.0':
        app= qt.QApplication(sys.argv)
        wid= SpecFileDataInfo(info)
        wid.show()

        app.setMainWidget(wid)
        app.exec_loop()
    else:
        app= qt.QApplication([])
        wid= SpecFileDataInfo(info)
        wid.show()
        app.exec_()

if __name__=="__main__":
    test()
