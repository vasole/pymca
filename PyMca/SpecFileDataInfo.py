#/*##########################################################################
# Copyright (C) 2004-2006 European Synchrotron Radiation Facility
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
# is a problem to you.
#############################################################################*/
import qt
if 0 and qt.qVersion() < '3.0.0':
    import Myqttable as qttable
else:
    import qttable
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
    ]

    def __init__(self, info, parent=None, name="DataSpecFileInfo", fl=0):
        qt.QTabWidget.__init__(self, parent, name, fl)

        self.setMargin(5)
        self.info= info
        self.__createInfoTable()
        self.__createMotorTable()
        self.__createCounterTable()
        self.__createHeaderText()

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
        table= qttable.QTable(self)
        table.setNumCols(2)
        table.setNumRows(rows)
        if qt.qVersion() > '3.0.0':
            table.setReadOnly(1)
            table.setSelectionMode(qttable.QTable.SingleRow)
        table.verticalHeader().hide()
        table.setLeftMargin(0)
        table.horizontalHeader().setLabel(0, head_par)
        table.horizontalHeader().setLabel(1, head_val)
        return table

    def __adjustTable(self, table):
        for col in range(table.numCols()):
            table.adjustColumn(col)

    def __createMotorTable(self):
        names= self.info.get("MotorNames", None)
        pos= self.info.get("MotorValues", None)
        if names is not None and pos is not None:
            num= len(names)
            if num:
                table= self.__createTable(num, "Motor", "Position")
                for idx in range(num):
                    table.setText(idx, 0, str(names[idx]))
                    table.setText(idx, 1, str(pos[idx]))
                self.__adjustTable(table)
                self.addTab(table, "Motors")

    def __createCounterTable(self):
        cnts= self.info.get("LabelNames", None)
        vals= self.info.get("LabelValues", None)
        if cnts is not None and vals is not None:
            num= len(cnts)
            if num:
                table= self.__createTable(num, "Counter", "Value")
                for idx in range(num):
                    table.setText(idx, 0, str(cnts[idx]))
                    table.setText(idx, 1, str(vals[idx]))
                self.__adjustTable(table)
                self.addTab(table, "Counters")

    def __createHeaderText(self):
        text= self.info.get("Header", None)
        if text is not None:
            if qt.qVersion() < '3.0.0':
                wid = qt.QTextView(self)
            else:
                wid = qt.QTextEdit(self)
                wid.setReadOnly(1)
            wid.setText(string.join(text, "\n"))
            self.addTab(wid, "File Header")

def test():
    import sys
    #from SpecFileData import SpecFileData
    import SpecFileLayer

    if not len(sys.argv):
        print "USAGE: %s <filename> <key>"%sys.argv[0]
        sys.exit(0)

    #d= SpecFileData()
    d = SpecFileLayer.SpecFileLayer()

    d.SetSource(sys.argv[1])
    info,data = d.LoadSource(sys.argv[2])
    #info= d.GetPageInfo({"SourceName":sys.argv[1], "Key":sys.argv[2]})

    app= qt.QApplication(sys.argv)
    wid= SpecFileDataInfo(info)
    wid.show()

    app.setMainWidget(wid)
    app.exec_loop()

if __name__=="__main__":
    test()
