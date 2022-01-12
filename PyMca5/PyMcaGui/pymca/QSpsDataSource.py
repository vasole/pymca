#/*##########################################################################
# Copyright (C) 2004-2020 V.A. Sole, European Synchrotron Radiation Facility
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
import sys
from PyMca5.PyMcaGui.pymca import QSource
from PyMca5.PyMcaCore import SpsDataSource
import logging
_logger = logging.getLogger(__name__)
qt = QSource.qt
QTVERSION = qt.qVersion()

SOURCE_TYPE = SpsDataSource.SOURCE_TYPE

class QSpsDataSource(QSource.QSource):
    """Shared memory source

    The shared memory source object uses SPS through the SPSWrapper
    module to get access to shared memory zones created by Spec or Device Servers

    Emitted signals are :
    updated
    """
    def __init__(self, sourceName):
        QSource.QSource.__init__(self)
        self.__dataSource = SpsDataSource.SpsDataSource(sourceName)
        #easy speed up by making a local reference
        self.sourceName = self.__dataSource.sourceName
        self.isUpdated  = self.__dataSource.isUpdated
        self.sourceType = self.__dataSource.sourceType
        self.getKeyInfo = self.__dataSource.getKeyInfo
        self.refresh    = self.__dataSource.refresh
        self.getSourceInfo = self.__dataSource.getSourceInfo

    def __getattr__(self,attr):
        if not attr.startswith("__"):
            #if not hasattr(qt.QObject, attr):
            if not hasattr(self, attr):
                try:
                    return getattr(self.__dataSource, attr)
                except:
                    pass
        raise AttributeError

    def getDataObject(self,key_list,selection=None, poll=True):
        if poll:
            data = self.__dataSource.getDataObject(key_list,selection)
            self.addToPoller(data)
            return data
        else:
            return self.__dataSource.getDataObject(key_list,selection)

    def customEvent(self, event):
        ddict = event.dict
        if "SourceType" in ddict:
            if ddict["SourceType"] != SOURCE_TYPE:
                _logger.debug("Not a SPS event")
                return
        ddict['SourceName'] = self.__dataSource.sourceName
        ddict['SourceType'] = SOURCE_TYPE
        key = ddict['Key']

        idtolook = []
        ddict['selectionlist'] = []
        
        if key in self.surveyDict:
            for object_ in self.surveyDict[key]:
                idtolook.append(id(object_))

        if key in self.selections.keys():
            n = len(self.selections[key])
            if n:
                a = list(range(n))
                a.reverse()
                legendlist = []
                for i in a:
                    objectId, info = self.selections[key][i]
                    scanselection = 0
                    if 'scanselection' in info:
                        scanselection = info['scanselection']
                    if info['legend'] in legendlist:
                        if not scanselection:
                            del self.selections[key][i]
                            continue
                    if objectId in idtolook:
                        sel = {}
                        sel['SourceName'] = self.__dataSource.sourceName
                        sel['SourceType'] = SOURCE_TYPE
                        sel['Key']        = key
                        sel['selection'] = info['selection']
                        sel['legend'] = info['legend']
                        legendlist.append(info['legend'])
                        sel['targetwidgetid'] = info.get('targetwidgetid', None)
                        sel['scanselection'] = info.get('scanselection', False)
                        sel['imageselection'] = info.get('imageselection', False)
                        ddict['selectionlist'].append(sel)
                    #else:
                        del self.selections[key][i]

                self.sigUpdated.emit(ddict)
            else:
                print("No info????")

if __name__ == "__main__":
    try:
        specname=sys.argv[1]
        arrayname=sys.argv[2]
    except:
        print("Usage: SpsDataSource <specversion> <arrayname>")
        sys.exit()
    app=qt.QApplication([])
    obj = QSpsDataSource(specname)
    def mytest(ddict):
        print(ddict['Key'])
    app.mytest = mytest
    data = obj.getDataObject(arrayname,poll=True)
    obj.sigUpdated.connect(mytest)
    app.exec()

