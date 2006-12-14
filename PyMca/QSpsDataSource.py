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
import sys
import QSource
import SpsDataSource
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

    def __getattr__(self,attr):
        print " attr = ",attr
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
        dict = event.dict
        dict['SourceName'] = self.__dataSource.sourceName
        dict['SourceType'] = SOURCE_TYPE
        #dict['Key'] should be already there
        #dict['event'] should be there
        #dict['id'] should also be there
        #print "emitted dict =", dict
        #for objectref in dict['id']:
        #    print "dict[id].info = ",objectref.info
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL("updated"), (dict,))
        else:
            self.emit(qt.SIGNAL("updated"), dict)
        
if __name__ == "__main__":
    import sys
    try:
        specname=sys.argv[1]
        arrayname=sys.argv[2]        
    except:
        print "Usage: SpsDataSource <specversion> <arrayname>"
        sys.exit()
    app=qt.QApplication([])
    obj = QSpsDataSource(specname)    
    def mytest(dict):
        print dict['Key']
    app.mytest = mytest
    data = obj.getDataObject(arrayname,poll=True)
    if QTVERSION < '4.0.0':
        qt.QObject.connect(obj,qt.PYSIGNAL('updated'),mytest) 
        app.exec_loop()
    else:
        qt.QObject.connect(obj,qt.SIGNAL('updated'),mytest) 
        app.exec_()
    
