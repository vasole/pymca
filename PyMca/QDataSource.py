"""
Demo example of generic access to data sources.

"""
import sys
if 'qt' not in sys.modules:
    try:
        import PyQt4.Qt as qt
        if qt.qVersion() < '4.1.3':
            print "WARNING: Tested from Qt 4.1.3 on"
    except:
        import qt
else:
    import qt
QTVERSION = qt.qVersion()

#import QSpecFileDataSource
#import QEdfFileDataSource
#import QSPSDataSource
import SpecFileDataSource
import EdfFileDataSource
import QEdfFileWidget
import os
if 0 and QTVERSION < '4.0.0':
    import MySpecFileSelector as QSpecFileWidget
    QSpecFileWidget.QSpecFileWidget = QSpecFileWidget.SpecFileSelector
else:
    import QSpecFileWidget

source_types = { SpecFileDataSource.SOURCE_TYPE: SpecFileDataSource.SpecFileDataSource,
                 EdfFileDataSource.SOURCE_TYPE:  EdfFileDataSource.EdfFileDataSource}
                 #,
                 #QSPSDataSource.SOURCE_TYPE: QSPSDataSource.QSPSDataSource}

source_widgets = { SpecFileDataSource.SOURCE_TYPE: QSpecFileWidget.QSpecFileWidget,
                   EdfFileDataSource.SOURCE_TYPE: QEdfFileWidget.QEdfFileWidget}


def getSourceType(sourceName0):
    if type(sourceName0) == type([]):
        sourceName = sourceName0[0]
    else:
        sourceName = sourceName0
    if os.path.exists(sourceName):
        f = open(sourceName)
        line = f.readline()
        if not len(line):
            line = f.readline()
        if line[0] == "{":
            return EdfFileDataSource.SOURCE_TYPE
        else:
            return SpecFileDataSource.SOURCE_TYPE
    else:
        return QSPSDataSource.SOURCE_TYPE

def QDataSource(name=None, source_type=None):
    if name is None:
        raise ValueError,"Invalid Source Name"
    if source_type is None:
        source_type = getSourceType(name)    
    try:
        sourceClass = source_types[source_type]
    except KeyError:
        #ERROR invalid source type
        raise TypeError,"Invalid Source Type, source type should be one of %s" % source_types.keys()
    return sourceClass(name)
  
  
if __name__ == "__main__":
    import sys,time
    try:
        sourcename=sys.argv[1]
        key       =sys.argv[2]        
    except:
        print "Usage: QDataSource <sourcename> <key>"
        sys.exit()
    #one can use this:
    #obj = EdfFileDataSource(sourcename)
    #or this:
    obj = QDataSource(sourcename)
    #data = obj.getData(key,selection={'pos':(10,10),'size':(40,40)})
    #data = obj.getData(key,selection={'pos':None,'size':None})
    data = obj.getDataObject(key)
    print "info = ",data.info
    print "data shape = ",data.data.shape


