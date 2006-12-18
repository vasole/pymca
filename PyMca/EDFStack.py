import DataObject
import EdfFile
import EdfFileDataSource
import Numeric
import sys
import os
SOURCE_TYPE = "EdfFileStack"
DEBUG = 0

X_AXIS=0
Y_AXIS=1
Z_AXIS=2

class EDFStack(DataObject.DataObject):
    def __init__(self, filelist = None):
        DataObject.DataObject.__init__(self)
        self.incrProgressBar=0
        self.__keyList = []
        if filelist is not None:
            if type(filelist) != type([]):
                filelist = [filelist]
            if len(filelist) == 1:
                self.loadIndexedStack(filelist)
            else:
                self.loadFileList(filelist)

    def loadFileList(self, filelist):
        if type(filelist) == type(''):filelist = [filelist]
        self.__keyList = []
        self.sourceName = filelist
        self.__indexedStack = True
        self.sourceType = SOURCE_TYPE
        self.info = {}
        self.nbFiles=len(filelist)

        #read first edf file
        #get information
        if 1:
            tempEdf=EdfFileDataSource.EdfFileDataSource(filelist[0])
            keylist = tempEdf.getSourceInfo()['KeyList']
            nImages = len(keylist)
            dataObject = tempEdf.getDataObject(keylist[0])
            self.info.update(dataObject.info)
            arrRet = dataObject.data
        else:
            tempEdf = EdfFile.EdfFile(filelist[0])        
            header=tempEdf.GetHeader(0)
            arrRet=tempEdf.GetData(0)
            self.info.update(header)
            nImages = tempEdf.GetNumImages()

        self.onBegin(self.nbFiles)
        singleImageShape = arrRet.shape
        if len(singleImageShape) == 1:
            #single line
            #be ready for specfile stack?
            raise "IOError", "Not implemented yet"
            self.data = Numeric.zeros((arrRet.shape[0],
                                       nImages,
                                       self.nbFiles),
                                       arrRet.typecode())
            self.incrProgressBar=0
            for tempEdfFileName in filelist:
                tempEdf=EdfFile.EdfFile(tempEdfFileName)
                for i in range(nImages):
                    pieceOfStack=tempEdf.GetData(i)
                    self.data[:,i, self.incrProgressBar] = pieceOfStack[:]
                self.incrProgressBar += 1
                self.onProgress(self.incrProgressBar)
            self.onEnd()
        else:
            if nImages > 1:
                #this is not the common case
                #should I try to convert it to a standard one
                #using a 3D matrix or kepp as 4D matrix?
                raise "IOError", "Not implemented yet"
                self.data = Numeric.zeros((arrRet.shape[0],
                                           arrRet.shape[1],
                                           nImages * self.nbFiles),
                                           arrRet.typecode())
                self.incrProgressBar=0
                for tempEdfFileName in filelist:
                    tempEdf=EdfFile.EdfFile(tempEdfFileName)
                    for i in range(nImages):
                        pieceOfStack=tempEdf.GetData(i)
                        self.data[:,:,
                                  nImages*self.incrProgressBar+i] = \
                                                  pieceOfStack[:,:]
                    self.incrProgressBar += 1
            else:
                #this is the common case
                self.data = Numeric.zeros((arrRet.shape[0],
                                           arrRet.shape[1],
                                           self.nbFiles),
                                           arrRet.typecode())
                self.incrProgressBar=0
                for tempEdfFileName in filelist:
                    tempEdf=EdfFile.EdfFile(tempEdfFileName)
                    pieceOfStack=tempEdf.GetData(0)    
                    self.data[:,:, self.incrProgressBar] = pieceOfStack[:,:]
                    self.incrProgressBar += 1
                    self.onProgress(self.incrProgressBar)
                self.onEnd()

        self.__nFiles         = self.incrProgressBar
        self.__nImagesPerFile = nImages
        shape = self.data.shape
        for i in range(len(shape)):
            key = 'Dim_%d' % (i+1,)
            self.info[key] = shape[i]
        self.info["SourceType"] = SOURCE_TYPE
        self.info["SourceName"] = self.sourceName
        self.info["Size"]       = self.__nFiles * self.__nImagesPerFile
        self.info["NumberOfFiles"] = self.__nFiles * 1

    def onBegin(self, n):
        pass

    def onProgress(self, n):
        pass

    def onEnd(self):
        pass

    def loadIndexedStack(self,filename,begin=None,end=None, skip = None):
        if begin is None: begin = 0
        if type(filename) == type([]):
            filename = filename[0]
        if not os.path.exists(filename):
            raise "IOError","File %s does not exists" % filename
        name = os.path.basename(filename)
        n = len(name)
        i = 1
        numbers = ['0', '1', '2', '3', '4', '5',
                   '6', '7', '8','9']
        while (i <= n):
            c = name[n-i:n-i+1]
            if c in ['0', '1', '2',
                                '3', '4', '5',
                                '6', '7', '8',
                                '9']:
                break
            i += 1
        suffix = name[n-i+1:]
        if len(name) == len(suffix):
            #just one file, one should use standard widget
            #and not this one.
            self.loadFileList(filename)
        else:
            nchain = []
            while (i<=n):
                c = name[n-i:n-i+1]
                if c not in ['0', '1', '2',
                                    '3', '4', '5',
                                    '6', '7', '8',
                                    '9']:
                    break
                else:
                    nchain.append(c)
                i += 1
            number = ""
            nchain.reverse()
            for c in nchain:
                number += c
            format = "%" + "0%dd" % len(number)
            if (len(number) + len(suffix)) == len(name):
                prefix = ""
            else:
                prefix = name[0:n-i+1]
            prefix = os.path.join(os.path.dirname(filename),prefix)
            if not os.path.exists(prefix + number + suffix):
                print "Internal error in EDFStack"
                print "file should exist:",prefix + number + suffix
                return
            i = 0
            if begin is None:
                i = 0
            else:
                if not os.path.exists(prefix+format % begin+suffix):
                    raise "ValueError","Invalid start index file = %s" % \
                          prefix+format % begin+suffix
                else:
                    i = begin
            while not os.path.exists(prefix+format % i+suffix):
                i += 1
            f = prefix+format % i+suffix
            filelist = []
            while os.path.exists(f):
                filelist.append(f)
                i += 1
                if end is not None:
                    if i > end:
                        break
                f = prefix+format % i+suffix
            self.loadFileList(filelist)


    def loadIndexedStackOLD(self,path,begin=0,end=70, format = "%04d"):
        """
        This is for Artemis-like indexed files
        """
        self.__keyList = []
        radix=self.getRadix(path)
        begin,end=int(begin),int(end)
        
        self.__indexedStack = True
        self.sourceName = ["IndexedStack",
                           "%s" % path,
                           format % begin,
                           format % end]
        
        self.sourceType = SOURCE_TYPE
        
        minVal,maxVal=0,0
        x,y=0,0
        self.nbFiles=end-begin+1

        self.incrProgressBar=0
        
        #read first edf file thru EdfFileSource???
        tempEdfFileName=radix+self.completeRadix(str(begin))
        self.info = {}
        if 1:
            tempEdf=EdfFileDataSource.EdfFileDataSource(tempEdfFileName)
            keylist = tempEdf.getSourceInfo()['KeyList']
            dataObject = tempEdf.getDataObject(keylist[0])
            self.info.update(dataObject.info)
            arrRet = dataObject.data
        else:
            tempEdf=EdfFile.EdfFile(tempEdfFileName)
            header=tempEdf.GetHeader(0)
            arrRet=tempEdf.GetData(0)
            self.info.update(header)
        #self.eh.event(self.incrProgressBar)
        self.incrProgressBar += 1

        self.data = Numeric.zeros((arrRet.shape[0],
                                   arrRet.shape[1],
                                   self.nbFiles),
                              arrRet.typecode())
        for row in range(begin+1,end+1):
            #build EDF file name
            #print "read filename number : ",row
            tempEdfFileName=radix+self.completeRadix(str(row))
            tempEdf=EdfFile.EdfFile(tempEdfFileName)

            pieceOfStack=tempEdf.GetData(0)
            self.incrProgressBar += 1

            #update extremal values perhaps temporary
            self.Xmax,self.Ymax=pieceOfStack.shape
            self.data[:,:, row] = pieceOfStack[:,:]

        self.__nFiles         = self.incrProgressBar
        self.__nImagesPerFile = 1
        shape = self.data.shape
        for i in range(len(shape)):
            key = 'Dim_%d' % (i+1,)
            self.info[key] = shape[i]
        self.info["SourceType"] = SOURCE_TYPE
        self.info["SourceName"] = self.sourceName
        self.info["Size"]       = self.__nFiles * self.__nImagesPerFile
        self.info["NumberOfFiles"] = self.__nFiles * 1

    def getSourceInfo(self):
        sourceInfo = {}
        sourceInfo["SourceType"]=SOURCE_TYPE
        if self.__keyList == []:
            for i in range(1, self.__nFiles + 1):
                for j in range(1, self.__nImages + 1):
                    self.__keyList.append("%d.%d" % (i,j))
        sourceInfo["KeyList"]= self.__keyList

    def getKeyInfo(self, key):
        print "Not implemented"
        return {}

    def isIndexedStack(self):
        return self.__indexedStack
    
    def getZSelectionArray(self,z=0):
        return (self.data[:,:,z]).astype(Numeric.Float)
        
    def getXYSelectionArray(self,coord=(0,0)):
        x,y=coord    
        return (self.data[y,x,:]).astype(Numeric.Float)
    
    
    def getRadix(self,chaine):
        """
        split the number extension (ex: '_0000.edf' ) of an edf filename
        """
        return(chaine[:(len(chaine)-9)])
    
    def completeRadix(self,strNumber):
        """
        return the number extension (ex: '_0000.edf' ) corresponding to the strNumber  to complete the radix and make a standard .edf filename
        """
        number=int(strNumber)
        if (number<10) :
            return("_000"+strNumber+".edf")
        elif ((number>=10) and (number<100)) :
            return("_00"+strNumber+".edf")
        elif ((number>=100) and (number<1000)) :
            return("_0"+strNumber+".edf")
        else :
            return("_"+strNumber+".edf")
    
    def packDownPages(self):
        x,y=self.GetPageSize()
        z=self.GetNumberPages()
        arr=None
        info=self.GetPageInfo(0)
        for page in range(self.GetNumberPages()):
            a=self.GetPageArray(0)
            self.eh.event(self.incrProgressBar)
            
            if arr is None : 
                arr=a
            else:
                arr=concatenate((arr,a))
            self.Delete(0)
                
        arr=reshape(arr,(z,y,x))
        self.AppendPage(info,arr)
        
    def splitPageAlongAxis(self,page=0,axis=X_AXIS):
        x,y,z=self.GetPageArray(0).shape
        info=self.GetPageInfo(0)
        if axis==X_AXIS:
            for i in range(x):
                arraySlice=(self.GetPageArray(0))[i,:,:]
                self.AppendPage(info,arraySlice)
            self.Delete(0)
        elif axis==Y_AXIS:
            for i in range(y):
                arraySlice=(self.GetPageArray(0))[:,i,:]
                self.AppendPage(info,arraySlice)
            self.Delete(0)
        elif axis==Z_AXIS:
            for i in range(z):
                arraySlice=(self.GetPageArray(0))[:,:,i]
                self.AppendPage(info,arraySlice)
            self.Delete(0)

if __name__ == "__main__":
    import time
    t0= time.time()
    stack = EDFStack()
    #stack.loadIndexedStack("Z:\COTTE\ch09\ch09__mca_0005_0000_0070.edf")
    stack.loadIndexedStack(".\COTTE\ch09\ch09__mca_0005_0000_0070.edf")
    shape = stack.data.shape
    print "elapsed = ", time.time() - t0
    #guess the MCA
    imax = 0
    for i in range(len(shape)):
        if shape[i] > shape[imax]:
            imax = i

    print "selections ",
    print "getZSelectionArray  shape = ", stack.getZSelectionArray().shape
    print "getXYSelectionArray shape = ", stack.getXYSelectionArray().shape

    try:
        import PyQt4.Qt as qt
    except:
        import qt
    app = qt.QApplication([])
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                       app, qt.SLOT("quit()"))
    if 1:
        import RGBCorrelatorGraph
        w = RGBCorrelatorGraph.RGBCorrelatorGraph()
        graph = w.graph
    else:
        import QtBlissGraph
        w = QtBlissGraph.QtBlissGraph()
        graph = w
    print "shape sum 0 = ",Numeric.sum(stack.data, 0).shape
    print "shape sum 1 = ",Numeric.sum(stack.data, 1).shape
    print "shape sum 2 = ",Numeric.sum(stack.data, 2).shape
    a = Numeric.sum(stack.data, imax)
    print a.shape
    graph.setX1AxisLimits(0, a.shape[0])
    if 0:
        w.setY1AxisLimits(0, a.shape[1])
        w.setY1AxisInverted(True)
    else:
        graph.setY1AxisInverted(True)
        graph.setY1AxisLimits(0, a.shape[1])
    graph.imagePlot(a, ymirror=0)
    if imax == 0:
        graph.x1Label('Column Number')
    else:
        graph.x1Label('Row Number')
    graph.ylabel('File Number')
    w.show()

    if imax == 0:
        mcaData0 = Numeric.sum(Numeric.sum(stack.data, 2),1)
    else:
        mcaData0 = Numeric.sum(Numeric.sum(stack.data, 2),0)

    import McaWindow
    mca = McaWindow.McaWidget()
    sel = {}
    sel['SourceName'] = "EDF Stack"
    sel['Key']        = "SUM"
    sel['legend']     = "EDF Stack SUM"
    mcaData = DataObject.DataObject()
    mcaData.info = {'McaCalib': [0 , 2.0 ,0],
                    "selectiontype":"1D",
                    "SourceName":"EDF Stack",
                    "Key":"SUM"}
    mcaData.x = [Numeric.arange(len(mcaData0)).astype(Numeric.Float)]
    mcaData.y = [mcaData0]
    sel['dataobject'] = mcaData
    mca.show()
    mca._addSelection([sel])
    graph.enableSelection(True)
    def graphSlot(ddict):
        if ddict['event'] == "MouseSelection":
            ix1 = int(ddict['xmin'])
            ix2 = int(ddict['xmax'])+1
            iy1 = int(ddict['xmin'])
            iy2 = int(ddict['xmax'])+1
            if imax == 0:
                selectedData = Numeric.sum(Numeric.sum(stack.data[:,ix1:ix2, iy1:iy2], 2),1)
            else:
                selectedData = Numeric.sum(Numeric.sum(stack.data[ix1:ix2,:, iy1:iy2], 2),0)
            sel = {}
            sel['SourceName'] = "EDF Stack"
            sel['Key'] = "Selection"
            sel["selectiontype"] = "1D"
            sel['legend'] = "EDF Stack Selection"
            selDataObject = DataObject.DataObject()
            selDataObject.info={'McaCalib': [100 , 2.0 ,0],
                                "selectiontype":"1D",
                                "SourceName":"EDF Stack Selection",
                                "Key":"Selection"}
            selDataObject.x = [Numeric.arange(len(mcaData0)).astype(Numeric.Float)]
            selDataObject.y = [selectedData]
            sel['dataobject'] = selDataObject
            mca._addSelection([sel])
    qt.QObject.connect(graph, qt.SIGNAL('QtBlissGraphSignal'),
                       graphSlot)
    #w.replot()
    if qt.qVersion() < '4.0.0':
        app.exec_loop()
    else:
        app.exec_()
