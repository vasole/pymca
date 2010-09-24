"""

A Stack plugin is a module that will be automatically added to the PyMca stack windows
in order to perform user defined operations on the data stack.

These plugins will be compatible with any stack window that provides the functions:
    #data related
    getStackDataObject
    getStackData
    getStackInfo
    setStack

    #images related
    addImage
    removeImage
    replaceImage

    #mask related
    setSelectionMask
    getSelectionMask

    #displayed curves
    getActiveCurve
    getGraphXLimits
    getGraphYLimits

    #information method
    stackUpdated
    selectionMaskUpdated
"""
import os
import StackPluginBase
try:
    from PyMca import PyMcaQt as qt
    from PyMca import EDFStack
    from PyMca import PyMcaFileDialogs
    from PyMca import StackPluginResultsWindow
    from PyMca import ExternalImagesWindow
    import PyMca.PyMca_Icons as PyMca_Icons
except ImportError:
    print "ExternalImagesWindow importing from somewhere else"
    import PyMcaQt as qt
    import EDFStack
    import PyMcaFileDialogs
    import StackPluginResultsWindow
    import ExternalImagesWindow
    import PyMca_Icons

DEBUG = 0
class ExternalImagesStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {'Load':[self._loadImageFiles,
                                   "Load Images",
                                   PyMca_Icons.fileopen],
                           'Show':[self._showWidget,
                                   "Show Image Browser",
                                   PyMca_Icons.brushselect]}
        self.__methodKeys = ['Load', 'Show']
        self.widget = None

    def stackUpdated(self):
        self.widget = None
            
    def selectionMaskUpdated(self):
        if self.widget is None:
            return
        if self.widget.isHidden():
            return
        mask = self.getStackSelectionMask()
        self.widget.setSelectionMask(mask)

    def mySlot(self, ddict):
        if DEBUG:
            print "mySlot ", ddict['event'], ddict.keys()
        if ddict['event'] == "selectionMaskChanged":
            self.setStackSelectionMask(ddict['current'])
        elif ddict['event'] == "addImageClicked":
            self.addImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "removeImageClicked":
            self.removeImage(ddict['title'])
        elif ddict['event'] == "replaceImageClicked":
            self.replaceImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "resetSelection":
            self.setStackSelectionMask(None)

    #Methods implemented by the plugin
    def getMethods(self):
        if self.widget is None:
            return [self.__methodKeys[0]]
        else:
            return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return apply(self.methodDict[name][0])


    def _loadImageFiles(self):
        if self.getStackDataObject() is None:
            return
        getfilter = True
        fileTypeList = ["PNG Files (*png)",
                        "JPEG Files (*jpg *jpeg)",
                        "IMAGE Files (*)",
                        "EDF Files (*edf)",
                        "EDF Files (*ccd)",
                        "EDF Files (*)"]
        message = "Open image file"
        filenamelist, filefilter = PyMcaFileDialogs.getFileList(parent=None,
                                    filetypelist=fileTypeList,
                                    message=message,
                                    getfilter=getfilter,
                                    single=False,
                                    currentfilter=None)
        if len(filenamelist) < 1:
            return        
        imagelist = []
        imagenames= []
        mask = self.getStackSelectionMask()
        if mask is None:
            r, n = self.getStackROIImagesAndNames()            
            shape = r[0].shape
        else:
            shape = mask.shape
        if filefilter.split()[0] in ["EDF"]:
            for filename in filenamelist:
                #read the edf file
                edf = EDFStack.EdfFileDataSource.EdfFileDataSource(filename)

                #the list of images
                keylist = edf.getSourceInfo()['KeyList']
                if len(keylist) < 1:
                    msg = qt.QMessageBox(None)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Cannot read image from file")
                    msg.exec_()
                    return

                for key in keylist:
                    #get the data
                    dataObject = edf.getDataObject(key)
                    data = dataObject.data
                    if data.shape[0] not in shape:
                        continue
                    if data.shape[1] not in shape:
                        continue
                    imagename  = dataObject.info.get('Title', "")
                    if imagename != "":
                        imagename += " "
                    imagename += os.path.basename(filename)+" "+key
                    imagelist.append(data)
                    imagenames.append(imagename)
            if len(imagelist) == 0:
                msg = qt.QMessageBox(None)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Cannot read a valid image from the file")
                msg.exec_()
                return
            crop = False
            self.widget = StackPluginResultsWindow.StackPluginResultsWindow(parent=None,
                                                    usetab=False)
            self.widget.buildAndConnectImageButtonBox()
            qt.QObject.connect(self.widget,
                   qt.SIGNAL('MaskImageWidgetSignal'),
                   self.mySlot)
            self.widget.setStackPluginResults(imagelist,
                                              image_names=imagenames)
            self._showWidget()
            return
        else:
            #Try pure Image formats
            for filename in filenamelist:
                image = qt.QImage(filename)
                if image.isNull():
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Cannot read file %s as an image" % filename)
                    msg.exec_()
                    return
                imagelist.append(image)
                imagenames.append(os.path.basename(filename))

            if len(imagelist) == 0:
                msg = qt.QMessageBox(None)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Cannot read a valid image from the file")
                msg.exec_()
                return
            self.widget = ExternalImagesWindow.ExternalImagesWindow(parent=None,
                                                    rgbwidget=None,
                                                    selection=True,
                                                    colormap=True,
                                                    imageicons=True,
                                                    standalonesave=True)
            self.widget.buildAndConnectImageButtonBox()
            qt.QObject.connect(self.widget,
                   qt.SIGNAL('MaskImageWidgetSignal'),
                   self.mySlot)
            self.widget.setImageData(None)
            self.widget.setQImageList(imagelist, shape[1], shape[0],
                                                clearmask=False,
                                                data=None,
                                                imagenames=imagenames)
                                                #data=self.__stackImageData)
            self._showWidget()
            return


    def _showWidget(self):
        if self.widget is None:
            return

        #Show
        self.widget.show()
        self.widget.raise_()        

        #update
        self.selectionMaskUpdated()


MENU_TEXT = "External Images Tool"
def getStackPluginInstance(stackWindow, **kw):
    ob = ExternalImagesStackPlugin(stackWindow)
    return ob
