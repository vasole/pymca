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
import numpy
try:
    import StackPluginBase
except ImportError:
    from . import StackPluginBase

DEBUG = 0

class StackNormalizationPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {}
        text  = "Stack/I0 where I0 is the active curve\n"
        function = self.divideByCurve
        info = text
        icon = None
        self.methodDict["I/I0 Normalization"] =[function,
                                                info,
                                                icon]
        text  = "-log(Stack/I0) Normalization where I0 is the active curve\n"
        function = self.logNormalizeByCurve
        info = text
        icon = None
        self.methodDict["-log(I/I0) Normalization"] =[function,
                                                      info,
                                                      icon]
        self.__methodKeys = ["I/I0 Normalization",
                             "-log(I/I0) Normalization"]
        
    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    def divideByCurve(self):
        stack = self.getStackDataObject()
        if not isinstance(stack.data, numpy.ndarray):
            text = "This method does not work with dynamically loaded stacks"
            raise TypeError(text)
        curve = self.getActiveCurve()
        if curve is None:
            text = "Please make sure to have an active curve"
            raise TypeError(text)
        x, y, legend, info = self.getActiveCurve()
        yWork = y[y!=0].astype(numpy.float)
        mcaIndex = stack.info.get('McaIndex', -1)
        if mcaIndex in [-1, 2]:
            for i, value in enumerate(yWork):
                stack.data[:, :, i] = stack.data[:,:,i]/value
        elif mcaIndex == 0:
            for i, value in enumerate(yWork):
                stack.data[i, :, :] = stack.data[i,:,:]/value
        elif mcaIndex == 1:
            for i, value in enumerate(yWork):
                stack.data[:, i, :] = stack.data[:,i,:]/value
        else:
            raise ValueError("Invalid 1D index %d" % mcaIndex)
        self.setStack(stack) 

    def logNormalizeByCurve(self):
        stack = self.getStackDataObject()
        if not isinstance(stack.data, numpy.ndarray):
            text = "This method does not work with dynamically loaded stacks"
            raise TypeError(text)
        curve = self.getActiveCurve()
        if curve is None:
            text = "Please make sure to have an active curve"
            raise TypeError(text)
        x, y, legend, info = self.getActiveCurve()
        yWork = y[y>0]
        mcaIndex = stack.info.get('McaIndex', -1)
        if mcaIndex in [-1, 2]:
            for i, value in enumerate(yWork):
                stack.data[:, :, i] = -numpy.log(stack.data[:,:,i]/value)
        elif mcaIndex == 0:
            for i, value in enumerate(yWork):
                stack.data[i, :, :] = -numpy.log(stack.data[i,:,:]/value)
        elif mcaIndex == 1:
            for i, value in enumerate(yWork):
                stack.data[:, i, :] = -numpy.log(stack.data[:,i,:]/value)
        else:
            raise ValueError("Invalid 1D index %d" % mcaIndex)
        self.setStack(stack) 

MENU_TEXT = "Stack Normalization"
def getStackPluginInstance(stackWindow, **kw):
    ob = StackNormalizationPlugin(stackWindow)
    return ob
