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
import StackPluginBase
try:
    from PyMca import SGWindow
    from PyMca import SNIPWindow
    import PyMca.PyMca_Icons as PyMca_Icons
except ImportError:
    print "Plugin importing from somewhere else"
    import SGWindow
    import SNIPWindow
    import PyMca_Icons

DEBUG = 0

class BackgroundStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        SGtext  = "Replace current stack by a\n"
        SGtext += "Savitsky-Golay treated one."
        SNIP1Dtext  = "Replace current stack by a\n"
        SNIP1Dtext += "SNIP1D treated one."
        SNIP2Dtext  = "Replace current stack by a\n"
        SNIP2Dtext += "SNIP2D treated one."
        self.methodDict = {}
        function = self.replaceStackWithSavitzkyGolayFiltering
        info = SGtext
        icon = PyMca_Icons.substract
        self.methodDict["Savitzky-Golay Filtering"] =[function,
                                                      info,
                                                      icon]
        function = self.subtract1DSnipBackground
        info = SNIP1Dtext
        self.methodDict["Subtract SNIP 1D Background"] =[function,
                                                      info,
                                                      icon]
        function = self.replaceWith1DSnipBackground
        info  = "Smooth and replace current stack\n"
        info += "by its SNIP1D background."
        self.methodDict["Deglitch with SNIP 1D Background"] =[function,
                                                              info,
                                                              PyMca_Icons.smooth]
        function = self.subtract2DSnipBackground
        info = SNIP2Dtext
        self.methodDict["Subtract SNIP 2D Background"] =[function,
                                                      info,
                                                      icon]

        self.__methodKeys = ["Savitzky-Golay Filtering",
                             "Deglitch with SNIP 1D Background",
                             "Subtract SNIP 1D Background",
                             "Subtract SNIP 2D Background"]
                                     
    def stackUpdated(self):
        self.dialogWidget = None

    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return apply(self.methodDict[name][0])

    def replaceStackWithSavitzkyGolayFiltering(self):
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        x, spectrum, legend, info = activeCurve
        snipWindow = SGWindow.SGDialog(None,
                                           spectrum, x=x)
        snipWindow.graph.setGraphXTitle(info['xlabel'])
        snipWindow.graph.setGraphYTitle(info['ylabel'])
        #snipWindow.setModal(True)
        snipWindow.show()
        ret = snipWindow.exec_()
        if ret:
            snipParametersDict = snipWindow.getParameters()
            snipWindow.close()
            function = snipParametersDict['function']
            arguments = snipParametersDict['arguments']
            stack = self.getStackDataObject()
            function(stack, *arguments)
            self.setStack(stack)

    def subtract1DSnipBackground(self, smooth=False):
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        x, spectrum, legend, info = activeCurve
        snipWindow = SNIPWindow.SNIPDialog(None,
                                           spectrum, x=x, smooth=smooth)
        snipWindow.graph.setGraphXTitle(info['xlabel'])
        snipWindow.graph.setGraphYTitle(info['ylabel'])
        #snipWindow.setModal(True)
        snipWindow.show()
        ret = snipWindow.exec_()
        if ret:
            snipParametersDict = snipWindow.getParameters()
            snipWindow.close()
            function = snipParametersDict['function']
            arguments = snipParametersDict['arguments']
            stack = self.getStackDataObject()
            function(stack, *arguments)
            self.setStack(stack)

    def replaceWith1DSnipBackground(self):
        return self.subtract1DSnipBackground(smooth=True)

    def subtract2DSnipBackground(self):
        imageList = self.getStackROIImagesAndNames()
        if imageList is None:
            return
        imageList, imageNames = imageList
        if not len(imageList):
            return
        snipWindow = SNIPWindow.SNIPDialog(None,
                                           imageList[0]*1)
        #snipWindow.setModal(True)
        snipWindow.show()
        ret = snipWindow.exec_()
        if ret:
            snipParametersDict = snipWindow.getParameters()
            snipWindow.close()
            function = snipParametersDict['function']
            arguments = snipParametersDict['arguments']
            stack = self.getStackDataObject()
            function(stack, *arguments)
            self.setStack(stack)


MENU_TEXT = "Stack Filtering Options"
def getStackPluginInstance(stackWindow, **kw):
    ob = BackgroundStackPlugin(stackWindow)
    return ob
