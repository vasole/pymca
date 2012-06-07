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

try:
    from PyMca import XASNormalization
    from PyMca import XASNormalizationWindow
except ImportError:
    print("XASStackNormalizationPlugin importing from somewhere else")
    import XASNormalization
    import XASNormalizationWindow

DEBUG = 0

class XASStackNormalizationPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {}
        text = "Replace current stack by a normalized one."
        function = self.XASNormalize
        info = text
        icon = None
        self.methodDict["XANES Normalization"] =[function,
                                                 info,
                                                 icon]

        self.__methodKeys = ["XANES Normalization"]
        self.widget = None
        
    #Methods implemented by the plugin
    def stackUpdated(self):
        if self.widget is not None:
            self.widget.close()
        self.widget = None

    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    def XASNormalize(self):
        stack = self.getStackDataObject()
        if not isinstance(stack.data, numpy.ndarray):
            text = "This method does not work with dynamically loaded stacks"
            raise TypeError(text)
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        x, spectrum, legend, info = activeCurve
        if self.widget is None:
            self.widget = XASNormalizationWindow.XASNormalizationDialog(None,
                                                spectrum, energy=x)
        else:
            self.widget.setData(spectrum, energy=x)
        ret = self.widget.exec_()
        if ret:
            parameters = self.widget.getParameters()
            # TODO: this dictionary adaptation should be made
            #       by the configuration
            if parameters['auto_edge']:
                edge = None
            else:
                edge = parameters['edge_energy']
            energy = x
            pre_edge_regions = parameters['pre_edge']['regions']
            post_edge_regions = parameters['post_edge']['regions']
            algorithm ='polynomial'
            algorithm_parameters = {}
            algorithm_parameters['pre_edge_order'] = parameters['pre_edge']\
                                                             ['polynomial']
            algorithm_parameters['post_edge_order'] = parameters['post_edge']\
                                                             ['polynomial']
            XASNormalization.replaceStackByXASNormalizedData(stack,
                                            energy=energy,
                                            edge=edge,
                                            pre_edge_regions=pre_edge_regions,
                                            post_edge_regions=post_edge_regions,
                                            algorithm=algorithm,
                                            algorithm_parameters=algorithm_parameters)
            self.setStack(stack)

MENU_TEXT = "XAS Stack Normalization"
def getStackPluginInstance(stackWindow, **kw):
    ob = XASStackNormalizationPlugin(stackWindow)
    return ob
