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
            oldParameters = self.widget.getParameters()
            oldEnergy = self.widget.parametersWidget.energy
            oldEMin = oldEnergy.min()
            oldEMax = oldEnergy.max()
            self.widget.setData(spectrum, energy=x)
            if abs(oldEMin - x.min()) < 1:
                if abs(oldEMax - x.max()) < 1:
                    self.widget.setParameters(oldParameters)
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
            self.replaceStackByXASNormalizedData(stack,
                                            energy=energy,
                                            edge=edge,
                                            pre_edge_regions=pre_edge_regions,
                                            post_edge_regions=post_edge_regions,
                                            algorithm=algorithm,
                                            algorithm_parameters=algorithm_parameters)
            self.setStack(stack)

    def replaceStackByXASNormalizedData(self,
                                        stack,
                                        energy=None,
                                        edge=None,
                                        pre_edge_regions=None,
                                        post_edge_regions=None,
                                        algorithm='polynomial',
                                        algorithm_parameters=None):
        """
        Performs an in place replacement of a set of spectra by a set of
        normalized spectra.
        """
        mcaIndex = -1
        if hasattr(stack, "info") and hasattr(stack, "data"):
            actualData = stack.data
            mcaIndex = stack.info.get('McaIndex', -1)
        else:
            actualData = stack
        if not isinstance(actualData, numpy.ndarray):
            raise TypeError("Currently this method only supports numpy arrays")

        # Take a data view
        oldShape = actualData.shape
        data = actualData[:]
        DONE = 0
        if mcaIndex in [-1, len(data.shape)-1]:
            data.shape = -1, oldShape[-1]
            for i in range(data.shape[0]):
                ene, spe, ed = XASNormalization.XASNormalization(data[i,:],
                                        energy=energy,
                                        edge=edge,
                                        pre_edge_regions=pre_edge_regions,
                                        post_edge_regions=post_edge_regions,
                                        algorithm=algorithm,
                                        algorithm_parameters=algorithm_parameters)[0:3]
                if not DONE:
                    c0 = (numpy.nonzero(energy >= (ed + pre_edge_regions[0][0]))[0]).min()
                    c1 = (numpy.nonzero(energy <= (ed + post_edge_regions[-1][1]))[-1]).max()
                    DONE = True
                data[i,:c0] = spe[c0]
                data[i, c0:c1] = spe[c0:c1]
                data[i, c1:] = spe[c1]
            data.shape = oldShape
        elif mcaIndex == 0:
            data.shape = oldShape[0], -1
            for i in range(data.shape[-1]):
                ene, spe, ed = XASNormalization.XASNormalization(data[i,:],
                                        energy=energy,
                                        edge=edge,
                                        pre_edge_regions=pre_edge_regions,
                                        post_edge_regions=post_edge_regions,
                                        algorithm=algorithm,
                                        algorithm_parameters=algorithm_parameters)[0:3]
                if not DONE:
                    c0 = (numpy.nonzero(energy >= (ed + pre_edge_regions[0][0]))[0]).min()
                    c1 = (numpy.nonzero(energy <= (ed + post_edge_regions[-1][1]))[-1]).max()
                    DONE = True
                data[:c0, i] = result[c0]
                data[c0:c1, i] = result[c0:c1]
                data[c1:, i] = result[c1]
            data.shape = oldShape
        else:
            raise ValueError("Invalid 1D index %d" % mcaIndex)
        return


MENU_TEXT = "XAS Stack Normalization"
def getStackPluginInstance(stackWindow, **kw):
    ob = XASStackNormalizationPlugin(stackWindow)
    return ob
