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
import StackPluginBase
try:
    from PyMca.PCAWindow import PCAParametersDialog
    from PyMca import StackPluginResultsWindow    
    import PyMca.PyMca_Icons as PyMca_Icons
except ImportError:
    print "PCAStackPlugin importing from somewhere else"
    from PCAWindow import PCAParametersDialog
    import StackPluginResultsWindow
    import PyMca_Icons

DEBUG = 0

class PCAStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {'Calculate':[self.calculate,
                                        "Perform PCA",
                                        None],
                           'Show':[self._showWidget,
                                   "Show last results",
                                   PyMca_Icons.brushselect]}
        self.__methodKeys = ['Calculate', 'Show']
        self.configurationWidget = None
        self.widget = None

    def stackUpdated(self):
        if DEBUG:
            print "StackBrowserPlugin.stackUpdated() called"
        self.configurationWidget = None
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


    #The specific part
    def calculate(self):
        if self.configurationWidget is None:
            self.configurationWidget = PCAParametersDialog(None, regions=True)
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            #I could get some defaults from the stack itslef 
            raise ValueError("Please select an active curve")
            return
        x, spectrum, legend, info = activeCurve
        spectrumLength = max(spectrum.shape)
        oldValue = self.configurationWidget.nPC.value()
        self.configurationWidget.nPC.setMaximum(spectrumLength)
        self.configurationWidget.nPC.setValue(min(oldValue, spectrumLength))
        binningOptions=[1]
        for number in [2, 3, 4, 5, 7, 9, 10, 11, 13, 15, 17, 19]:
            if (spectrumLength % number) == 0:
                binningOptions.append(number)
        ddict = {'options':binningOptions, 'binning': 1, 'method': 0}
        self.configurationWidget.setParameters(ddict)
        y = spectrum
        self.configurationWidget.setSpectrum(x, y)
        ret = self.configurationWidget.exec_()
        if ret:
            self.widget = None
            pcaParameters = self.configurationWidget.getParameters()
            self.configurationWidget.close()
            method = pcaParameters['method']
            methodlabel = pcaParameters.get('methodlabel', "")
            function = pcaParameters['function']
            pcaParameters['ncomponents'] = pcaParameters['npc']
            #At some point I should make sure I get directly the
            #function and the parameters from the configuration widget
            del pcaParameters['npc']
            del pcaParameters['method']
            del pcaParameters['function']
            del pcaParameters['methodlabel']
            binning = pcaParameters['binning']
            mask = pcaParameters['mask']
            stack = self.getStackDataObject()
            if isinstance(stack, numpy.ndarray):
                if stack.data.dtype not in [numpy.float, numpy.float32]:
                    print "WARNING: Non floating point data"

            images, eigenValues, eigenVectors = function(stack,
                                                         **pcaParameters)
            imageNames=None
            vectorNames=None
            if " ICA " in methodlabel:
                nimages = images.shape[0]
                imageNames = []
                vectorNames = []
                itmp = nimages/2
                for i in range(itmp):
                    imageNames.append("ICAimage %02d" % i)
                    vectorNames.append("ICAvector %02d" % i)
                for i in range(itmp):
                    imageNames.append("Eigenimage %02d" % i)
                    vectorNames.append("Eigenvector %02d" % i)
            self.widget = StackPluginResultsWindow.StackPluginResultsWindow(\
                                            usetab=True)
            self.widget.buildAndConnectImageButtonBox()
            qt = StackPluginResultsWindow.qt
            qt.QObject.connect(self.widget,
                               qt.SIGNAL('MaskImageWidgetSignal'),
                               self.mySlot)

            self.widget.setStackPluginResults(images,
                                              spectra=eigenVectors,
                                              image_names=imageNames,
                                              spectra_names=vectorNames)
            self._showWidget()

    
    def _showWidget(self):
        if self.widget is None:
            return
        #Show
        self.widget.show()
        self.widget.raise_()

        #update
        self.selectionMaskUpdated()

MENU_TEXT = "PyMca PCA"
def getStackPluginInstance(stackWindow, **kw):
    ob = PCAStackPlugin(stackWindow)
    return ob
