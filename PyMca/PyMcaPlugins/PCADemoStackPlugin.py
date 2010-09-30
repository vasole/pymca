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
import numpy.linalg
try:
    import numpy.core._dotblas as dotblas
except ImportError:
    print "WARNING: Not using BLAS"
    dotblas = numpy
import StackPluginBase
import weakref
try:
    from PyMca import StackPluginResultsWindow
    import PyMca.PyMca_Icons as PyMca_Icons
except ImportError:
    import StackPluginResultsWindow
    import PyMca_Icons

qt = StackPluginResultsWindow.qt
HorizontalSpacer = qt.HorizontalSpacer

DEBUG = 0
class PCADemoParametersDialog(qt.QDialog):
    def __init__(self, parent = None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("PCA Configuration Dialog")
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)

        hbox1 = qt.QWidget(self)
        hbox1.mainLayout = qt.QHBoxLayout(hbox1)
        hbox1.mainLayout.setMargin(0)
        hbox1.mainLayout.setSpacing(2)
        labelPC = qt.QLabel(hbox1)
        labelPC.setText("Number of PC:")
        self.nPC = qt.QSpinBox(hbox1)
        self.nPC.setMinimum(1)
        self.nPC.setValue(10)
        self.nPC.setMaximum(100)
        hbox1.mainLayout.addWidget(labelPC)
        hbox1.mainLayout.addWidget(self.nPC)
        self.mainLayout.addWidget(hbox1)

        hbox = qt.QWidget(self)
        hbox.mainLayout = qt.QHBoxLayout(hbox)
        hbox.mainLayout.setMargin(0)
        hbox.mainLayout.setSpacing(2)
        self.okButton = qt.QPushButton(hbox)
        self.okButton.setText("Accept")
        self.okButton.setAutoDefault(False)
        self.dismissButton = qt.QPushButton(hbox)
        self.dismissButton.setText("Dismiss")
        self.dismissButton.setAutoDefault(False)
        hbox.mainLayout.addWidget(self.okButton)
        hbox.mainLayout.addWidget(HorizontalSpacer(hbox))
        hbox.mainLayout.addWidget(self.dismissButton)
        self.mainLayout.addWidget(hbox)

        self.connect(self.okButton,
                     qt.SIGNAL("clicked()"),
                     self.accept)
        
        self.connect(self.dismissButton,
                     qt.SIGNAL("clicked()"),
                     self.reject)
        
    def getNumberOfComponents(self):
        return self.nPC.value()

    def setMaximumNumberOfComponents(self, n):
        current = self.nPC.value()
        if n < current:
            self.nPC.setValue(n)
        self.nPC.setMaximum(n)

class PCADemoStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {'Calculate':[self.calculatePCA,
                                        "Perform PCA",
                                        None],
                           'Show':[self.showResults,
                                   "Show last results",
                                   PyMca_Icons.brushselect]}
        self.results = None
        self.dialogWidget = None
        self.resultsWidget = None
        
    def stackUpdated(self):
        self.dialogWidget = None
        self.resultsWidget = None
        self.results = None

    def selectionMaskUpdated(self):
        if self.resultsWidget is not None:
            mask = self.getStackSelectionMask()
            self.resultsWidget.setSelectionMask(mask)

    #Methods implemented by the plugin
    def getMethods(self):
        return ['Calculate', 'Show']

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return apply(self.methodDict[name][0])

    def calculatePCA(self):
        if self.dialogWidget is None:
            self.dialogWidget = PCADemoParametersDialog()
        stackDataObject = self.getStackDataObject()
        maxNPC = stackDataObject.data.shape[-1]
        self.dialogWidget.setMaximumNumberOfComponents(maxNPC)
        ret = self.dialogWidget.exec_()
        if ret:
            nPC = self.dialogWidget.getNumberOfComponents()
            images, eigenvalues, eigenvectors = self._calculatePCA(nPC)
            if self.resultsWidget is None:
                self.resultsWidget = StackPluginResultsWindow.StackPluginResultsWindow(usetab=True)
                self.resultsWidget.buildAndConnectImageButtonBox()
                qt.QObject.connect(self.resultsWidget,
                                   qt.SIGNAL('MaskImageWidgetSignal'),
                                   self.mySlot)
            self.resultsWidget.show()
            imageNames = []
            spectraNames = []
            for i in xrange(nPC):
                if nPC < 100:
                    imageNames.append("Eigenimage %02d" % i)
                    spectraNames.append("Eigenvector %02d" % i)
                elif nPC < 1000:
                    imageNames.append("Eigenimage %03d" % i)
                    spectraNames.append("Eigenvector %03d" % i)
                else:
                    imageNames.append("Eigenimage %d" % i)
                    spectraNames.append("Eigenvector %d" % i)
            self.resultsWidget.setStackPluginResults(images,
                                                     spectra=eigenvectors,
                                                     image_names=imageNames,
                                                     spectra_names=spectraNames)
            self.showResults()

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

    def _calculatePCA(self, ncomponents):
        """
        Very primitive implementation
        This is a covariance method using numpy numpy.linalg.eigh
        """
        stackDataObject = self.getStackDataObject()
        data = stackDataObject.data
        info = stackDataObject.info

        if not isinstance(data, numpy.ndarray):
            raise TypeError, "This Plugin only supports numpy arrays"

        print info.keys()        
            
        if len(data.shape) == 3:
            r, c, N = data.shape
            data.shape = r*c, N
        else:
            r, N = data.shape
            c = 1

        if ncomponents > N:
            raise ValueError, "Number of components too high."
        #end of common part

        #begin the specific coding
        avg = numpy.sum(data, 0)/(1.0*r*c)
        numpy.subtract(data, avg, data)
        cov = dotblas.dot(data.T, data)
        evalues, evectors = numpy.linalg.eigh(cov)
        cov = None
        images = numpy.zeros((ncomponents, r * c), data.dtype)
        eigenvectors = numpy.zeros((ncomponents, N), data.dtype)
        eigenvalues = numpy.zeros((ncomponents,), data.dtype)
        #sort eigenvalues
        a = [(evalues[i], i) for i in range(len(evalues))]
        a.sort()
        a.reverse()
        for i0 in range(ncomponents):
            i = a[i0][1]
            eigenvalues[i0] = evalues[i]
            eigenvectors[i0,:] = evectors[:,i]
            images[i0,:] = dotblas.dot(data , eigenvectors[i0,:])
        
        #restore the original data
        numpy.add(data, avg, data)
        data.shape = r, c, N

        #reshape the images
        images.shape = ncomponents, r, c
        return images, eigenvalues, eigenvectors

    def showResults(self):
        if self.resultsWidget is None:
            return
        mask = self.getStackSelectionMask()
        self.resultsWidget.setSelectionMask(mask)
        self.resultsWidget.show()
        self.resultsWidget.raise_()

MENU_TEXT = "Simple PCA Demo"
def getStackPluginInstance(stackWindow, **kw):
    ob = PCADemoStackPlugin(stackWindow)
    return ob
