#/*##########################################################################
# Copyright (C) 2004-2009 European Synchrotron Radiation Facility
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
# is a problem for you.
#############################################################################*/
__author__ = "V.A. Sole - ESRF BLISS Group"
import PyMcaQt as qt
from PyMca_Icons import IconDict
import MaskImageWidget
import ScanWindow
import sys
MATPLOTLIB = MaskImageWidget.MATPLOTLIB
QTVERSION = MaskImageWidget.QTVERSION


class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                                          qt.QSizePolicy.Fixed))
class PCAParametersDialog(qt.QDialog):
    def __init__(self, parent = None, options=[1, 2, 3, 4, 5, 10], mdp = False):
        qt.QDialog.__init__(self, parent)
        if QTVERSION < '4.0.0':
            self.setCaption("PCA Configuration Dialog")
        else:
            self.setWindowTitle("PCA Configuration Dialog")
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(11)
        self.mainLayout.setSpacing(0)

        #
        self.methodOptions = qt.QGroupBox(self)
        self.methodOptions.setTitle('PCA Module to use')
        self.methods = ['Correct', 'Fast']
        self.methodOptions.mainLayout = qt.QGridLayout(self.methodOptions)
        self.methodOptions.mainLayout.setMargin(0)
        self.methodOptions.mainLayout.setSpacing(2)
        if mdp:
            #self.methods.append("MDP (PCA + ICA)")
            self.methods.append("MDP")
            
        self.buttonGroup = qt.QButtonGroup(self.methodOptions)
        i = 0
        for item in self.methods:
            rButton = qt.QRadioButton(self.methodOptions)
            self.methodOptions.mainLayout.addWidget(rButton, 0, i)
            #self.l.setAlignment(rButton, qt.Qt.AlignHCenter)
            if i == 1:
                rButton.setChecked(True)
            rButton.setText(item)
            self.buttonGroup.addButton(rButton)
            self.buttonGroup.setId(rButton, i)
            i += 1
        self.connect(self.buttonGroup,
                     qt.SIGNAL('buttonPressed(QAbstractButton *)'),
                     self._slot)

        self.mainLayout.addWidget(self.methodOptions)

        #built in speed options
        self.speedOptions = qt.QGroupBox(self)
        self.speedOptions.setTitle("Speed Options")
        self.speedOptions.mainLayout = qt.QGridLayout(self.speedOptions)
        self.speedOptions.mainLayout.setMargin(0)
        self.speedOptions.mainLayout.setSpacing(2)
        labelPC = qt.QLabel(self)
        labelPC.setText("Number of PC:")
        self.nPC = qt.QSpinBox(self.speedOptions)
        self.nPC.setMinimum(0)
        self.nPC.setValue(10)
        self.nPC.setMaximum(40)

        self.binningLabel = qt.QLabel(self.speedOptions)
        self.binningLabel.setText("Spectral Binning:")
        self.binningCombo = qt.QComboBox(self.speedOptions)
        for option in options:
            self.binningCombo.addItem("%d" % option)
        self.speedOptions.mainLayout.addWidget(labelPC, 0, 0)
        self.speedOptions.mainLayout.addWidget(self.nPC, 0, 1)
        #self.speedOptions.mainLayout.addWidget(HorizontalSpacer(self), 0, 2)
        self.speedOptions.mainLayout.addWidget(self.binningLabel, 1, 0)
        self.speedOptions.mainLayout.addWidget(self.binningCombo, 1, 1)
        self.binningCombo.setEnabled(False)
        

        #the OK button
        hbox = qt.QWidget(self)
        hboxLayout = qt.QHBoxLayout(hbox)
        hboxLayout.setMargin(0)
        hboxLayout.setSpacing(0)
        self.okButton = qt.QPushButton(hbox)
        self.okButton.setText("Accept")
        hboxLayout.addWidget(HorizontalSpacer(hbox))
        hboxLayout.addWidget(self.okButton)
        hboxLayout.addWidget(HorizontalSpacer(hbox))
        self.mainLayout.addWidget(self.speedOptions)
        self.mainLayout.addWidget(hbox)
        self.connect(self.okButton,
                     qt.SIGNAL("clicked()"),
                     self.accept)

    def _slot(self, button):
        button.setChecked(True)
        index = self.buttonGroup.checkedId()
        if index == 0:
            self.binningLabel.setText("Spectral Binning:")
            self.binningCombo.setEnabled(True)
        elif index == 2:
            self.binningLabel.setText("Spectral Binning:")
            self.binningCombo.setEnabled(True)
        else:
            self.binningCombo.setEnabled(False)

    def setParameters(self, ddict):
        if ddict.has_key('options'):
            self.binningCombo.clear()
            for option in ddict['options']:
                self.binningCombo.addItem("%d" % option)
        if ddict.has_key('binning'):
            option = "%d" % ddict['binning']
            for i in range(self.binningCombo.count()):
                if str(self.binningCombo.itemText(i)) == option:
                    self.binningCombo.setCurrentIndex(i)
        if ddict.has_key('npc'):
            self.nPC.setValue(ddict['npc'])
        if ddict.has_key('method'):
            self.buttonGroup.buttons()[ddict['method']].setChecked(True)
            if ddict['method'] in [0, 2]:
                self.binningCombo.setEnabled(True)
            else:
                self.binningCombo.setEnabled(False)                
        return

    def getParameters(self):
        ddict = {}
        ddict['binning'] = int(self.binningCombo.currentText())
        ddict['npc'] = self.nPC.value()
        i = self.buttonGroup.checkedId()
        ddict['method'] = i
        ddict['methodlabel'] = self.methods[i]
        return ddict

class PCAWindow(MaskImageWidget.MaskImageWidget):
    def __init__(self, *var, **kw):
        ddict = {}
        ddict['usetab'] = True
        ddict.update(kw)
        ddict['standalonesave'] = False
        MaskImageWidget.MaskImageWidget.__init__(self, *var, **ddict) 
        self.slider = qt.QSlider(self)
        self.slider.setOrientation(qt.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)

        # The 1D graph
        self.vectorGraph = ScanWindow.ScanWindow(self)
        self.mainTab.addTab(self.vectorGraph, "VECTORS")
        
        self.mainLayout.addWidget(self.slider)
        self.connect(self.slider,
                     qt.SIGNAL("valueChanged(int)"),
                     self._showImage)

        self.imageList = None
        self.eigenValues = None
        self.eigenVectors = None
        self.eigenNames = None
        standalonesave = kw.get("standalonesave", True)
        if standalonesave:
            self.connect(self.graphWidget.saveToolButton,
                         qt.SIGNAL("clicked()"), 
                         self._saveToolButtonSignal)
            self._saveMenu = qt.QMenu()
            self._saveMenu.addAction(qt.QString("Image Data"),
                                     self.saveImageList)
            self._saveMenu.addAction(qt.QString("Standard Graphics"),
                                     self.graphWidget._saveIconSignal)
            if QTVERSION > '4.0.0':
                if MATPLOTLIB:
                    self._saveMenu.addAction(qt.QString("Matplotlib") ,
                                     self._saveMatplotlibImage)
        self.multiplyIcon = qt.QIcon(qt.QPixmap(IconDict["swapsign"]))
        infotext = "Multiply image by -1"
        self.multiplyButton = self.graphWidget._addToolButton(\
                                        self.multiplyIcon,
                                        self._multiplyIconChecked,
                                        infotext,
                                        toggle = False,
                                        position = 12)

    def sizeHint(self):
        return qt.QSize(400, 400)

    def _multiplyIconChecked(self):
        if self.imageList is None:
            return
        index = self.slider.value()
        self.imageList[index] *= -1
        if self.eigenVectors is not None:
            self.eigenVectors[index] *= -1

        self._showImage(index)

    def _showImage(self, index):
        if len(self.imageList):
            self.showImage(index, moveslider=False)
        if self.eigenVectors is not None:
            legend = self.vectorNames[index]
            y = self.eigenVectors[index]
            self.vectorGraph.newCurve(range(len(y)), y, legend, replace=True) 
            
    def showImage(self, index=0, moveslider=True):
        if self.imageList is None:
            return
        if len(self.imageList) == 0:
            return
        self.setImageData(self.imageList[index])
        self.graphWidget.graph.setTitle(self.imageNames[index])
        if moveslider:
            self.slider.setValue(index)

    def setPCAData(self, images, eigenvalues=None, eigenvectors=None,
                   imagenames = None, vectornames = None):
        self.eigenValues = eigenvalues
        self.eigenVectors = eigenvectors
        if type(images) == type([]):
            self.imageList = images
        elif len(images.shape) == 3:
            nimages = images.shape[0]
            self.imageList = [0] * nimages
            for i in range(nimages):
                self.imageList[i] = images[i,:]
                if self.imageList[i].max() < 0:
                    self.imageList[i] *= -1
                    if self.eigenVectors is not None:
                        self.eigenVectors [i] *= -1
            if imagenames is None:
                self.imageNames = []
                for i in range(nimages):
                    self.imageNames.append("Eigenimage %02d" % i)
            else:
                self.imageNames = imagenames
                
        if self.imageList is not None:
            self.slider.setMaximum(len(self.imageList)-1)
            self.showImage(0)
        else:
            self.slider.setMaximum(0)

        if self.eigenVectors is not None:
            if vectornames is None:
                self.vectorNames = []
                for i in range(nimages):
                    self.vectorNames.append("Component %02d" % i)
            else:
                self.vectorNames = vectornames
            legend = self.vectorNames[0]
            y = self.eigenVectors[0]
            self.vectorGraph.newCurve(range(len(y)), y, legend, replace=True) 
            
        self.slider.setValue(0)


    def saveImageList(self, filename=None, imagelist=None, labels=None):
        if self.imageList is None:
            return
        labels = []
        for i in range(len(self.imageList)):
            labels.append(self.imageNames[i].replace(" ","_"))
        return MaskImageWidget.MaskImageWidget.saveImageList(self,
                                                             imagelist=self.imageList,
                                                             labels=labels)

    def setImageList(self, imagelist):
        self.imageList = imagelist
        self.eigenValues = None
        self.eigenVectors = None
        if imagelist is not None:
            self.slider.setMaximum(len(self.imageList)-1)
            self.showImage(0)
            

def test2():
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit()'))

    dialog = PCAParametersDialog()
    dialog.setParameters({'options':[1,3,5,7,9],'method':1, 'npc':8,'binning':3})
    dialog.setModal(True)
    ret = dialog.exec_()
    if ret:
        dialog.close()
        print dialog.getParameters()
    #app.exec_()

def test():
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit()'))

    container = PCAWindow()
    data = numpy.arange(20000)
    data.shape = 2, 100, 100
    data[1, 0:100,0:50] = 100
    container.setPCAData(data, eigenvectors=[numpy.arange(100.), numpy.arange(100.)+10],
                                imagenames=["I1", "I2"], vectornames=["V1", "V2"])
    container.show()
    def theSlot(ddict):
        print ddict['event']

    if QTVERSION < '4.0.0':
        qt.QObject.connect(container,
                       qt.PYSIGNAL("MaskImageWidgetSignal"),
                       updateMask)
        app.setMainWidget(container)
        app.exec_loop()
    else:
        qt.QObject.connect(container,
                           qt.SIGNAL("MaskImageWidgetSignal"),
                           theSlot)
        app.exec_()

if __name__ == "__main__":
    import numpy
    test()
        
