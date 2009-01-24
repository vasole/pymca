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
import sys
import os
import numpy
try:
    import EdfFile
    EDF = True
except ImportError:
    EDF = False
MATPLOTLIB = MaskImageWidget.MATPLOTLIB
QTVERSION = MaskImageWidget.QTVERSION


class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                                          qt.QSizePolicy.Fixed))

class ExternalImagesWindow(MaskImageWidget.MaskImageWidget):
    def __init__(self, *var, **kw):
        ddict = {}
        ddict['usetab'] = False
        ddict.update(kw)
        ddict['standalonesave'] = False
        if kw.has_key('dynamic'):
            del ddict['dynamic']
        MaskImageWidget.MaskImageWidget.__init__(self, *var, **ddict) 
        self.slider = qt.QSlider(self)
        self.slider.setOrientation(qt.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        
        self.mainLayout.addWidget(self.slider)
        self.connect(self.slider,
                     qt.SIGNAL("valueChanged(int)"),
                     self._showImage)

        self.imageList = None
        self._imageDict = None
        self.imageNames = None
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

        dynamic = kw.get("dynamic", False)
        self._dynamic = dynamic

                    
        self.cropIcon = qt.QIcon(qt.QPixmap(IconDict["crop"]))
        infotext = "Crop image to the currently zoomed window"
        cropPosition = 12
        if kw.has_key('imageicons'):
            if not kw['imageicons']:
                cropPosition = 6
        self.cropButton = self.graphWidget._addToolButton(\
                                        self.cropIcon,
                                        self._cropIconChecked,
                                        infotext,
                                        toggle = False,
                                        position = cropPosition)

        self.flipIcon = qt.QIcon(qt.QPixmap(IconDict["crop"]))

        infotext = "Flip image and data, not the scale."
        self.graphWidget.hFlipToolButton.setToolTip('Flip image')
        self._flipMenu = qt.QMenu()
        self._flipMenu.addAction(qt.QString("Flip Image and Vertical Scale"),
                                 self.__hFlipIconSignal)
        self._flipMenu.addAction(qt.QString("Flip Image Left-Right"),
                                 self._flipLeftRight)
        self._flipMenu.addAction(qt.QString("Flip Image Up-Down"),
                                 self._flipUpDown)        

    def sizeHint(self):
        return qt.QSize(400, 400)

    def _cropIconChecked(self):
        #get current index
        index = self.slider.value()
        #current image
        label = self.imageNames[index]
        qimage = self._imageDict[label]
        width = qimage.width()
        height = qimage.height()

        xmin, xmax = self.graphWidget.graph.getX1AxisLimits()
        ymin, ymax = self.graphWidget.graph.getY1AxisLimits()
        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)
        dummy = xmin
        if (xmin > xmax):
            xmin = xmax
            xmax = dummy
        dummy = ymin
        if (ymin > ymax):
            ymin = ymax
            ymax = dummy
        xmin = max(xmin, 0)
        xmax = min(xmax, width)

        ymin = max(ymin, 0)
        ymax = min(ymax, height)

        rect = qt.QRect(xmin, ymin, xmax-xmin, ymax-ymin)
        newImageList = []
        for i in range(len(self.imageList)):
            image = self._imageDict[self.imageNames[i]].copy(rect)
            newImageList.append(image)

        #replace current image by the new one
        self.setQImageList(newImageList, width, height,
                       clearmask=False,
                       data=None,
                       imagenames=self.imageNames*1)
        
        ###self._imageDict[label] = self.getQImage()            
        ###self.imageList.append(self.getImageData())
        self._showImage(index)

    def _flipIconChecked(self):
        if not self.graphWidget.graph.yAutoScale:
            qt.QMessageBox.information(self, "Open",
                    "Please set Y Axis to AutoScale first")
            return
        if not self.graphWidget.graph.xAutoScale:
            qt.QMessageBox.information(self, "Open",
                    "Please set X Axis to AutoScale first")
            return
        if self.imageList is None:
            return
        if self._imageDict is None:
            return
        if self.imageNames is None:
            return
        self._flipMenu.exec_(self.cursor().pos())

    def _hFlipIconSignal(self):
        if not self.graphWidget.graph.yAutoScale:
            qt.QMessageBox.information(self, "Open",
                    "Please set Y Axis to AutoScale first")
            return
        if not self.graphWidget.graph.xAutoScale:
            qt.QMessageBox.information(self, "Open",
                    "Please set X Axis to AutoScale first")
            return
        if self.getQImage is None:
            return
        if self.imageNames is None:
            #just setImage data used
            #I use the default flip
            self.__hFlipIconSignal()
            return
        if self.imageList is None:
            return
        if self._imageDict is None:
            return
        self._flipMenu.exec_(self.cursor().pos())

    def __hFlipIconSignal(self):
        MaskImageWidget.MaskImageWidget._hFlipIconSignal(self)

    def _flipUpDown(self):
        for i in range(len(self.imageList)):
            label = self.imageNames[i]
            self._imageDict[label] = self._imageDict[label].mirrored(0, 1)
            self.imageList[i] = numpy.flipud(self.getImageData())
        self.showImage(self.slider.value())

    def _flipLeftRight(self):
        for i in range(len(self.imageList)):
            label = self.imageNames[i]
            self._imageDict[label] = self._imageDict[label].mirrored(1, 0)
            self.imageList[i] = numpy.fliplr(self.getImageData())
        self.showImage(self.slider.value())

    def _showImage(self, index):
        if len(self.imageList):
            self.showImage(index, moveslider=False)
            
    def showImage(self, index=0, moveslider=True):
        if self.imageList is None:
            return
        if len(self.imageList) == 0:
            return
        if self._dynamic:
            self._dynamicAction(index)
        else:
            qimage = self._imageDict[self.imageNames[index]]
            data = self.imageList[index]
            self.setQImage(qimage,
                       qimage.width(),
                       qimage.height(),
                       clearmask=False,
                       data=self.imageList[index])
            self.graphWidget.graph.setTitle(self.imageNames[index])
        if moveslider:
            self.slider.setValue(index)

    def _dynamicAction(self, index):
        #just support edffiles
        fileName = self.imageList[index]
        edf = EdfFile.EdfFile(fileName)
        self.setImageData(edf.GetData(0))
        self.graphWidget.graph.setTitle(os.path.basename(fileName))

    def setQImageList(self, images, width, height,
                      clearmask = False, data=None, imagenames = None):
        nimages = len(images)
        if imagenames is None:
            self.imageNames = []
            for i in range(nimages):
                self.imageNames.append("ExternalImage %02d" % i)
        else:
            self.imageNames = imagenames
                
        i = 0
        self._imageDict = {}
        self.imageList = []
        for label in self.imageNames:
            self.setQImage(images[i], width, height,
                           clearmask=clearmask,
                           data=data)
            self._imageDict[label] = self.getQImage()            
            self.imageList.append(self.getImageData())
            i += 1

        if self.imageList is not None:
            self.slider.setMaximum(len(self.imageList)-1)
            self.showImage(0)
        else:
            self.slider.setMaximum(0)

        self.slider.setValue(0)


    def saveImageList(self, filename=None, imagelist=None, labels=None):
        if self.imageList is None:
            return
        if self._dynamic:
            #save only one image
            MaskImageWidget.MaskImageWidget.saveImageList(self)
            return
        labels = []
        for i in range(len(self.imageList)):
            labels.append(self.imageNames[i].replace(" ","_"))
        return MaskImageWidget.MaskImageWidget.saveImageList(self,
                                          imagelist=self.imageList,
                                          labels=labels)

    def setImageList(self, imagelist):
        self.imageList = imagelist
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
    if len(sys.argv) > 1:
        if sys.argv[1][-3:].upper() == 'EDF':
            container = ExternalImagesWindow(selection=False,
                                             colormap=True,
                                             imageicons=False,
                                             standalonesave=True,
                                             dynamic=True)
            container.setImageList(sys.argv[1:])
        else:
            container = ExternalImagesWindow()
            image = qt.QImage(sys.argv[1])
            #container.setQImage(image, image.width(),image.height())
            container.setQImageList([image], 200, 100)
    else:
        container = ExternalImagesWindow()
        data = numpy.arange(10000)
        data.shape = 100, 100
        container.setImageData(data)
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
        if not container._dynamic:
            qt.QObject.connect(container,
                           qt.SIGNAL("MaskImageWidgetSignal"),
                           theSlot)
        app.exec_()

if __name__ == "__main__":
    import numpy
    test()
        
