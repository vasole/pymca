#/*##########################################################################
# Copyright (C) 2004-2006 European Synchrotron Radiation Facility
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
# is a problem to you.
#############################################################################*/
__author__ = "V.A. Sole - ESRF BLISS Group"
import sys
import os
import RGBCorrelatorSlider
import RGBCorrelatorTable
import Numeric
import spslut
qt = RGBCorrelatorSlider.qt

QTVERSION = qt.qVersion()
DEBUG = 0
    
class RGBCorrelatorWidget(qt.QWidget):
    def __init__(self, parent = None, bgrx = True):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(11)
        self.mainLayout.setSpacing(6)
        self.labelWidget = qt.QWidget(self)
        self.labelWidget.mainLayout = qt.QGridLayout(self.labelWidget)
        self.labelWidget.mainLayout.setMargin(0)
        self.labelWidget.mainLayout.setSpacing(0)
        alignment = qt.Qt.AlignVCenter | qt.Qt.AlignCenter
        label1 = MyQLabel(self.labelWidget, color = qt.Qt.black)
        label1.setAlignment(alignment)
        label1.setText("Image Size")
        self.__sizeLabel = MyQLabel(self.labelWidget,
                                    bold = True,
                                    color = qt.Qt.red)
        self.__sizeLabel.setAlignment(alignment)
        self.__sizeLabel.setText("No image set")

        #self.__rowLineEdit = qt.QLineEdit(self.labelWidget)
        #self.__columnLineEdit = qt.QLineEdit(self.labelWidget)
        self.__imageResizeButton = qt.QPushButton(self.labelWidget)
        self.__imageResizeButton.setText("Resize")

        self.labelWidget.mainLayout.addWidget(label1, 0, 0)
        self.labelWidget.mainLayout.addWidget(self.__sizeLabel, 0, 1)
        
        #self.labelWidget.mainLayout.addWidget(self.__rowLineEdit, 1, 0)
        #self.labelWidget.mainLayout.addWidget(self.__columnLineEdit, 1, 1)
        self.labelWidget.mainLayout.addWidget(self.__imageResizeButton, 0, 2)
        
        self.sliderWidget = RGBCorrelatorSlider.RGBCorrelatorSlider(self,
                                        autoscalelimits=[5.0, 80.0])
        self.tableWidget  = RGBCorrelatorTable.RGBCorrelatorTable(self)

        self.mainLayout.addWidget(self.labelWidget)
        self.mainLayout.addWidget(self.sliderWidget)
        self.mainLayout.addWidget(self.tableWidget)

        if bgrx:
            self.bgrx = "BGRX"
        else:
            self.bgrx = "RGBX"
        self._imageList = []
        self._imageDict = {}
        self.__imageLength = None
        self.__redLabel = None
        self.__greenLabel = None
        self.__blueLabel = None 
        self.__redImageData   = None
        self.__greenImageData = None
        self.__blueImageData  = None
        self.__redMin     = 0.0
        self.__redMax     = 100.0
        self.__greenMin   = 0.0
        self.__greenMax   = 100.0
        self.__blueMin   = 0.0
        self.__blueMax     = 0.0
        self.__redImage = None
        self.__greenImage = None
        self.__blueImage = None
        
        self.connect(self.__imageResizeButton,
                     qt.SIGNAL("clicked()"),
                     self._imageResizeSlot)
        self.connect(self.sliderWidget,
                     qt.SIGNAL("RGBCorrelatorSliderSignal"),
                     self._sliderSlot)

        self.connect(self.tableWidget,
                     qt.SIGNAL("RGBCorrelatorTableSignal"),
                     self._tableSlot)
        
    def _sliderSlot(self, ddict):
        if DEBUG: print "RGBCorrelatorWidget._sliderSlot()"
        if self.__imageLength is None: return
        tableDict = self.tableWidget.getElementSelection()
        if ddict['event'] == 'redChanged':
            self.__redMin = ddict['min']
            self.__redMax = ddict['max']
            if len(tableDict['r']):
                self.__recolor(['r'])
        elif ddict['event'] == 'greenChanged':
            self.__greenMin = ddict['min']
            self.__greenMax = ddict['max']
            if len(tableDict['g']):
                self.__recolor(['g'])
        elif ddict['event'] == 'blueChanged':
            self.__blueMin = ddict['min']
            self.__blueMax = ddict['max']
            if len(tableDict['b']):
                self.__recolor(['b'])
        elif ddict['event'] == 'allChanged':
            self.__redMin = ddict['red'][0]
            self.__redMax = ddict['red'][1]
            self.__greenMin = ddict['green'][0]
            self.__greenMax = ddict['green'][1]
            self.__blueMin = ddict['blue'][0]
            self.__blueMax = ddict['blue'][1]
            if not len(tableDict['r']):
                if not len(tableDict['g']):
                    if not len(tableDict['b']):
                        return
            self.__recolor(['r', 'g', 'b'])
            
    def _tableSlot(self, ddict):
        if DEBUG: print "RGBCorrelatorWidget._tableSlot()"
        if self.__imageLength is None: return
        if ddict['r'] == []:ddict['r'] = None
        if ddict['g'] == []:ddict['g'] = None
        if ddict['b'] == []:ddict['b'] = None

        if ddict['r'] is None:
            self.__redImageData = Numeric.zeros(self.__imageShape).astype(Numeric.Float)
            self.__redLabel = None
        else:
            self.__redLabel = ddict['elementlist'][ddict['r'][0]]
            self.__redImageData = self._imageDict[self.__redLabel]['image']

        if ddict['g'] is None:
            self.__greenImageData = Numeric.zeros(self.__imageShape).astype(Numeric.Float)
            self.__greenLabel = None
        else:
            self.__greenLabel = ddict['elementlist'][ddict['g'][0]]
            self.__greenImageData = self._imageDict[self.__greenLabel]['image']

        if ddict['b'] is None:
            self.__blueImageData = Numeric.zeros(self.__imageShape).astype(Numeric.Float)
            self.__blueLabel = None
        else:
            self.__blueLabel = ddict['elementlist'][ddict['b'][0]]
            self.__blueImageData = self._imageDict[self.__blueLabel]['image']
        self.__recolor(['r', 'g', 'b'])        

    def __recolor(self, color = None):
        if color is None:colorlist = ['r', 'g', 'b']
        elif type(color) == type("") : colorlist = [color]
        else:colorlist = color * 1
        ddict = {}
        ddict['event'] = 'updated'
        if 'r' in colorlist:
            #get slider
            label = self.__redLabel 
            if label is None:
                valmin = 0.0
                valmax = 1.0
            else:
                valmin = self._imageDict[label]['min'] 
                valmax = self._imageDict[label]['max']
                delta  = 0.01 * (valmax  - valmin)
                valmin = valmin + delta * self.__redMin
                valmax = valmin + delta * self.__redMax
            red, size, minmax = self.getColorImage(self.__redImageData,
                                     spslut.RED,
                                     valmin,
                                     valmax)
            self.__redImage = Numeric.array(red).astype(Numeric.UInt8)
            ddict['red'] = red
            ddict['size']= size
        if 'g' in colorlist:
            #get slider
            label = self.__greenLabel 
            if label is None:
                valmin = 0.0
                valmax = 1.0
            else:
                valmin = self._imageDict[label]['min'] 
                valmax = self._imageDict[label]['max']
                delta  = 0.01 * (valmax  - valmin)
                valmin = valmin + delta * self.__greenMin
                valmax = valmin + delta * self.__greenMax
            green, size, minmax = self.getColorImage(self.__greenImageData,
                                     spslut.GREEN,
                                     valmin,
                                     valmax)
            self.__greenImage = Numeric.array(green).astype(Numeric.UInt8)
            ddict['green'] = green
            ddict['size']= size

        if 'b' in colorlist:
            #get slider
            label = self.__blueLabel 
            if label is None:
                valmin = 0.0
                valmax = 1.0
            else:
                valmin = self._imageDict[label]['min'] 
                valmax = self._imageDict[label]['max']
                #if valmax == valmin:valmax = valmin + 1
                delta  = 0.01 * (valmax  - valmin)
                valmin = valmin + delta * self.__blueMin
                valmax = valmin + delta * self.__blueMax
            blue, size, minmax = self.getColorImage(self.__blueImageData,
                                     spslut.BLUE,
                                     valmin,
                                     valmax)
            self.__blueImage = Numeric.array(blue).astype(Numeric.UInt8)
            ddict['blue'] = blue
            ddict['size'] = size
        image = self.__redImage + self.__greenImage + self.__blueImage
        #ddict['image'] = image.tostring()
        ddict['image'] = image
        self.emit(qt.SIGNAL("RGBCorrelatorWidgetSignal"), ddict)

    def getColorImage(self, image, colormap, datamin=None, datamax=None):
        COLORMAPLIST = [spslut.GREYSCALE, spslut.REVERSEGREY, spslut.TEMP,
                        spslut.RED, spslut.GREEN, spslut.BLUE, spslut.MANY]
        if colormap not in COLORMAPLIST:
            raise "ValueError", "Unknown color scheme %s" % colormap

        if (datamin is None) or (datamax is None):
            #spslut already calculates min and max
            #tmp = Numeric.ravel(image)
            (image_buffer, size, minmax)= spslut.transform(image,
                                     (1,0),
                                     (spslut.LINEAR,3.0),
                                      self.bgrx, colormap,
                                      1,
                                      (0,1))
                                     #(min(tmp),max(tmp)))
        else:
            (image_buffer, size, minmax)= spslut.transform(image,
                                     (1,0),
                                     (spslut.LINEAR,3.0),
                                      self.bgrx, colormap,
                                      0,
                                     (datamin, datamax))

        return image_buffer, size, minmax

    def addImage(self, image0, label = None):
        image = Numeric.array(image0).astype(Numeric.Float)
        if label is None:
            label = "Unnamed 00"
            i = 0
            while(label in self._imageList):
                i += 1
                label = "Unnamed %02d" % i
        if not len(image): return
        firstTime = False
        if self.__imageLength is None:
            if not len(image.shape): return
            self.__imageLength = 1
            for value in image.shape:
                self.__imageLength *= value
            if len(image.shape) == 1:
                image = Numeric.resize(image, (image.shape[0], 1))
            self.__imageShape = image.shape
            self._updateSizeLabel()
            firstTime = True

        if image.shape != self.__imageShape:
            length = 1
            for value in image.shape:
                length *= value
            if length == self.__imageLength:
                image = Numeric.resize(image,
                         (self.__imageShape[0], self.__imageShape[1]))
            else:
                raise "ValueError", "Image cannot be reshaped to %d x %d" % \
                          (self.__imageShape[0], self.__imageShape[1])  

        if label not in self._imageList:
            self._imageList.append(label)
        self._imageDict[label] = {}
        self._imageDict[label]['image'] = image
        tmp = Numeric.ravel(image)
        self._imageDict[label]['min'] = min(tmp)
        self._imageDict[label]['max'] = max(tmp)

        self.tableWidget.build(self._imageList)
        i = 0
        for label in self._imageList:
            mintext = "%g" % self._imageDict[label]['min']
            maxtext = "%g" % self._imageDict[label]['max']
            item = self.tableWidget.item(i, 4)
            if item is None:
                item = qt.QTableWidgetItem(mintext,
                                       qt.QTableWidgetItem.Type)
                item.setTextAlignment(qt.Qt.AlignHCenter | qt.Qt.AlignVCenter)
                item.setFlags(qt.Qt.ItemIsEnabled)
                self.tableWidget.setItem(i, 4, item)
            else:
                item.setText(mintext)
            item = self.tableWidget.item(i, 5)
            if item is None:
                item = qt.QTableWidgetItem(maxtext,
                                       qt.QTableWidgetItem.Type)
                item.setTextAlignment(qt.Qt.AlignHCenter | qt.Qt.AlignVCenter)
                item.setFlags(qt.Qt.ItemIsEnabled)
                self.tableWidget.setItem(i, 5, item)
            else:
                item.setText(maxtext)
            i += 1
        if firstTime:
            self.tableWidget.setElementSelection({'r':[0]})
                                                 #, 'g':[0],'b':[0]})
            self.sliderWidget.autoScaleFromAToB()
            #self.__recolor()
            #self.tableWidget._update()

    def removeImage(self, label):
        if label not in self._imageList:return
        self._imageDict[label] = {}
        del self._imageDict[label]
        del self._imageList[self._imageList.index(label)]
        if self.__redLabel == label:   self.__redLabel = None
        if self.__greenLabel == label: self.__greenLabel = None
        if self.__blueLabel == label:self.__blueLabel = None 

        self.tableWidget.build(self._imageList)
        self.tableWidget._update()

    def _imageResizeSlot(self):
        if self.__imageLength is None: return
        dialog = ImageShapeDialog(self, shape = self.__imageShape)
        dialog.setModal(True)
        ret = dialog.exec_()
        if ret:
            shape = dialog.getImageShape()
            dialog.close()
            del dialog
            try:
                if (shape[0]*shape[1]) <= 0:
                    self.reset()
                else:
                    self.setImageShape(shape)
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error reshaping: %s" % sys.exc_info()[1])
                msg.exec_()


    def setImageShape(self, shape):
        if self.__imageLength is None: return
        length = 1
        for value in shape:
            length *= value
        if length != self.__imageLength:
            raise "ValueError","New length %d different of old length %d" % \
                    (length, self.__imageLength)
        self.__imageShape = shape
        self._updateSizeLabel()
        for key in self._imageDict.keys():
            self._imageDict[key]['image'].shape = shape
        self.tableWidget._update()

    def _updateSizeLabel(self):
        if self.__imageLength is None:
            self.__sizeLabel.setText("No image set")
            return
        text = ""
        n = len(self.__imageShape)
        for i in range(n):
            value = self.__imageShape[i]
            if i == (n-1):
                text += " %d" % value
            else:
                text += " %d x" % value
        self.__sizeLabel.setText(text)
        

    def reset(self):
        #ask the possible graph client to delete the image
        self._tableSlot({'r':[],'g':[],'b':[]})
        self._imageList = []
        self._imageDict = {}
        self.__imageLength = None
        self.__imageShape = None
        self.__redLabel    = None
        self.__greenLabel  = None
        self.__blueLabel   = None 
        self.__redImageData   = None
        self.__greenImageData = None
        self.__blueImageData  = None
        self.__redMin     = 0.0
        self.__redMax     = 100.0
        self.__greenMin   = 0.0
        self.__greenMax   = 100.0
        self.__blueMin   = 0.0
        self.__blueMax     = 0.0
        self.__redImage = None
        self.__greenImage = None
        self.__blueImage = None
        self._updateSizeLabel()
        self.tableWidget.setRowCount(0)

    def update(self):
        self.__recolor()

    def tiffExport(self, filename = "test.tif"):
        import Image
        Image.preinit()
        image = self.__redImage + self.__greenImage + self.__blueImage
        width  = self.__imageShape[0]
        height = self.__imageShape[1] 
        pilImage = Image.fromstring("RGBX",(width,height),image)
        if os.path.exists(filename):
            os.remove(filename)
        pilImage.save(filename) 

        

class ImageShapeDialog(qt.QDialog):
    def __init__(self, parent = None, shape = None):
        qt.QDialog.__init__(self, parent)
        self.mainLayout = qt.QGridLayout(self)
        label1 = MyQLabel(self, bold = False, color= qt.Qt.black)
        label1.setText("Number of rows    = ")
        self.rows = qt.QLineEdit(self)
        if shape is not None:self.rows.setText("%g" % shape[0])
        self.columns = qt.QLineEdit(self)
        if shape is not None:self.columns.setText("%g" % shape[1])
        label2 = MyQLabel(self, bold = False, color= qt.Qt.black)
        label2.setText("Number of columns = ")
        self.cancelButton = qt.QPushButton(self)
        self.cancelButton.setText("Dismiss")
        self.okButton    = qt.QPushButton(self)
        self.okButton.setText("Accept")
        self.mainLayout.addWidget(label1, 0, 0)
        self.mainLayout.addWidget(self.rows, 0, 1)
        self.mainLayout.addWidget(label2, 1, 0)
        self.mainLayout.addWidget(self.columns, 1, 1)
        self.mainLayout.addWidget(self.cancelButton, 2, 0)
        self.mainLayout.addWidget(self.okButton, 2, 1)
        self.connect(self.cancelButton,
                     qt.SIGNAL("clicked()"),
                     self.reject)
        self.connect(self.okButton,
                     qt.SIGNAL("clicked()"),
                     self.accept)

    def getImageShape(self):
        text = str(self.rows.text())
        if len(text): nrows = int(text)
        else: nrows = None
        text = str(self.columns.text())
        if len(text): ncolumns = int(text)
        else: ncolumns = None
        ncolumns = int(text)
        return nrows, ncolumns

class MyQLabel(qt.QLabel):
    def __init__(self,parent=None,name=None,fl=0,bold=True, color= qt.Qt.red):
        qt.QLabel.__init__(self,parent)
        if qt.qVersion() <'4.0.0':
            self.color = color
            self.bold  = bold
        else:
            palette = self.palette()
            role = self.foregroundRole()
            palette.setColor(role,color)
            self.setPalette(palette)
            self.font().setBold(bold)


    if qt.qVersion() < '4.0.0':
        def drawContents(self, painter):
            painter.font().setBold(self.bold)
            pal =self.palette()
            pal.setColor(qt.QColorGroup.Foreground,self.color)
            self.setPalette(pal)
            qt.QLabel.drawContents(self,painter)
            painter.font().setBold(0)

def test():
    import RGBCorrelatorGraph
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit()'))

    container = qt.QSplitter()
    #containerLayout = qt.QHBoxLayout(container)
    w = RGBCorrelatorWidget(container)
    graph = RGBCorrelatorGraph.RGBCorrelatorGraph(container)
    def slot(ddict):
        if ddict.has_key('image'):
            image_buffer = ddict['image'].tostring()
            size = ddict['size']
            graph.pixmapPlot(image_buffer,size)
            graph.replot()
    app.connect(w, qt.SIGNAL("RGBCorrelatorWidgetSignal"), slot)
    import getopt
    options=''
    longoptions=[]
    opts, args = getopt.getopt(
                    sys.argv[1:],
                    options,
                    longoptions)      
    for opt,arg in opts:
        pass
    filelist=args
    if len(filelist):
        try:
            import DataSource
            DataReader = DataSource.DataSource
        except:
            import EdfFileDataSource
            DataReader = EdfFileDataSource.EdfFileDataSource
        for fname in filelist:
            source = DataReader(fname)
            for key in source.getSourceInfo()['KeyList']:
                dataObject = source.getDataObject(key)
                w.addImage(dataObject.data, os.path.basename(fname)+" "+key)
    else:
        array1 = Numeric.arange(10000)
        array2 = Numeric.resize(Numeric.arange(10000), (100, 100))
        array2 = Numeric.transpose(array2)
        array3 = array1 * 1
        w.addImage(array1)
        w.addImage(array2)
        w.addImage(array3)
        w.setImageShape([100, 100])
    #containerLayout.addWidget(w)
    #containerLayout.addWidget(graph)
    container.show()
    app.exec_()

if __name__ == "__main__":
    test()
        
