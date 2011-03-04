#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2004-2011 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Software Group"
import PyMcaQt as qt
import sys
import os
import numpy
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
import PyMcaMatplotlibSave
from PyMca_Icons import IconDict
import PyMcaPrintPreview
import PyMcaDirs

DEBUG = 0

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)

        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                                          qt.QSizePolicy.Fixed))

class VerticalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed,
                                          qt.QSizePolicy.Expanding))

class TopWidget(qt.QWidget):
    def __init__(self, parent = None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        self.labelList = ['Title', 'X Label', 'Y Label']
        self.keyList   = ['title', 'xlabel', 'ylabel']
        self.lineEditList = []
        for i in range(len(self.labelList)):
            label = qt.QLabel(self)
            label.setText(self.labelList[i])
            lineEdit = qt.QLineEdit(self)
            self.mainLayout.addWidget(label, i, 0)
            self.mainLayout.addWidget(lineEdit, i, 1)
            self.lineEditList.append(lineEdit)

    def getParameters(self):
        ddict = {}
        i = 0
        for label in self.keyList:
            ddict[label] = str(self.lineEditList[i].text())
            i = i + 1
        return ddict

    def setParameters(self, ddict):
        for label in ddict.keys():
            if label.lower() in self.keyList:
                i = self.keyList.index(label)
                self.lineEditList[i].setText(ddict[label])
        return

class ButtonsWidget(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)

        self.updateButton = qt.QPushButton(self)
        self.updateButton.setText("Update")

        self.printButton = qt.QPushButton(self)
        self.printButton.setText("Print")

        self.saveButton = qt.QPushButton(self)
        self.saveButton.setText("Save")

        self.mainLayout.addWidget(self.updateButton)
        self.mainLayout.addWidget(self.printButton)
        self.mainLayout.addWidget(self.saveButton)

class SaveImageSetup(qt.QWidget):
    def __init__(self, parent=None, image=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setColumnStretch(0, 1)
        self.mainLayout.setColumnStretch(1, 0)
        self.setWindowTitle("PyMca - Matplotlib save image")
        self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
        self.lastOutputDir = None
        self.printPreview = PyMcaPrintPreview.PyMcaPrintPreview(modal = 0)

        #top
        self.top = TopWidget(self)
        self.mainLayout.addWidget(self.top, 0, 0)

        #image
        self.imageWidget = QPyMcaMatplotlibImage(self, image)
        self.mainLayout.addWidget(self.imageWidget, 1, 0)

        #right
        self.right = RightWidget(self)
        self.mainLayout.addWidget(self.right, 1, 1)

        #buttons
        self._buttonContainer = ButtonsWidget(self)
        self.mainLayout.addWidget(self._buttonContainer, 0, 1)

        self.connect(self._buttonContainer.updateButton,
                     qt.SIGNAL('clicked()'),
                     self.updateClicked)

        self.connect(self._buttonContainer.printButton,
                     qt.SIGNAL('clicked()'),
                     self.printClicked)

        self.connect(self._buttonContainer.saveButton, qt.SIGNAL('clicked()'),
                     self.saveClicked)


    def sizeHint(self):
        return qt.QSize(3 * qt.QWidget.sizeHint(self).width(),
                        3 * qt.QWidget.sizeHint(self).height())

    def setImageData(self, image=None):
        self.imageWidget.imageData = image
        self.updateClicked()

    def setPixmapImage(self, image=None, bgr=False):
        #this is not to loose time plotting twice
        self.imageWidget.setPixmapImage(None, bgr)
        if image is None:
            self.right.setPixmapMode(False)
        else:
            self.right.setPixmapMode(True)
        #update configuration withoutplotting because of having
        #set the current pixmap to None
        self.updateClicked()
        #and plot
        self.imageWidget.setPixmapImage(image, bgr)

    def getParameters(self):
        ddict = self.imageWidget.getParameters()
        ddict.update(self.top.getParameters())
        ddict.update(self.right.getParameters())
        return ddict

    def setParameters(self, ddict):
        self.top.setParameters(ddict)
        self.imageWidget.setParameters(ddict)
        self.right.setParameters(ddict)

    def updateClicked(self):
        try:
            ddict = self.getParameters()
            self.imageWidget.setParameters(ddict)
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error updating image: %s" % sys.exc_info()[1])
            msg.setWindowTitle('Matplotlib Save Image')
            msg.exec_()
            

    def printClicked(self):
        try:
            pixmap = qt.QPixmap.grabWidget(self.imageWidget)
            self.printPreview.addPixmap(pixmap)
            if self.printPreview.isHidden():
                self.printPreview.show()
            self.printPreview.raise_()
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error printing image: %s" % sys.exc_info()[1])
            msg.setWindowTitle('Matplotlib Save Image')
            msg.exec_()
        
    def saveClicked(self):
        outfile = qt.QFileDialog(self)
        outfile.setModal(1)
        if self.lastOutputDir is None:
            self.lastOutputDir = PyMcaDirs.outputDir

        outfile.setWindowTitle("Output File Selection")
        strlist = qt.QStringList()
        format_list = []
        format_list.append('Graphics PNG *.png')
        format_list.append('Graphics EPS *.eps')
        format_list.append('Graphics SVG *.svg')
        for f in format_list:
            strlist.append(f)
        outfile.setFilters(strlist)

        outfile.setFileMode(outfile.AnyFile)
        outfile.setAcceptMode(qt.QFileDialog.AcceptSave)
        outfile.setDirectory(self.lastOutputDir)
        ret = outfile.exec_()
        if ret:
            filterused = str(outfile.selectedFilter()).split()
            filedescription = filterused[0]
            filetype  = filterused[1]
            extension = filterused[2]
            try:
                outstr=str(outfile.selectedFiles()[0])
            except UnicodeError:
                print("WARNING: Unsupported characters in file name, trying workaround")
                try:
                    outstr = str(outfile.selectedFiles()[0].toLocal8Bit())
                except:
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Error saving image: %s" % sys.exc_info()[1])
                    msg.setWindowTitle('Matplotlib Save Image')
                    msg.exec_()
            try:            
                outputDir  = os.path.dirname(outstr)
                self.lastOutputDir = outputDir
                PyMcaDirs.outputDir = outputDir
            except:
                outputDir  = "."
            try:            
                outputFile = os.path.basename(outstr)
            except:
                outputFile  = outstr
            outfile.close()
            del outfile
        else:
            outfile.close()
            del outfile
            return
        #always overwrite for the time being
        if len(outputFile) < len(extension[1:]):
            outputFile += extension[1:]
        elif outputFile[-4:] != extension[1:]:
            outputFile += extension[1:]

        finalFile = os.path.join(outputDir, outputFile)
        if os.path.exists(finalFile):
            try:
                os.remove(finalFile)
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Cannot overwrite file: %s" % sys.exc_info()[1])
                msg.setWindowTitle('Matplotlib Save Image')
                msg.exec_()
                return
        try:
            self.imageWidget.print_figure(finalFile,
                                          edgecolor='w',
                                          facecolor='w',
                                          format=finalFile[-3:])
        except:
            print("WARNING: trying to save using obsolete method")
            config = self.imageWidget.getParameters()
            try:
                s=PyMcaMatplotlibSave.PyMcaMatplotlibSaveImage(self.imageWidget.imageData)
                if self.imageWidget.pixmapImage is not None:
                    s.setPixmapImage(self.imageWidget.pixmapImage)
                s.setParameters(config)
                s.saveImage(finalFile)
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error saving file: %s" % sys.exc_info()[1])
                msg.setWindowTitle('Matplotlib Save Image')
                msg.exec_()

class SimpleComboBox(qt.QComboBox):
    def __init__(self, parent=None, options=['1', '2', '3']):
        qt.QComboBox.__init__(self,parent)
        self.setOptions(options)
        self.setDuplicatesEnabled(False)
        self.setEditable(False)

    def setOptions(self,options=['1','2','3']):
        self.clear()
        for item in options:
    	    self.addItem(item)

    def setCurrentText(self, text):
        for i in range(self.count()):
            if str(self.itemText(i)) == text:
                self.setCurrentIndex(i)
                break

class RightWidget(qt.QWidget):
    def __init__(self, parent = None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.gridWidget = qt.QWidget(self) 
        self.gridLayout = qt.QGridLayout(self.gridWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setSpacing(2)
        self.labelList = ['X Axis',
                        'Y Axis',
                        'Origin',
                        'Interpolation',
                        'Colormap',
                        'Lin/Log Colormap',
                        'Colorbar',
                        'Contour',
                        'Contour Labels',
                        'Contour Label Format',
                        'Contour Levels',
                        'Image Background',
                        'X Pixel Size',
                        'Y Pixel Size',
                        'X Origin',
                        'Y Origin',
                        'Zoom X Min',
                        'Zoom X Max',
                        'Zoom Y Min',
                        'Zoom Y Max',
                        'Value Min',
                        'Value Max']
        self.keyList = []
        for label in self.labelList:
            self.keyList.append(label.lower().replace(' ','').replace('/',""))
        self.comboBoxList = []
        for i in range(len(self.labelList)):
            label = qt.QLabel(self)
            label.setText(self.labelList[i])
            if self.labelList[i] in ['X Axis', 'Y Axis']:
                options = ['Off', 'On']
            elif self.labelList[i] in ['Colormap']:
                options = ['Temperature','Grey', 'Yerg',\
                           'Red', 'Green', 'Blue',\
                           'Rainbow', 'Jet','Hot', 'Cool', 'Copper']
                if hasattr(cm, 'spectral'):
                    options.append('Spectral')
            elif self.labelList[i] in ['Lin/Log Colormap']:
                options = ['Linear','Logarithmic']
            elif self.labelList[i] in ['Colorbar']:
                options = ['None', 'Vertical', 'Horizontal']
            elif self.labelList[i] in ['Origin']:
                options = ['Lower', 'Upper']
            elif self.labelList[i] in ['Interpolation']:
                options = ['Nearest', 'Bilinear']
            elif self.labelList[i] in ['Contour']:
                options = ['Off', 'Line']
            elif self.labelList[i] in ['Contour Labels']:
                options = ['On', 'Off']
            elif self.labelList[i] in ['Contour Label Format']:
                options = ['%.3f', '%.2f', '%.1f', '%.0f', '%.1e', '%.2e', '%.3e']
            elif self.labelList[i] in ['Contour Levels']:
                options = ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]
            elif self.labelList[i] in ['Image Background']:
                options = ['Black', 'White', 'Grey']
            if i <= self.labelList.index('Image Background'):
                line = SimpleComboBox(self, options)
            else:
                line = MyLineEdit(self)
                validator = qt.QDoubleValidator(line)
                line.setValidator(validator)
                if 'Zoom' in self.labelList[i]:
                    tip  = "This zoom is in physical units.\n"
                    tip += "This means pixel size corrected.\n"
                    tip += "To disable zoom, just set both\n"
                    tip += "limits to the same value."
                    line.setToolTip(tip)
                    line.setText('0.0')
                elif 'Origin' in self.labelList[i]:
                    tip  = "First pixel coordinates in physical units.\n"
                    tip += "This means pixel size corrected.\n"
                    line.setToolTip(tip)
                    line.setText('0.0')
                elif 'Value' in self.labelList[i]:
                    tip  = "Clipping values of the data.\n"
                    tip += "To disable clipping, just set both\n"
                    tip += "limits to the same value."
                    line.setToolTip(tip)
                    line.setText('0.0')
                else:
                    line.setText('1.0')
            self.gridLayout.addWidget(label, i, 0)
            self.gridLayout.addWidget(line, i, 1)
            self.comboBoxList.append(line)

        self.mainLayout.addWidget(self.gridWidget)
        self.mainLayout.addWidget(VerticalSpacer(self))
        self.setPixmapMode(False)

    def setPixmapMode(self, flag):
        if flag:
            disable = ['Colormap', 'Lin/Log Colormap', 'Contour', 'Contour Labels',
                       'Contour Label Format',
                       'Contour Levels', 'Colorbar', 'Value Min','Value Max']
        else:
            disable = ['Image Background']

        for label in self.labelList:
            index = self.labelList.index(label)
            if label in disable:
                self.comboBoxList[index].setEnabled(False)
            else:
                self.comboBoxList[index].setEnabled(True)

    def getParameters(self):
        ddict = {}
        i = 0
        for label in self.keyList:
            if i > self.labelList.index('Image Background'):
                text = str(self.comboBoxList[i].text())
                if len(text):
                    ddict[label] = float(text)
                else:
                    ddict[label] = None
            else:
                ddict[label] = str(self.comboBoxList[i].currentText()).lower()
            if (ddict[label] == 'none') or (ddict[label] == 'default'):
                ddict[label] = None
            i = i + 1
        return ddict

    def setParameters(self, ddict):
        for label in ddict.keys():
            if label.lower() in self.keyList:
                i = self.keyList.index(label)
                if i > self.labelList.index('Image Background'):
                    if ddict[label] is not None:
                        self.comboBoxList[i].setText("%f" % ddict[label])
                else:
                    txt = ddict[label]
                    if ddict[label] is not None:
                        try:
                            txt = ddict[label][0].upper() +\
                                  ddict[label][1:].lower()
                        except:
                            pass
                    self.comboBoxList[i].setCurrentText(txt)
        return
	
class MyLineEdit(qt.QLineEdit):
    def sizeHint(self):
        return qt.QSize(0.6 * qt.QLineEdit.sizeHint(self).width(),
                        qt.QLineEdit.sizeHint(self).height())


class QPyMcaMatplotlibImage(FigureCanvas):
    def __init__(self, parent, imageData=None,
		     dpi=100,
                     size=(5, 5),
                     xaxis='off',
                     yaxis='off',
                     xlabel='',
                     ylabel='',
                     colorbar=None,
                     title='',
                     interpolation='nearest',
                     colormap=None,
                     linlogcolormap='linear',
                     origin='lower',
                     contour='off',
                     contourlabels='on',
                     contourlabelformat='%.3f',                 
                     contourlevels=2,
                     extent=None,
                     xpixelsize=1.0,
                     ypixelsize=1.0,
                     xorigin=0.0,
                     yorigin=0.0,
                     xlimits=None,
                     ylimits=None,
                     vlimits=None):
        self.figure = Figure(figsize=size, dpi=dpi) #in inches

        #How to set this color equal to the other widgets color?
        #self.figure.set_facecolor('1.0')
        #self.figure.set_edgecolor('1.0')

        FigureCanvas.__init__(self, self.figure)
        FigureCanvas.setSizePolicy(self,
                                   qt.QSizePolicy.Expanding,
                                   qt.QSizePolicy.Expanding)

        self.imageData = imageData
        self.pixmapImage = None
        self.config={'xaxis':xaxis,
                     'yaxis':yaxis,
                     'title':title,
                     'xlabel':xlabel,
                     'ylabel':ylabel,
                     'colorbar':colorbar,
                     'colormap':colormap,
                     'linlogcolormap':linlogcolormap,
                     'interpolation':interpolation,
                     'origin':origin,
                     'contour':contour,
                     'contourlabels':contourlabels,
                     'contourlabelformat':contourlabelformat,
                     'contourlevels':contourlevels,
                     'extent':extent,
                     'imagebackground':'black',
                     'xorigin':xorigin,
                     'yorigin':yorigin,
                     'xpixelsize':xpixelsize,
                     'ypixelsize':ypixelsize,
                     'zoomxmin':None,
                     'zoomxmax':None,
                     'zoomymin':None,
                     'zoomymax':None,
                     'valuemin':None,
                     'valuemax':None,
                     'xlimits':xlimits,
                     'ylimits':ylimits,
                     'vlimits':vlimits}

        #generate own colormaps
        cdict = {'red': ((0.0, 0.0, 0.0),
                         (1.0, 1.0, 1.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 0.0, 0.0),
                          (1.0, 0.0, 0.0))}
        self.__redCmap = LinearSegmentedColormap('red',cdict,256)

        cdict = {'red': ((0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (1.0, 1.0, 1.0)),
                 'blue': ((0.0, 0.0, 0.0),
                          (1.0, 0.0, 0.0))}
        self.__greenCmap = LinearSegmentedColormap('green',cdict,256)

        cdict = {'red': ((0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 0.0, 0.0),
                          (1.0, 1.0, 1.0))}
        self.__blueCmap = LinearSegmentedColormap('blue',cdict,256)

        # Temperature as defined in spslut
        cdict = {'red': ((0.0, 0.0, 0.0),
                         (0.5, 0.0, 0.0),
                         (0.75, 1.0, 1.0),
                         (1.0, 1.0, 1.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (0.25, 1.0, 1.0),
                           (0.75, 1.0, 1.0),
                           (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 1.0, 1.0),
                          (0.25, 1.0, 1.0),
                          (0.5, 0.0, 0.0),
                          (1.0, 0.0, 0.0))}
        #but limited to 256 colors for a faster display (of the colorbar)
        self.__temperatureCmap = LinearSegmentedColormap('temperature',
                                                         cdict, 256)

        #reversed gray
        cdict = {'red':     ((0.0, 1.0, 1.0),
                             (1.0, 0.0, 0.0)),
                 'green':   ((0.0, 1.0, 1.0),
                             (1.0, 0.0, 0.0)),
                 'blue':    ((0.0, 1.0, 1.0),
                             (1.0, 0.0, 0.0))}
                         
        self.__reversedGrayCmap = LinearSegmentedColormap('yerg', cdict, 256)

        self.updateFigure()

    def updateFigure(self):
        self.figure.clear()
        if (self.imageData is None) and \
           (self.pixmapImage is None):
            return

	# The axes
        self.axes = self.figure.add_axes([.15, .15, .75, .8])
        if self.config['xaxis'] == 'off':
            self.axes.xaxis.set_visible(False)
        else:
            self.axes.xaxis.set_visible(True)
        if self.config['yaxis'] == 'off':
            self.axes.yaxis.set_visible(False)
        else:
            self.axes.yaxis.set_visible(True)

        if self.pixmapImage is not None:
            self._updatePixmapFigure()
            return

        interpolation = self.config['interpolation']
        origin = self.config['origin']

        cmap = self.__temperatureCmap
        ccmap = cm.gray
        if self.config['colormap'] in ['grey','gray']:
            cmap  = cm.gray
            ccmap = self.__temperatureCmap
        elif self.config['colormap'] in ['yarg','yerg']:
            cmap  = self.__reversedGrayCmap
            ccmap = self.__temperatureCmap
        elif self.config['colormap']=='jet':
            cmap = cm.jet
        elif self.config['colormap']=='hot':
            cmap = cm.hot
        elif self.config['colormap']=='cool':
            cmap = cm.cool
        elif self.config['colormap']=='copper':
            cmap = cm.copper
        elif self.config['colormap']=='spectral':
            cmap = cm.spectral
        elif self.config['colormap']=='hsv':
            cmap = cm.hsv
        elif self.config['colormap']=='rainbow':
            cmap = cm.gist_rainbow
        elif self.config['colormap']=='red':
            cmap = self.__redCmap
        elif self.config['colormap']=='green':
            cmap = self.__greenCmap
        elif self.config['colormap']=='blue':
            cmap = self.__blueCmap
        elif self.config['colormap']=='temperature':
            cmap = self.__temperatureCmap

        if self.config['extent'] is None:
            h, w = self.imageData.shape
            x0 = self.config['xorigin']
            y0 = self.config['yorigin']
            w = w * self.config['xpixelsize']
            h = h * self.config['ypixelsize']
            if origin == 'upper':
                extent = (x0, w+x0,
                          h+y0, y0)
            else:
                extent = (x0, w+x0,
                          y0, h+y0)
        else:
            extent = self.config['extent']

    
        vlimits = self.__getValueLimits()
        if vlimits is None:
            imageData = self.imageData
            vmin = self.imageData.min()
            vmax = self.imageData.max()
        else:
            vmin = min(vlimits[0], vlimits[1])
            vmax = max(vlimits[0], vlimits[1])
            imageData = self.imageData.clip(vmin,vmax)

        if self.config['linlogcolormap'] != 'linear':
            if vmin <= 0:
                if vmax > 0:                   
                    vmin = min(imageData[imageData>0])
                else:
                    vmin = 0.0
                    vmax = 1.0                
            self._image  = self.axes.imshow(imageData.clip(vmin,vmax),
                                        interpolation=interpolation,
                                        origin=origin,
                                        cmap=cmap,
                                        extent=extent,
                                        norm=LogNorm(vmin, vmax))
        else:
            self._image  = self.axes.imshow(imageData,
                                        interpolation=interpolation,
                                        origin=origin,
                                        cmap=cmap,
                                        extent=extent,
                                        norm=Normalize(vmin, vmax))

        ylim = self.axes.get_ylim()
        
        if self.config['colorbar'] is not None:
            barorientation = self.config['colorbar']
            self._colorbar = self.figure.colorbar(self._image,
                                        orientation=barorientation)

        #contour plot
        if self.config['contour'] != 'off':
            dataMin = imageData.min()
            dataMax = imageData.max()
            ncontours = int(self.config['contourlevels'])
            levels = (numpy.arange(ncontours)) *\
                     (dataMax - dataMin)/float(ncontours)	    
            if self.config['contour'] == 'filled':
                self._contour = self.axes.contourf(imageData, levels,
                     origin=origin,
                     cmap=ccmap,
                     extent=extent)
            else:
                self._contour = self.axes.contour(imageData, levels,
                     origin=origin,
                     cmap=ccmap,
                     linewidths=2,
                     extent=extent)
            if self.config['contourlabels'] != 'off':
                self.axes.clabel(self._contour, fontsize=9,
                        inline=1, fmt=self.config['contourlabelformat'])
            if 0 and  self.config['colorbar'] is not None:
                if barorientation == 'horizontal':
                    barorientation = 'vertical'
                else:
                    barorientation = 'horizontal'
                self._ccolorbar=self.figure.colorbar(self._contour,
                                                     orientation=barorientation,
                                                     extend='both')

        self.__postImage(ylim)

    def getParameters(self):
        return self.config

    def setParameters(self, ddict):
        self.config.update(ddict)
        self.updateFigure()

    def setPixmapImage(self, image=None, bgr=False):
        if image is None:
            self.pixmapImage = None
            self.updateFigure()
            return
        
        if bgr:
            self.pixmapImage = image * 1
            self.pixmapImage[:,:,0] = image[:,:,2]
            self.pixmapImage[:,:,2] = image[:,:,0]
        else:
            self.pixmapImage = image

        shape = self.pixmapImage.shape
        self.pixmapMask = numpy.ones(shape, numpy.uint8)
        shape = self.pixmapImage.shape
        if 0:
            # This is slow, but I do not expect huge images
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if (self.pixmapImage[i,j,0] == 0):
                        if (self.pixmapImage[i,j,1] == 0):
                            if (self.pixmapImage[i,j,2] == 0):
                                self.pixmapMask[i,j,0:3] = [0, 0, 0]
        else:
            #the image is RGBA, so the sum when there is nothing is 255
            s = self.pixmapImage.sum(axis=-1)
            self.pixmapMask[s==255, 0:3] = 0
        self.updateFigure()

    def _updatePixmapFigure(self):
        interpolation = self.config['interpolation']
        origin = self.config['origin']
        if self.config['extent'] is None:
            h= self.pixmapImage.shape[0]
            w= self.pixmapImage.shape[1]
            x0 = self.config['xorigin']
            y0 = self.config['yorigin']
            w = w * self.config['xpixelsize']
            h = h * self.config['ypixelsize']
            if origin == 'upper':
                extent = (x0, w+x0,
                          h+y0, y0)
            else:
                extent = (x0, w+x0,
                          y0, h+y0)
        else:
            extent = self.config['extent']
        if self.config['imagebackground'].lower() == 'white':
            if 0:
                self.pixmapImage[:] = (self.pixmapImage * self.pixmapMask) +\
                               (self.pixmapMask == 0) * 255
            else:
                self.pixmapImage[self.pixmapMask == 0] = 255
        elif self.config['imagebackground'].lower() == 'grey':
            if 0:
                self.pixmapImage[:] = (self.pixmapImage * self.pixmapMask) +\
                               (self.pixmapMask == 0) * 128
            else:
                self.pixmapImage[self.pixmapMask == 0] = 128
        else:
            if 0:
                self.pixmapImage[:] = (self.pixmapImage * self.pixmapMask)
            else:
                self.pixmapImage[self.pixmapMask == 0]= 0
        self._image = self.axes.imshow(self.pixmapImage,
                                       interpolation=interpolation,
                                       origin=origin,
                                       extent=extent)

        ylim = self.axes.get_ylim()
        self.__postImage(ylim)

    def __getValueLimits(self):
        if (self.config['valuemin'] is not None) and\
           (self.config['valuemax'] is not None) and\
           (self.config['valuemin'] != self.config['valuemax']):
            vlimits = (self.config['valuemin'],
                           self.config['valuemax'])
        elif self.config['vlimits'] is not None:
            vlimits = self.config['vlimits']
        else:
            vlimits = None
        return vlimits

    def __postImage(self, ylim):
        self.axes.set_title(self.config['title'])
        self.axes.set_xlabel(self.config['xlabel'])
        self.axes.set_ylabel(self.config['ylabel'])

        origin = self.config['origin']
        if (self.config['zoomxmin'] is not None) and\
           (self.config['zoomxmax'] is not None)and\
           (self.config['zoomxmax'] != self.config['zoomxmin']):
            xlimits = (self.config['zoomxmin'],
                           self.config['zoomxmax'])
        elif self.config['xlimits'] is not None:
            xlimits = self.config['xlimits']
        else:
            xlimits = None

        if (self.config['zoomymin'] is not None) and\
           (self.config['zoomymax'] is not None) and\
           (self.config['zoomymax'] != self.config['zoomymin']):
            ylimits = (self.config['zoomymin'],
                           self.config['zoomymax'])
        elif self.config['ylimits'] is not None:
            ylimits = self.config['ylimits']
        else:
            ylimits = None
        
        if ylimits is None:
            self.axes.set_ylim(ylim[0],ylim[1])
        else:
            ymin = min(ylimits)
            ymax = max(ylimits)
            if origin == "lower":
                self.axes.set_ylim(ymin, ymax)
            else:
                self.axes.set_ylim(ymax, ymin)
                
        if xlimits is not None:
            xmin = min(xlimits)
            xmax = max(xlimits)
            self.axes.set_xlim(xmin, xmax)

        self.draw()

def test():
    app = qt.QApplication([])
    a=numpy.arange(256.)
    a.shape = 16, 16
    w = SaveImageSetup(None, a)
    w.setParameters(w.getParameters())
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()   
