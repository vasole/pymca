#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2004-2007 European Synchrotron Radiation Facility
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
import PyQt4.Qt as qt
import os
import numpy
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import PyMcaMatplotlibSave
from PyMca_Icons import IconDict
import PyMcaPrintPreview
import PyMcaDirs
DEBUG = 0

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)

        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed))

class VerticalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed,qt.QSizePolicy.Expanding))


class SaveImageSetup(qt.QWidget):
    def __init__(self, parent=None, image=None):
	qt.QWidget.__init__(self, parent)
	self.mainLayout = qt.QGridLayout(self)
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
	self._buttonContainer = qt.QWidget(self)
	self._buttonContainer.mainLayout = qt.QVBoxLayout(self._buttonContainer)

	self.updateButton = qt.QPushButton(self._buttonContainer)
	self.updateButton.setText("Update")

	self.printButton = qt.QPushButton(self._buttonContainer)
	self.printButton.setText("Print")

	self.saveButton = qt.QPushButton(self._buttonContainer)
	self.saveButton.setText("Save")

	self._buttonContainer.mainLayout.addWidget(self.updateButton)
	self._buttonContainer.mainLayout.addWidget(self.printButton)
	self._buttonContainer.mainLayout.addWidget(self.saveButton)
	self.mainLayout.addWidget(self._buttonContainer, 0, 1)

	self.connect(self.updateButton, qt.SIGNAL('clicked()'),
		     self.updateClicked)

	self.connect(self.printButton, qt.SIGNAL('clicked()'),
		     self.printClicked)

	self.connect(self.saveButton, qt.SIGNAL('clicked()'),
		     self.saveClicked)


    def sizeHint(self):
	return qt.QSize(3 * qt.QWidget.sizeHint(self).width(),
			3 * qt.QWidget.sizeHint(self).height())

    def setImage(self, image=None):
        self.imageWidget.imageData = image
        self.updateClicked()

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
	ddict = self.getParameters()
	self.imageWidget.setParameters(ddict)

    def printClicked(self):
        pixmap = qt.QPixmap.grabWidget(self.imageWidget)
        self.printPreview.addPixmap(pixmap)
        if self.printPreview.isHidden():
            self.printPreview.show()
        self.printPreview.raise_()
        
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
            outstr=str(outfile.selectedFiles()[0])
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
                print "Cannot delete output file"
                pass
        config = self.imageWidget.getParameters()
        s=PyMcaMatplotlibSave.PyMcaMatplotlibSaveImage(self.imageWidget.imageData)
        s.setParameters(config)
        s.saveImage(finalFile)


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
	for i in self.count():
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
	self.labelList = ['X Axis', 'Y Axis',
			  'Origin', 'Colormap',
			  'Colorbar',
			  'Interpolation',
			  'Contour']
	self.keyList = []
	for label in self.labelList:
	    self.keyList.append(label.lower().replace(' ',''))
	self.comboBoxList = []
	for i in range(len(self.labelList)):
	    label = qt.QLabel(self)
	    label.setText(self.labelList[i])
	    if self.labelList[i] in ['X Axis', 'Y Axis']:
		options = ['Off', 'On']
	    elif self.labelList[i] in ['Colormap']:
		options = ['Default', 'Gray',\
                           'Red', 'Green', 'Blue',\
                           'Rainbow', 'Hot', 'Cool', 'Copper']
                if hasattr(cm, 'spectral'):
                    options.append('Spectral')
	    elif self.labelList[i] in ['Colorbar']:
		options = ['None', 'Vertical', 'Horizontal']
	    elif self.labelList[i] in ['Origin']:
		options = ['Lower', 'Upper']
	    elif self.labelList[i] in ['Interpolation']:
		options = ['Nearest', 'Bilinear']
	    elif self.labelList[i] in ['Contour']:
		options = ['Off', 'Line']
	    line = SimpleComboBox(self, options)
	    self.gridLayout.addWidget(label, i, 0)
	    self.gridLayout.addWidget(line, i, 1)
	    self.comboBoxList.append(line)

	self.mainLayout.addWidget(self.gridWidget)
	self.mainLayout.addWidget(VerticalSpacer(self))


    def getParameters(self):
	ddict = {}
	i = 0
	for label in self.keyList:
	    ddict[label] = str(self.comboBoxList[i].currentText()).lower()
	    if (ddict[label] == 'none') or (ddict[label] == 'default'):
		ddict[label] = None
	    i = i + 1
	return ddict

    def setParameters(self, ddict):
	for label in ddict.keys():
	    if label.lower() in self.keyList:
		i = self.keyList.index(label)
		self.lineEditList[i].setCurrentText(ddict[label])
	return
	

class QPyMcaMatplotlibImage(FigureCanvas):
    def __init__(self, parent, imageData,
		     dpi=100,
                     size=(4, 4),
                     xaxis='off',
                     yaxis='off',
                     xlabel='',
                     ylabel='',
                     colorbar=None,
                     title='',
                     interpolation='nearest',
		     colormap=None,
                     origin='lower',
		     contour='off',
                     extent=None):
	self.figure = Figure(figsize=size, dpi=dpi) #in inches
        FigureCanvas.__init__(self, self.figure)
        FigureCanvas.setSizePolicy(self,
                                   qt.QSizePolicy.Expanding,
                                   qt.QSizePolicy.Expanding)

	self.imageData = imageData
	self.config={'xaxis':xaxis,
		     'yaxis':yaxis,
		     'title':title,
		     'xlabel':xlabel,
		     'ylabel':ylabel,
		     'colorbar':colorbar,
		     'colormap':colormap,
		     'interpolation':interpolation,
		     'origin':origin,
		     'contour':contour,
                     'extent':extent}

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
		     
	self.updateFigure()

    def updateFigure(self):
	self.figure.clear()
	if self.imageData is None:
	    return

	# The axes
        self.axes = self.figure.add_axes([.1, .15, .75, .8])
        if self.config['xaxis'] == 'off':
            self.axes.xaxis.set_visible(False)
        else:
            self.axes.xaxis.set_visible(True)
        if self.config['yaxis'] == 'off':
            self.axes.yaxis.set_visible(False)
        else:
            self.axes.yaxis.set_visible(True)

	interpolation = self.config['interpolation']
	origin = self.config['origin']

	cmap = cm.jet
	ccmap = cm.gray
	if self.config['colormap']=='gray':
	    cmap  = cm.gray
	    ccmap = cm.jet
	elif self.config['colormap']=='hot':
	    cmap = cm.hot
	elif self.config['colormap']=='cool':
	    cmap = cm.cool
	elif self.config['colormap']=='copper':
	    cmap = cm.copper
	elif self.config['colormap']=='spectral':
            cmap = cm.spectral
	elif self.config['colormap']=='rainbow':
            cmap = cm.gist_rainbow
	elif self.config['colormap']=='red':
            cmap = self.__redCmap
	elif self.config['colormap']=='green':
            cmap = self.__greenCmap
	elif self.config['colormap']=='blue':
            cmap = self.__blueCmap

        if self.config['extent'] is None:
            h, w = self.imageData.shape
	    extent = (0,w,0,h)
            if origin == 'upper':
                extent = (0, w, h, 0)
	else:
            extent = self.config['extent'] 

            
        self._image  = self.axes.imshow(self.imageData,
                                        interpolation=interpolation,
                                        origin=origin,
					cmap=cmap,
                                        extent=extent)
        ylim = self.axes.get_ylim()

        self.axes.set_title(self.config['title'])
        self.axes.set_xlabel(self.config['xlabel'])
        self.axes.set_ylabel(self.config['ylabel'])
        
        if self.config['colorbar'] is not None:
	    barorientation = self.config['colorbar']
	    self._colorbar = self.figure.colorbar(self._image,
	                                orientation=barorientation)

	#contour plot
	if self.config['contour'] != 'off':
	    dataMin = self.imageData.min()
	    dataMax = self.imageData.max()
	    levels = (numpy.arange(10)) * (dataMax - dataMin)/10.
	    if self.config['contour'] == 'filled':
		self._contour = self.axes.contourf(self.imageData, levels,
	             origin=origin,
                     cmap=ccmap,
                     extent=extent)
	    else:
		self._contour = self.axes.contour(self.imageData, levels,
	             origin=origin,
                     cmap=ccmap,
	             linewidths=2,
                     extent=extent)
	    self.axes.clabel(self._contour, fontsize=9, inline=1)
            if 0 and  self.config['colorbar'] is not None:
                if barorientation == 'horizontal':
                    barorientation = 'vertical'
                else:
                    barorientation = 'horizontal'
        	self._ccolorbar=self.figure.colorbar(self._contour,
                                                     orientation=barorientation,
                                                     extend='both')

        self.axes.set_ylim(ylim[0],ylim[1])

	self.draw()

    def getParameters(self):
	return self.config

    def setParameters(self, ddict):
	self.config.update(ddict)
	self.updateFigure()

if __name__ == "__main__":
    app = qt.QApplication([])
    a=numpy.arange(1200.)
    a.shape = 20, 60
    w = SaveImageSetup(None, a)
    w.show()
    app.exec_()
		
   
