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
import os
import numpy
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

DEBUG = 0

colordict = {}
colordict['blue']   = '#0000ff'
colordict['red']    = '#ff0000'
colordict['green']  = '#00ff00'
colordict['black']  = '#000000'
colordict['white']  = '#ffffff'
colordict['pink']   = '#ff66ff'
colordict['brown']  = '#a52a2a'
colordict['orange'] = '#ff9900'
colordict['violet'] = '#6600ff'
colordict['grey']   = '#808080'
colordict['yellow'] = '#ffff00'
colordict['darkgreen'] = 'g'
colordict['darkbrown'] = '#660000' 
colordict['magenta']   = 'm' 
colordict['cyan']      = 'c'
colordict['bluegreen'] = '#33ffff'
colorlist  = [colordict['black'],
              colordict['red'],
              colordict['blue'],
              colordict['green'],
              colordict['pink'],
              colordict['brown'],
              colordict['cyan'],
              colordict['orange'],
              colordict['violet'],
              colordict['bluegreen'],
              colordict['grey'],
              colordict['magenta'],
              colordict['darkgreen'],
              colordict['darkbrown'],
              colordict['yellow']]

class PyMcaMatplotlibSave:
    def __init__(self, size = (6,3),
                 logx = False,
                 logy = False,
                 legends = True,
                 bw = False):

        self.fig = Figure(figsize=size) #in inches
        self.canvas = FigureCanvas(self.fig)

        self._logX = logx
        self._logY = logy
        self._bw   = bw
        self._legend   = legends
        self._legendList = []
        self._dataCounter = 0

        if not legends:
            if self._logY:
                ax = self.fig.add_axes([.15, .15, .75, .8])
            else:
                ax = self.fig.add_axes([.15, .15, .75, .75])
        else:
            if self._logY:
                ax = self.fig.add_axes([.15, .15, .7, .8])
            else:
                ax = self.fig.add_axes([.15, .15, .7, .8])

        ax.set_axisbelow(True)

        self.ax = ax


        if self._logY:
            self._axFunction = ax.semilogy
        else:
            self._axFunction = ax.plot

        if self._bw:
            self.colorList = ['k']   #only black
            self.styleList = ['-', ':', '-.', '--']
            self.nColors   = 1
        else:
            self.colorList = colorlist
            self.styleList = ['-', '-.', ':']
            self.nColors   = len(colorlist)
        self.nStyles   = len(self.styleList)

        self.colorIndex = 0
        self.styleIndex = 0

        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.limitsSet = False

    def setLimits(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.limitsSet = True


    def _filterData(self, x, y):
        index = numpy.flatnonzero((self.xmin <= x) & (x <= self.xmax))
        x = numpy.take(x, index)
        y = numpy.take(y, index)
        index = len(index)
        if index:
            index = numpy.flatnonzero((self.ymin <= y) & (y <= self.ymax))
            index = len(index)
        return index

    def _getColorAndStyle(self):
        color = self.colorList[self.colorIndex]
        style = self.styleList[self.styleIndex]
        self.colorIndex += 1
        if self.colorIndex >= self.nColors:
            self.colorIndex = 0
            self.styleIndex += 1
            if self.styleIndex >= self.nStyles:
                self.styleIndex = 0        
        return color, style

    def addDataToPlot(self, x, y, legend = None,
                      color = None,
                      linewidth = None,
                      linestyle = None, **kw):
        n = max(x.shape)
        if self.limitsSet is not None:
            n = self._filterData(x, y)
        if n == 0:
            #nothing to plot
            if DEBUG:
                print "nothing to plot"
            return
        style = None
        if color is None:
            color, style = self._getColorAndStyle()
        if linestyle is None:
            if style is None:
                style = '-'
        else:
            style = linestyle

        if linewidth is None:linewidth = 1.0
        self._axFunction( x, y, linestyle = style, color=color, linewidth = linewidth, **kw)
        self._dataCounter += 1
        if legend is None:
            #legend = "%02d" % self._dataCounter    #01, 02, 03, ...
            legend = "%c" % (96+self._dataCounter)  #a, b, c, ..
        self._legendList.append(legend)

    def setXLabel(self, label):
        self.ax.set_xlabel(label)

    def setYLabel(self, label):
        self.ax.set_ylabel(label)
        
    def plotLegends(self):
        if not self._legend:return
        if not len(self._legendList):return
        loc = (1.01, 0.0)
        labelsep = 0.015
        drawframe = True
        if len(self._legendList) > 14:
            drawframe = False
            loc = (1.05, -0.2)
            fontproperties = FontProperties(size=8)
        else:
            fontproperties = FontProperties(size=10)

        legend = self.ax.legend(self._legendList,
                                loc = loc,
                                prop = fontproperties,
                                labelsep = labelsep,
                                pad = 0.15)

        legend.draw_frame(drawframe)


    def saveFile(self, filename, format=None):
        if format is None:
            format = filename[-3:]

        if format.upper() not in ['EPS', 'PNG', 'SVG']:
            raise "Unknown format %s" % format

        if os.path.exists(filename):
            os.remove(filename)

        if self.limitsSet:
            self.ax.set_ylim(self.ymin, self.ymax)
            self.ax.set_xlim(self.xmin, self.xmax)

        self.canvas.print_figure(filename)
        return

class PyMcaMatplotlibSaveImage:
    def __init__(self, imageData=None, fileName=None,
		     dpi=300,
                     size=(5, 5),
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

        self.figure = Figure(figsize=size) #in inches
        self.canvas = FigureCanvas(self.figure)
	self.imageData = imageData
	self.pixmapImage = None
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
        
        #Do I really need as many colors?
        self.__temperatureCmap = LinearSegmentedColormap('temperature',
                                                         cdict, 65536)
        if fileName is not None:
            self.saveImage(fileName)

    def setImage(self, image=None):
        self.imageData = image

    def setParameters(self, ddict):
        self.config.update(ddict)

    def saveImage(self, fileName):
        self.figure.clear()
	if (self.imageData is None) and\
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
            self._savePixmapFigure(fileName)
            return

	interpolation = self.config['interpolation']
	origin = self.config['origin']

	cmap = self.__temperatureCmap
	ccmap = cm.gray
        if self.config['colormap'] in ['grey','gray']:
	    cmap  = cm.gray
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

        self.canvas.print_figure(fileName)
        

    def setPixmapImage(self, image=None, bgr=False):
        if bgr:
            self.pixmapImage = image * 1
            self.pixmapImage[:,:,0] = image[:,:,2]
            self.pixmapImage[:,:,2] = image[:,:,0]
        else:
            self.pixmapImage = image

    def _savePixmapFigure(self, fileName):
	interpolation = self.config['interpolation']
	origin = self.config['origin']
        if self.config['extent'] is None:
            h= self.pixmapImage.shape[0]
            w= self.pixmapImage.shape[1]
            
            extent = (0,w,0,h)
            if origin == 'upper':
                extent = (0, w, h, 0)
        else:
            extent = self.config['extent']
        self._image = self.axes.imshow(self.pixmapImage,
                                       interpolation=interpolation,
                                       origin=origin,
                                       extent=extent)

        ylim = self.axes.get_ylim()

        self.axes.set_title(self.config['title'])
        self.axes.set_xlabel(self.config['xlabel'])
        self.axes.set_ylabel(self.config['ylabel'])

        self.axes.set_ylim(ylim[0],ylim[1])

        self.canvas.print_figure(fileName)

        
if __name__ == "__main__":
    import sys
    a=numpy.arange(1200.)
    a.shape = 20, 60
    PyMcaMatplotlibSaveImage(a, "filename.png", colormap="rainbow")
    print "Image filename.png saved"
    sys.exit(0)
    
