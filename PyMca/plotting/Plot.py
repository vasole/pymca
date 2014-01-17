#/*##########################################################################
# Copyright (C) 2004-2014 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__doc__ = """
This module can be used for plugin testing purposes as well as for doing
the bookkeeping of actual plot windows. 

Functions to be implemented by an actual plotter can be found in the
abstract class PlotBackend.

"""
import numpy
import PlotBase
from PlotBase import PlotBackend

DEBUG = 0
if DEBUG:
    PlotBase.DEBUG = True

# should the color handling belong to the PlotBase class?
colordict = {}
colordict['b'] = colordict['blue']   = '#0000ff'
colordict['r'] = colordict['red']    = '#ff0000'
colordict['g'] = colordict['green']  = '#00ff00'
colordict['k'] = colordict['black']  = '#000000'
colordict['white']  = '#ffffff'
colordict['pink']   = '#ff66ff'
colordict['brown']  = '#a52a2a'
colordict['orange'] = '#ff9900'
colordict['violet'] = '#6600ff'
colordict['grey']   = '#808080'
colordict['y'] = colordict['yellow'] = '#ffff00'
colordict['darkgreen'] = '#006400'
colordict['darkbrown'] = '#660000' 
colordict['m'] = colordict['magenta'] = '#ff00ff'
colordict['c'] = colordict['cyan'] = '#00ffff'
colordict['bluegreen'] = '#33ffff'
colorlist  = [colordict['black'],
              colordict['blue'],
              colordict['red'],
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

#PyQtGraph symbols ['o', 's', 't', 'd', '+', 'x']

class Plot(PlotBase.PlotBase):
    PLUGINS_DIR = None

    def __init__(self, parent=None, backend=None, callback=None):
        self._parent = parent
        if backend is None:
            # an empty backend for testing purposes
            self._plot = PlotBackend.PlotBackend(parent)
        else:
            self._plot = backend(parent)
            self._default = False
        super(Plot, self).__init__()
        widget = self._plot.getWidgetHandle()
        if widget is None:
            self.widget_ = self._plot
        else:
            self.widget_ = widget

        if callback is None:
            self._plot.setCallback(self.graphCallback)

        self.setLimits = self._plot.setLimits

        # curve handling
        self._curveList = []
        self._curveDict = {}
        self._activeCurve = None

        #image handling
        self._imageList = []
        self._imageDict = {}
        self._activeImage = None

        # marker handling
        self._markerDict = {}
        self._markerList = []
        
        # colors and line types
        self._colorList = colorlist
        self._styleList = ['-', '--', '-.', ':']
        self._nColors   = len(colorlist)
        self._nStyles   = len(self._styleList)

        self._colorIndex = 1 # black is reserved
        self._styleIndex = 0

        # default properties
        self._logY = False
        self._logX = False
        
        self.setDefaultPlotPoints(False)
        self.setDefaultPlotLines(True)

        # zoom handling (should we take care of it?)
        self.enableZoom = self.setZoomModeEnabled
        self.setZoomModeEnabled(True)

    def getWidgetHandle(self):
        return self.widget_

    def setCallback(self, callbackFunction):
        self._callback = callbackFunction

    def graphCallback(self, ddict=None):
        """
        This callback is foing to receive all the events from the plot.
        Those events will consist on a dictionnary and among the dictionnary
        keys the key 'event' is madatory to describe the type of event.
        """

        if ddict is None:
            ddict = {}
        if DEBUG:
            print("Received dict keys = ", ddict.keys())
            print(ddict)
        if ddict['event'] in ["legendClicked", "curveClicked"]:
            if ddict['button'] == "left":
                self.setActiveCurve(ddict['label'])
        if self._callback is not None:
            self._callback(ddict)
    
    def setDefaultPlotPoints(self, flag):
        if flag:
            self._plotPoints = True
        else:
            self._plotPoints = False
        for key in self._curveList:
            del self._curveDict[key][3]['plot_symbol']
        if len(self._curveList):
            self._update()

    def setDefaultPlotLines(self, flag):
        if flag:
            self._plotLines = True
        else:
            self._plotLines = False
        if len(self._curveList):
            self._update()

    def _getColorAndStyle(self):
        self._lastColorIndex = self._colorIndex
        self._lastStyleIndex = self._styleIndex
        color = self._colorList[self._colorIndex]
        style = self._styleList[self._styleIndex]
        self._colorIndex += 1
        if self._colorIndex >= self._nColors:
            self._colorIndex = 1
            self._styleIndex += 1
            if self._styleIndex >= self._nStyles:
                self._styleIndex = 0        
        return color, style

    def setZoomModeEnabled(self, flag=True):
        self._plot.setZoomModeEnabled(flag)

    def setDrawModeEnabled(self, flag=True):
        self._plot.setDrawModeEnabled(flag)

    def addCurve(self, x, y, legend=None, info=None, replace=False,
                 replot=True, **kw):
        """
        Add the 1D curve given by x an y to the graph.
        :param x: The data corresponding to the x axis
        :type x: list or numpy.ndarray
        :param y: The data corresponding to the y axis
        :type y: list or numpy.ndarray
        :param legend: The legend to be associated to the curve
        :type legend: string or None
        :param info: Dictionary of information associated to the curve
        :type info: dict or None
        :param replace: Flag to indicate if already existing curves are to be deleted
        :type replace: boolean default False
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        """
        if legend is None:
            key = "Unnamed curve 1.1"
        else:
            key = str(legend)
        if info is None:
            info = {}
        xlabel = info.get('xlabel', 'X')
        ylabel = info.get('ylabel', 'Y')
        if 'xlabel' in kw:
            info['xlabel'] = kw['xlabel']
        if 'ylabel' in kw:
            info['ylabel'] = kw['ylabel']
        info['xlabel'] = str(xlabel)
        info['ylabel'] = str(ylabel)

        if replace:
            self._curveList = []
            self._curveDict = {}
            self._colorIndex = 0
            self._styleIndex = 0
            self._plot.clearCurves()

        if key in self._curveList:
            idx = self._curveList.index(key)
            self._curveList[idx] = key
            handle = self._curveDict[key][3].get('plot_handle', None)
            if handle is not None:
                # this can give errors if it is not present in the plot
                # self._plot.removeCurve(handle, replot=False)
                # this is safer
                self._plot.removeCurve(key, replot=False)
        else:
            self._curveList.append(key)
        #print("TODO: Here we can add properties to the info dictionnary")
        #print("For instance, color, symbol, style and width if not present")
        #print("They could come in **kw")
        #print("The actual plotting stuff should only take care of handling")
        #print("logarithmic filtering if needed")
        # deal with the symbol
        symbol = None
        symbol = info.get("plot_symbol", symbol)
        symbol = kw.get("symbol", symbol)
        if self._plotPoints:
            symbol = 'o'
        info["plot_symbol"] = symbol
        color = info.get("plot_color", None)
        color = kw.get("color", color)

        line_style = info.get("plot_line_style", None)
        line_style = kw.get("line_style", line_style)

        if not self._plotLines:
            line_style = ' '
        elif line_style == ' ':
            line_style = '-'

        if (color is None) and (line_style is None):
            color, line_style = self._getColorAndStyle()
        elif line_style is None:
            dummy, line_style = self._getColorAndStyle()
        elif color is None:
            color, dummy = self._getColorAndStyle()

        info["plot_color"] = color
        info["plot_line_style"] = line_style

        if self.isXAxisLogarithmic() or self.isYAxisLogarithmic():
            xplot, yplot = self.logFilterData(x, y)
        else:
            xplot, yplot = x, y
        if len(xplot):
            curveHandle = self._plot.addCurve(xplot, yplot, key, info,
                                              replot=False, replace=replace)
            info['plot_handle'] = curveHandle
        else:
            info['plot_handle'] = key
        self._curveDict[key] = [x, y, key, info]
        if len(self._curveList) == 1:
            self.setActiveCurve(key)
        if replot:
            self.resetZoom()
            self.replot()
        return legend

    def addImage(self, data, legend=None, info=None,
                 replace=True, replot=True,
                 xScale=None, yScale=None, z=0,
                 selectable=False, draggable=False, **kw):
        """
        :param data: (nrows, ncolumns) data or (nrows, ncolumns, RGBA) ubyte array 
        :type data: numpy.ndarray
        :param legend: The legend to be associated to the curve
        :type legend: string or None
        :param info: Dictionary of information associated to the image
        :type info: dict or None
        :param replace: Flag to indicate if already existing images are to be deleted
        :type replace: boolean default True
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        :param xScale: Two floats defining the x scale
        :type xScale: list or numpy.ndarray
        :param yScale: Two floats defining the y scale
        :type yScale: list or numpy.ndarray
        :param z: level at which the image is to be located (to allow overlays).
        :type z: A number bigger than or equal to zero (default)
        :param selectable: Flag to indicate if the image can be selected
        :type selectable: boolean, default False
        :param draggable: Flag to indicate if the image can be moved
        :type draggable: boolean, default False
        :returns: The legend/handle used by the backend to univocally access it.
        """
        if legend is None:
            key = "Unnamed curve 1.1"
        else:
            key = str(legend)
        if info is None:
            info = {}
        xlabel = info.get('xlabel', 'Column')
        ylabel = info.get('ylabel', 'Row')
        if 'xlabel' in kw:
            info['xlabel'] = kw['xlabel']
        if 'ylabel' in kw:
            info['ylabel'] = kw['ylabel']
        info['xlabel'] = str(xlabel)
        info['ylabel'] = str(ylabel)

        if replace:
            self._imageList = []
            self._imageDict = {}
        if data is not None:
            imageHandle = self._plot.addImage(data, legend=key, info=info,
                                              replot=False, replace=replace,
                                              xScale=xScale, yScale=yScale,
                                              z=z,
                                              selectable=selectable,
                                              draggable=draggable,
                                              **kw)
            info['plot_handle'] = imageHandle
        else:
            info['plot_handle'] = key
        self._imageDict[key] = [data, key, info, xScale, yScale, z]
        if len(self._imageDict) == 1:
            self.setActiveImage(key)
        if replot:
            self.resetZoom()
            self.replot()
        return key

    def removeCurve(self, legend, replot=True):
        """
        Remove the curve associated to the supplied legend from the graph.
        The graph will be updated if replot is true.
        :param legend: The legend associated to the curve to be deleted
        :type legend: string or None
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True        
        """
        if legend is None:
            return
        if legend in self._curveList:
            idx = self._curveList.index(legend)
            del self._curveList[idx]
        if legend in self._curveDict:
            handle = self._curveDict[legend][3].get('plot_handle', None)
            del self._curveDict[legend]
            if handle is not None:
                self._plot.removeCurve(handle, replot=replot)

    def removeImage(self, legend, replot=True):
        """
        Remove the image associated to the supplied legend from the graph.
        The graph will be updated if replot is true.
        :param legend: The legend associated to the image to be deleted
        :type legend: string or handle
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True        
        """
        if legend is None:
            return
        if legend in self._imageList:
            idx = self._imageList.index(legend)
            del self._imageList[idx]
        if legend in self._imageDict:
            handle = self._imageDict[legend][2].get('plot_handle', None)
            del self._imageDict[legend]
            if handle is not None:
                self._plot.removeImage(handle, replot=replot)
        return

    def getActiveCurve(self, just_legend=False):
        """
        :param just_legend: Flag to specify the type of output required
        :type just_legend: boolean
        :return: legend of the active curve or list [x, y, legend, info]
        :rtype: string or list 
        Function to access the graph currently active curve.
        It returns None in case of not having an active curve.

        Default output has the form:
            xvalues, yvalues, legend, dict
            where dict is a dictionnary containing curve info.
            For the time being, only the plot labels associated to the
            curve are warranted to be present under the keys xlabel, ylabel.

        If just_legend is True:
            The legend of the active curve (or None) is returned.
        """
        if self._activeCurve not in self._curveDict:
            self._activeCurve = None
        if just_legend:
            return self._activeCurve
        if self._activeCurve is None:
            return None
        else:
            return self._curveDict[self._activeCurve] * 1

    def getActiveImage(self, just_legend=False):
        """
        :param just_legend: Flag to specify the type of output required
        :type just_legend: boolean
        :return: legend of the active image or list [data, legend, info, xScale, yScale, z]
        :rtype: string or list 
        Function to access the plot currently active image.
        It returns None in case of not having an active image.

        Default output has the form:
            data, legend, dict, xScale, yScale, z
            where dict is a dictionnary containing image info.
            For the time being, only the plot labels associated to the
            image are warranted to be present under the keys xlabel, ylabel.

        If just_legend is True:
            The legend of the active imagee (or None) is returned.
        """
        if self._activeImage not in self._imageDict:
            self._activeImage = None
        if just_legend:
            return self._activeImage
        if self._activeImage is None:
            return None
        else:
            return self._imageDict[self._activeImage] * 1

    def getAllCurves(self, just_legend=False):
        """
        :param just_legend: Flag to specify the type of output required
        :type just_legend: boolean
        :return: legend of the curves or list [[x, y, legend, info], ...]
        :rtype: list of strings or list of curves 

        It returns an empty list in case of not having any curve.
        If just_legend is False:
            It returns a list of the form:
                [[xvalues0, yvalues0, legend0, dict0],
                 [xvalues1, yvalues1, legend1, dict1],
                 [...],
                 [xvaluesn, yvaluesn, legendn, dictn]]
            or just an empty list.
        If just_legend is True:
            It returns a list of the form:
                [legend0, legend1, ..., legendn]
            or just an empty list.
        """
        output = []
        keys = list(self._curveDict.keys())
        for key in self._curveList:
            if key in keys:
                if just_legend:
                    output.append(key)
                else:
                    output.append(self._curveDict[key])
        return output

    def _getAllLimits(self):
        """
        Internal method to retrieve the limits based on the curves, not
        on the plot. It might be of use to reset the zoom when one of the
        X or Y axesis not set to autoscale.
        """
        keys = list(self._curveDict.keys())
        if not len(keys):
            return 0.0, 0.0, 100., 100.
        xmin = None
        ymin = None
        xmax = None
        ymax = None
        for key in keys:
            x = self._curveDict[key][0]
            y = self._curveDict[key][1]
            if xmin is None:
                xmin = x.min()
            else:
                xmin = min(xmin, x.min())
            if ymin is None:
                ymin = y.min()
            else:
                ymin = min(ymin, y.min())
            if xmax is None:
                xmax = x.max()
            else:
                xmax = max(xmax, x.max())
            if ymax is None:
                ymax = y.max()
            else:
                ymax = max(ymax, y.max())
        return xmin, ymin, xmax, ymax

    def setActiveCurve(self, legend, replot=True):
        """
        Funtion to request the plot window to set the curve with the specified legend
        as the active curve.
        :param legend: The legend associated to the curve
        :type legend: string
        """
        oldActiveCurve = self.getActiveCurve(just_legend=True)
        key = str(legend)
        if key in self._curveDict.keys():
            self._activeCurve = key
        # this was giving troubles in the PyQtGraph binding
        #if self._activeCurve != oldActiveCurve:
        self._plot.setActiveCurve(self._activeCurve, replot=replot)
        return self._activeCurve

    def setActiveImage(self, legend, replot=True):
        """
        Funtion to request the plot window to set the image with the specified legend
        as the active image.
        :param legend: The legend associated to the image
        :type legend: string
        """
        oldActiveImage = self.getActiveImage(just_legend=True)
        key = str(legend)
        if key in self._imageDict.keys():
            self._activeImage = key
        self._plot.setActiveImage(self._activeImage, replot=replot)
        return self._activeImage

    def invertYAxis(self, flag=True):
        self._plot.invertYAxis(flag)

    def isYAxisLogarithmic(self):
        if self._logY:
            return True
        else:
            return False

    def isXAxisLogarithmic(self):
        if self._logX:
            return True
        else:
            return False
        
    def setYAxisLogarithmic(self, flag):
        if flag:
            if self._logY:
                if DEBUG:
                    print("y axis was already in log mode")
            else:
                self._logY = True
                if DEBUG:
                    print("y axis was in linear mode")
                self._plot.clearCurves()
                self._plot.setYAxisLogarithmic(self._logY)
                self._update()
        else:
            if self._logY:
                if DEBUG:
                    print("y axis was in log mode")
                self._logY = False
                self._plot.clearCurves()
                self._plot.setYAxisLogarithmic(self._logY)
                self._update()
            else:
                if DEBUG:
                    print("y axis was already linear mode")
        return

    def setXAxisLogarithmic(self, flag):
        if flag:
            if self._logX:
                if DEBUG:
                    print("x axis was already in log mode")
            else:
                self._logX = True
                if DEBUG:
                    print("x axis was in linear mode")
                self._plot.clearCurves()
                self._plot.setXAxisLogarithmic(self._logX)
                self._update()
        else:
            if self._logX:
                if DEBUG:
                    print("x axis was in log mode")
                self._logX = False
                self._plot.setXAxisLogarithmic(self._logX)
                self._update()
            else:
                if DEBUG:
                    print("x axis was already linear mode")
        return

    def logFilterData(self, x, y, xLog=None, yLog=None):
        if xLog is None:
            xLog = self._logX
        if yLog is None:
            yLog = self._logY

        if xLog and yLog:
            idx = numpy.nonzero((x > 0) & (y > 0))[0]
            x = numpy.take(x, idx)
            y = numpy.take(y, idx)
        elif yLog:
            idx = numpy.nonzero(y > 0)[0]
            x = numpy.take(x, idx)
            y = numpy.take(y, idx)
        elif xLog:
            idx = numpy.nonzero(x > 0)[0]
            x = numpy.take(x, idx)
            y = numpy.take(y, idx)
        return x, y

    def _update(self):
        curveList = self.getAllCurves()
        activeCurve = self.getActiveCurve(just_legend=True)
        #self._plot.clearCurves()
        for curve in curveList:
            x, y, legend, info = curve[0:4]
            self.addCurve(x, y, legend, info=info,
                          replace=False, replot=False)
        if len(curveList):
            if activeCurve not in curveList:
                activeCurve = curveList[0][2]
            print("setting active Curve", activeCurve)
            self.setActiveCurve(activeCurve)
        self.replot()

    def replot(self):
        if self.isXAxisLogarithmic() or self.isYAxisLogarithmic():
            for image in self._imageDict.keys():
                self._plot.removeImage(image[1]) 
        if hasattr(self._plot, 'replot_'):
            plot = self._plot.replot_
        else:
            plot = self._plot.replot
        

    def clear(self):
        self._curveList = []
        self._curveDict = {}
        self._colorIndex = 1
        self._styleIndex = 0
        self._markerDict = {}
        self._imageList = []
        self._imageDict = {}        
        self._markerList = []
        self._plot.clear()
        self.replot()

    def clearCurves(self):
        self._curveList = []
        self._curveDict = {}
        self._colorIndex = 0
        self._styleIndex = 0
        self._plot.clearCurves()
        self.replot()

    def clearImages(self):
        """
        Clear all images from the plot. Not the curves or markers.
        """
        self._imageList = []
        self._imageDict = {}
        self._plot.clearImages()
        self.replot()
        return

    def resetZoom(self):
        self._plot.resetZoom()

    def setXAxisAutoScale(self, flag=True):
        self._plot.setXAxisAutoScale(flag)

    def setYAxisAutoScale(self, flag=True):
        self._plot.setYAxisAutoScale(flag)

    def isXAxisAutoScale(self):
        return self._plot.isXAxisAutoScale()

    def isYAxisAutoScale(self):
        return self._plot.isYAxisAutoScale()

    def setGraphYLimits(self, ymin, ymax, replot=False):
        self._plot.setGraphYLimits(ymin, ymax)
        if replot:
            self.replot()

    def setGraphXLimits(self, xmin, xmax, replot=False):
        self._plot.setGraphXLimits(xmin, xmax)
        if replot:
            self.replot()

    def getGraphXLimits(self):
        """
        Get the graph X (bottom) limits.
        :return:  Minimum and maximum values of the X axis
        """
        if hasattr(self._plot, "getGraphXLimits"):
            xmin, xmax = self._plot.getGraphXLimits()
        else:
            xmin, ymin, xmax, ymax = self._getAllLimits()
        return xmin, xmax

    def getGraphYLimits(self):
        """
        Get the graph Y (left) limits.
        :return:  Minimum and maximum values of the X axis
        """
        if hasattr(self._plot, "getGraphYLimits"):
            ymin, ymax = self._plot.getGraphYLimits()
        else:
            xmin, ymin, xmax, ymax = self._getAllLimits()
        return ymin, ymax

    # Title and labels
    def setGraphTitle(self, title=""):
        self._plot.setGraphTitle(title)

    def setGraphXLabel(self, label="X"):
        self._plot.setGraphXLabel(label)

    def setGraphYLabel(self, label="Y"):
        self._plot.setGraphYLabel(label)
        
    # Marker handling
    def insertXMarker(self, x, label=None,
                     color=None,
                     selectable=False,
                     draggable=False,
                     **kw):
        """
        kw ->symbol
        """
        if color is None:
            color = colordict['black']
        elif color in colordict:
            color = colordict[color]
        if label is None:
            i = 0
            label = "Unnamed X Marker %d" % i
            while label in self._markerList:
                i += 1
                label = "Unnamed X Marker %d" % i

        if label in self._markerList:
            self.clearMarker(label)
        marker = self._plot.insertXMarker(x, label,
                                          color=color,
                                          selectable=selectable,
                                          draggable=draggable,
                                          **kw)
        self._markerList.append(label)
        self._markerDict[label] = kw
        self._markerDict[label]['marker'] = marker

    def insertYMarker(self, y, label=None,
                     color=None,
                     selectable=False,
                     draggable=False,
                     **kw):
        """
        kw -> color, symbol
        """
        if color is None:
            color = colordict['black']
        elif color in colordict:
            color = colordict[color]
        if label is None:
            i = 0
            label = "Unnamed Y Marker %d" % i
            while label in self._markerList:
                i += 1
                label = "Unnamed Y Marker %d" % i
        if label in self._markerList:
            self.clearMarker(label)
        marker = self._plot.insertYMarker(y, label,
                                          color=color,
                                          selectable=selectable,
                                          draggable=draggable,
                                          **kw)
        self._markerList.append(label)
        self._markerDict[label] = kw
        self._markerDict[label]['marker'] = marker

    def insertMarker(self, x, y, label=None,
                     color=None,
                     selectable=False,
                     draggable=False,
                     **kw):
        if color is None:
            color = colordict['black']
        elif color in colordict:
            color = colordict[color]
        if label is None:
            i = 0
            label = "Unnamed Marker %d" % i
            while label in self._markerList:
                i += 1
                label = "Unnamed Marker %d" % i

        if label in self._markerList:
            self.clearMarker(label)
        marker = self._plot.insertMarker(x, y, label,
                                          color=color,
                                          selectable=selectable,
                                          draggable=draggable,
                                          **kw)
        self._markerList.append(label)
        self._markerDict[label] = kw
        self._markerDict[label]['marker'] = marker

    def clearMarkers(self):
        self._markerDict = {}
        self._markerList = []
        self._plot.clearMarkers()

    def removeMarker(self, marker):
        if marker in self._markerList:
            idx = self._markerList.index(marker)
            self._plot.removeMarker(self._markerDict[marker]['marker'])
            del self._markerDict[marker]

    def setMarkerFollowMouse(self, marker, boolean):
        raise NotImplemented("Not necessary?")
        if marker not in self._markerList:
            raise ValueError("Marker %s not defined" % marker)
        pass

    def enableMarkerMode(self, flag):
        raise NotImplemented("Not necessary?")
        pass

    def isMarkerModeEnabled(self, flag):
        raise NotImplemented("Not necessary?")
        pass

def main():
    import numpy
    x = numpy.arange(100.)
    y = x * x
    plot = Plot()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x + 100, -x * x, "To set Active")
    print("Active curve = ", plot.getActiveCurve())
    print("X Limits = ", plot.getGraphXLimits())
    print("Y Limits = ", plot.getGraphYLimits())
    print("All curves = ", plot.getAllCurves())
    plot.removeCurve("dummy")
    plot.setActiveCurve("To set Active")
    print("All curves = ", plot.getAllCurves())
    plot.insertXMarker(50.)

if __name__ == "__main__":
    main()
