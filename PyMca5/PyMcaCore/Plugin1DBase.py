#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
A 1D plugin is a module that can be added to the PyMca 1D window in order to
perform user defined operations of the plotted 1D data.

Plugins can be automatically installed provided they are in the appropriate place:

    - In the user home directory: ${HOME}/PyMca/plugins (POSIX systems)
    - In "My Documents\\\\PyMca\\\\plugins" (Windows)

A plugin inherit the Plugin1DBase.Plugin1DBase class and implement the methods:

    - getMethods
    - getMethodToolTip (optional but convenient)
    - getMethodPixmap (optional)
    - applyMethod

and modify the static module variable MENU_TEXT and the static module function
getPlugin1DInstance according to the defined plugin.

These plugins will be compatible with any 1D-plot window that implements the Plot1D
interface. The plot window interface is described in the Plot1DBase class.

The main items are reproduced here and can be directly accessed as plugin methods.

    - addCurve
    - getActiveCurve
    - getAllCurves
    - getGraphXLimits
    - getGraphYLimits
    - getGraphTitle
    - getGraphXLabel
    - getGraphXTitle
    - getGraphYLabel
    - getGraphYTitle
    - removeCurve
    - setActiveCurve
    - setGraphTitle
    - setGraphXLimits
    - setGraphYLimits
    - setGraphXLabel
    - setGraphYLabel
    - setGraphXTitle
    - setGraphYTitle

A simple plugin example, normalizing each curve to its maximum and vertically
shifting the curves.

.. code-block:: python

    from PyMca5 import Plugin1DBase

    class Shifting(Plugin1DBase.Plugin1DBase):
        def getMethods(self, plottype=None):
            return ["Shift"]

        def getMethodToolTip(self, methodName):
            if methodName != "Shift":
                raise InvalidArgument("Method %s not valid" % methodName)
            return "Subtract minimum, normalize to maximum, and shift up by 0.1"

        def applyMethod(self, methodName):
            if methodName != "Shift":
                raise ValueError("Method %s not valid" % methodName)
            allCurves = self.getAllCurves()
            increment = 0.1
            for i in range(len(allCurves)):
                x, y, legend, info = allCurves[i][:4]
                delta = float(y.max() - y.min())
                if delta < 1.0e-15:
                    delta = 1.0
                y = (y - y.min())/delta + i * increment
                if i == (len(allCurves) - 1):
                    replot = True
                else:
                    replot = False
                if i == 0:
                    replace = True
                else:
                    replace = False
                self.addCurve(x, y, legend=legend + " %.2f" % (i * increment),
                                    info=info, replace=replace, replot=replot)
     
    MENU_TEXT="Simple Shift Example"
    def getPlugin1DInstance(plotWindow, **kw):
        ob = Shifting(plotWindow)
        return ob

"""
import weakref

try:
    from numpy import argsort, nonzero, take
except ImportError:
    print("WARNING: numpy not present")

class Plugin1DBase(object):
    def __init__(self, plotWindow, **kw):
        """
        plotWindow is the object instantiating the plugin.

        Unless one knows what (s)he is doing, only a proxy should be used.

        I pass the actual instance to keep all doors open.
        """
        self._plotWindow = weakref.proxy(plotWindow)

    #Window related functions
    def windowTitle(self):
        name = self._plotWindow.windowTitle()
        try:
            name = self._plotWindow.windowTitle()
        except:
            name = ""
        return name

    def addCurve(self, x, y, legend=None, info=None,
                 replace=False, replot=True, **kw):
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
        :param **kw: Additional keywords recognized by the plot window
        """
        return self._plotWindow.addCurve(x, y, legend=legend, info=info,
                                    replace=replace, replot=replot, **kw)

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
        return self._plotWindow.getActiveCurve(just_legend=just_legend)

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
        return self._plotWindow.getAllCurves(just_legend=just_legend)

    def getMonotonicCurves(self):
        """
        Convenience method that calls getAllCurves and makes sure that all of
        the X values are strictly increasing.

        :return: It returns a list of the form:
                [[xvalues0, yvalues0, legend0, dict0],
                 [xvalues1, yvalues1, legend1, dict1],
                 [...],
                 [xvaluesn, yvaluesn, legendn, dictn]]
        """
        allCurves = self.getAllCurves() * 1
        for i in range(len(allCurves)):
            curve = allCurves[i]
            x, y, legend, info = curve[0:4]
            # Sort
            idx = argsort(x, kind='mergesort')
            xproc = take(x, idx)
            yproc = take(y, idx)
            # Ravel, Increase
            xproc = xproc.ravel()
            idx = nonzero((xproc[1:] > xproc[:-1]))[0]
            xproc = take(xproc, idx)
            yproc = take(yproc, idx)
            allCurves[i][0:2] = xproc, yproc
        return allCurves

    def getGraphXLimits(self):
        """
        Get the graph X limits.

        :return: Two floats with the X axis limits
        """
        return self._plotWindow.getGraphXLimits()

    def getGraphYLimits(self):
        """
        Get the graph Y (left) limits.

        :return: Two floats with the Y (left) axis limits
        """
        return self._plotWindow.getGraphYLimits()

    def getGraphTitle(self):
        """
        :return: The graph title
        :rtype: string
        """
        return self._plotWindow.getGraphTitle()

    def getGraphXLabel(self):
        """
        :return: The graph X axis label
        :rtype: string
        """
        return self._plotWindow.getGraphXLabel()

    def getGraphXTitle(self):
        """
        :return: The graph X axis label
        :rtype: string
        """
        print("getGraphXTitle deprecated, use getGraphXLabel")
        return self._plotWindow.getGraphXLabel()

    def getGraphYLabel(self):
        """
        :return: The graph Y axis label
        :rtype: string
        """
        return self._plotWindow.getGraphYLabel()

    def getGraphYTitle(self):
        """
        :return: The graph Y axis label
        :rtype: string
        """
        print("getGraphYTitle deprecated, use getGraphYLabel")
        return self._plotWindow.getGraphYLabel()

    def setGraphXLimits(self, xmin, xmax, replot=False):
        """
        Set the graph X limits.

        :param xmin:  minimum value of the axis
        :type xmin: float
        :param xmax:  minimum value of the axis
        :type xmax: float
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default False
        """
        return self._plotWindow.setGraphXLimits(xmin, xmax, replot=replot)

    def setGraphYLimits(self, ymin, ymax, replot=False):
        """
        Set the graph Y (left) limits.

        :param ymin:  minimum value of the axis
        :type ymin: float
        :param ymax:  minimum value of the axis
        :type ymax: float
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default False
        """
        return self._plotWindow.setGraphYLimits(ymin, ymax, replot=replot)

    def removeCurve(self, legend, replot=True):
        """
        Remove the curve associated to the supplied legend from the graph.
        The graph will be updated if replot is true.

        :param legend: The legend associated to the curve to be deleted
        :type legend: string or None
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        """
        return self._plotWindow.removeCurve(legend, replot=replot)

    def setActiveCurve(self, legend):
        """
        Funtion to request the plot window to set the curve with the specified legend
        as the active curve.

        :param legend: The legend associated to the curve
        :type legend: string
        """
        return self._plotWindow.setActiveCurve(legend)

    def setGraphTitle(self, title):
        """
        :param title: The title to be set
        :type title: string
        """
        return self._plotWindow.setGraphTitle(title)

    def setGraphXTitle(self, title):
        """
        :param title: The title to be associated to the X axis
        :type title: string
        """
        return self._plotWindow.setGraphXTitle(title)

    def setGraphYTitle(self, title):
        """
        :param title: The title to be associated to the X axis
        :type title: string
        """
        return self._plotWindow.setGraphYTitle(title)

    #Methods to be implemented by the plugin
    def getMethods(self, plottype=None):
        """
        :param plottype: string or None for the case the plugin only support
         one type of plots. Implemented values "SCAN", "MCA" or None
        :return:  A list with the NAMES  associated to the callable methods
         that are applicable to the specified type plot. The list can be empty.
        :rtype: list[string]
        """
        print("getMethods not implemented")
        return []

    def getMethodToolTip(self, name):
        """
        Returns the help associated to the particular method name or None.

        :param name: The method for which a tooltip is asked
        :rtype: string
        """
        return None

    def getMethodPixmap(self, name):
        """
        :param name: The method for which a pixmap is asked
        :rtype: QPixmap or None
        """
        return None

    def applyMethod(self, name):
        """
        The plugin is asked to apply the method associated to name.
        """
        print("applyMethod not implemented")
        return

MENU_TEXT = "Plugin1D Base"
def getPlugin1DInstance(plotWindow, **kw):
    """
    This function will be called by the plot window instantiating and calling
    the plugins. It passes itslef as first argument, but the default implementation
    of the base class only keeps a weak reference to prevent cirvular references.
    """
    ob = Plugin1DBase(plotWindow)
    return ob
