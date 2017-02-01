#/*##########################################################################
# Copyright (C) 2004-2017 V.A. Sole, European Synchrotron Radiation Facility
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
"""This module defines a :class:`PlotProxySilx` class that provides a legacy
PyMca API for a *silx* plot widget.

For this, the proxy must ensure that the following list of public plot methods
use the signature defined in PyMca's :class:`PlotBase`.

    - getActiveCurve
    - getActiveImage
    - getAllCurves
    - getCurve
    - getImage
    - getMonotonicCurves
    - hideCurve
    - hideImage
    - isActiveCurveHandlingEnabled
    - isCurveHidden
    - isImageHidden
    - printGraph
    - setActiveCurve
    - showCurve
    - showImage

"""

# FIXME: according to PlotBase, these should exist, but can't find them anywhere:
#  - isImageHidden
#  - hideImage

import numpy
import sys

import PyMca5.PyMcaGui.PyMcaQt as qt

if sys.version_info[0] == 3:
    from io import BytesIO
else:
    import cStringIO as _StringIO
    BytesIO = _StringIO.StringIO


def _plotAsPNG(plot):
    """Save a :class:`silx.gui.plot.Plot` as PNG and return the payload.

    :param plot: The :class:`Plot` to save
    """
    pngFile = BytesIO()
    plot.saveGraph(pngFile, fileFormat='png')
    pngFile.flush()
    pngFile.seek(0)
    data = pngFile.read()
    pngFile.close()
    return data


def merge_dicts(dict1, dict2):
    """Return a copy of dict1 updated with dict2.

    If keys don't overlap, this will effectively concatenate dict1 and
    dict2."""
    dict3 = dict1.copy()
    dict3.update(dict2)
    return dict3


class PlotProxySilx(object):
    """Proxy object to use *silx* plot widgets using the PyMca API
    (for PyMca legacy plugins compatibility)"""
    def __init__(self, silx_plot):
        """
        :param silx_plot_instance: *silx* PlotWidget
        """
        self.silx_plot = silx_plot

    def __getattr__(self, attr):
        # method called for attributes/methods not explicitly overloaded
        methods_not_overloaded = [
            "isCurveHidden",
            "replot"
        ]
        assert attr in methods_not_overloaded
        return getattr(self.silx_plot, attr)

    # methods with different number of return values
    def getCurve(self, legend):
        """Turn silx's ``getCurve`` return values
        ``[x, y, legend, info, parameters]`` into
        ``[x, y, legend, info]``"""
        curve = self.silx_plot.getCurve(legend)
        if curve is None:
            return None
        x, y, legend, info, parameters = curve
        return x, y, legend, merge_dicts(info, parameters)

    def getActiveCurve(self, just_legend=False):
        """Turn silx's ``getActiveCurve`` return values
        ``[x, y, legend, info, parameters]`` into
        ``[x, y, legend, info]``"""
        curve = self.silx_plot.getActiveCurve(just_legend)
        if curve is None:
            return None
        x, y, legend,  info, params = curve
        if just_legend:
            return legend
        return x, y, legend, merge_dicts(info, params)

    def getAllCurves(self, just_legend=False):
        """Turn silx's ``getAllCurves`` return values
        ``[[x1, y1, legend1, info1, parameters1], ...]`` into
        ``[[x1, y1, legend1, info1], ...]``"""
        all_curves = []
        for x, y, legend, info, params in self.silx_plot.getAllCurves(just_legend):
            if just_legend:
                all_curves.append(legend)
            else:
                all_curves.append([x, y, legend, merge_dicts(info, params)])
        return all_curves

    def getActiveImage(self, just_legend=False):
        """Turn silx's ``getActiveImage`` return values
        ``[data, legend, info, pixmap, params]`` into
        ``[data, legend, dict, pixmap]``"""
        image = self.silx_plot.getActiveImage(just_legend)
        if image is None:
            return None
        data, legend, info, pixmap, params = image
        if just_legend:
            return legend
        return data, legend, merge_dicts(info, params), pixmap

    def getImage(self, legend=None):
        """Turn silx's ``getImage`` return values
        ``[data, legend, info, pixmap, params]`` into
        ``[data, legend, dict, pixmap]``"""
        image = self.silx_plot.getImage(legend)
        if image is None:
            return None
        data, legend, info, pixmap, params = image
        return data, legend, merge_dicts(info, params), pixmap


    # deprecated replot parameter
    def removeCurve(self, legend, replot=True):
        ret = self.silx_plot.removeCurve(legend)
        if replot:
            self.replot()
        return ret

    def hideCurve(self, legend, flag=True, replot=None):
        ret = self.silx_plot.hideCurve(legend, flag)
        if replot:
            self.replot()
        return ret

    def setActiveCurve(self, legend, replot=None):
        ret = self.silx_plot.setActiveCurve(legend)
        if replot:
            self.replot()
        return ret

    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True,
                 color=None, symbol=None, linestyle=None,
                 xlabel=None, ylabel=None, yaxis=None,
                 xerror=None, yerror=None, z=None, selectable=None, **kw):
        # replot was replaced by resetzoom
        fill = info.get("plot_fill", False) if info is not None else False
        fill = kw.get("fill", fill)
        linewidth = kw.get("linewidth", None)
        return self.silx_plot.addCurve(
                x, y, legend=legend, info=info, replace=replace,
                color=color, symbol=symbol, linewidth=linewidth,
                linestyle=linestyle, xlabel=xlabel, ylabel=ylabel,
                yaxis=yaxis, xerror=xerror, yerror=yerror, z=z,
                selectable=selectable, fill=fill, resetzoom=replot, **kw)

    def isActiveCurveHandlingEnabled(self):
        # isActiveCurveHandlingEnabled method deprecated in silx
        return self.silx_plot.isActiveCurveHandling()

    # not implemented in silx
    def getMonotonicCurves(self):
        allCurves = self.getAllCurves() * 1
        for i in range(len(allCurves)):
            curve = allCurves[i]
            x, y, legend, info = curve[0:4]
            if self.isCurveHidden(legend):
                continue
            # Sort
            idx = numpy.argsort(x, kind='mergesort')
            xproc = numpy.take(x, idx)
            yproc = numpy.take(y, idx)
            # Ravel, Increase
            xproc = xproc.ravel()
            idx = numpy.nonzero((xproc[1:] > xproc[:-1]))[0]
            xproc = numpy.take(xproc, idx)
            yproc = numpy.take(yproc, idx)
            allCurves[i][0:2] = xproc, yproc
        return allCurves

    def printGraph(self, *args, **kwargs):
        """Trigger plot.printAction if available"""
        # unlikely case of custom plot explicitly defining printGraph
        if hasattr(self.silx_plot, "printGraph"):
            self.silx_plot.printGraph(*args, **kwargs)

        # case of plot window defining a plotAction
        if hasattr(self.silx_plot, "getPrintAction"):
            printAction = self.silx_plot.getPrintAction()
            printAction.trigger()
            return

        # general case: code based on plotAction, relies on plot.saveGraph
        printer = qt.QPrinter()
        dialog = qt.QPrintDialog(printer, self.silx_plot)
        dialog.setWindowTitle('Print Plot')
        if not dialog.exec_():
            return False

        # Save Plot as PNG and make a pixmap from it with default dpi
        pngData = _plotAsPNG(self.silx_plot)

        pixmap = qt.QPixmap()
        pixmap.loadFromData(pngData, 'png')

        xScale = printer.pageRect().width() / pixmap.width()
        yScale = printer.pageRect().height() / pixmap.height()
        scale = min(xScale, yScale)

        # Draw pixmap with painter
        painter = qt.QPainter()
        if not painter.begin(printer):
            return False

        painter.drawPixmap(0, 0,
                           pixmap.width() * scale,
                           pixmap.height() * scale,
                           pixmap)
        painter.end()

        return True

    # not implemented in silx nor PyMca
    def showCurve(self, legend, replot=True):
        self.silx_plot.hideCurve(legend, flag=False)
        if replot:
            self.replot()

    def showImage(self, legend, replot=True):
        # TODO: find out if this can be done in silx
        # unlikely case of custom plot explicitly defining showImage
        if hasattr(self.silx_plot, "showImage"):
            self.silx_plot.showImage(legend)
            if replot:
                self.replot()
            return
        print("silx plot proxy: showImage not implemented")

    def hideImage(self, legend, replot=True):
        # TODO: find out if this can be done in silx
        # unlikely case of custom plot explicitly defining method
        if hasattr(self.silx_plot, "hideImage"):
            self.silx_plot.hideImage(legend)
            if replot:
                self.replot()
            return
        print("silx plot proxy: hideImage  not implemented")

    def isImageHidden(self, legend):
        # TODO: find out if this can be done in silx
        # unlikely case of custom plot explicitly defining method
        if hasattr(self.silx_plot, "isImageHidden"):
            return self.silx_plot.isImageHidden(legend)
        print("silx plot proxy: isImageHidden not implemented, returning False")
        return False
