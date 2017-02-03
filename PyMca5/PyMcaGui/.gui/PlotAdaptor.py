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
"""This module defines a :class:`PlotAdaptor` class, inheriting from
a *silx* :class:`PlotWindow. Public methods are overloaded to make them
compatible with programs and plugins expecting a PyMca plot API, while
also providing compatibility with programs and plugins expecting a *silx*
plot API.
"""

import numpy
import logging
import sys
from silx.gui.plot import PlotWindow as SilxPlotWindow

import PyMca5.PyMcaGui.PyMcaQt as qt

if sys.version_info[0] == 3:
    from io import BytesIO
else:
    import cStringIO as _StringIO
    BytesIO = _StringIO.StringIO

_logger = logging.getLogger(__name__)


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


def merge_info_params(info, params):
    """Return a copy of dictionary`info` updated with the content
    of dictionary `params`.

    If keys don't overlap, this will effectively concatenate info and
    params.

    :param dict info: Dictionary of custom curve metadata
    :param dict params: Dictionary of standard curve metadata (*xlabel,
        ylabel, linestyle...*)"""
    # TODO. much more keys and values to fix (see McaWindow)
    info_params = info.copy()
    info_params.update(params)

    # xlabel and ylabel default to None in *silx*,
    # PyMca plugins expect a string
    if info_params.get("xlabel", None) is None:
        info_params["xlabel"] = "X"
    if info_params.get("ylabel", None) is None:
        info_params["ylabel"] = "Y"

    return info_params


# wrappers
def deprecateMethodIfNotLegacy(replacementMethod):
    def wrapper_generator(method):
        def method_wrapper(self, *args, **kwargs):
            if not self.legacy:
                _logger.warning(
                        method.__name__ + " is deprecated. " +
                        "Use %s instead." % replacementMethod
                )
            return method(self, *args, **kwargs)
        return method_wrapper
    return wrapper_generator


def deprecateReplotIfNotLegacy(method):
    """If replot is not None, and global flag :attr:`legacy` is not True,
    print a deprecation warning.
    If replot is None, set it to a default value (True for legacy, False for
    *silx* API)"""
    def wrapper(self, *args, **kwargs):
        replot = kwargs.get("replot", None)
        # print deprecation warning if replot is defined for a silx plot
        if replot is not None and not self.legacy:
            _logger.warning("replot parameter is deprecated")
        # set a default value (True if legacy PyMca plot)
        else:
            kwargs["replot"] = True if self.legacy else False
        return method(self, *args, **kwargs)
    return wrapper


class PlotAdapter(SilxPlotWindow):    # fixme: do we need to use PlotWidget to be more generic?
    """Compatibily layer between the *silx* plot API and the *PyMca* plot API.

    See :class:`silx.gui.plot.Plot` for the documentation of the *silx*
    plot API.
    See :class:`PyMca5.PyMcaGraph.PlotBackend` for the documentation of the *PyMca*
    plot API.

    This class has a :attr:`legacy` attribute, to control its behavior.
    If *legacy=True*, the overloaded methods behave like a *PyMca* plot.
    If *legacy=False*, the overloaded methods behave like a *silx* plot.
    """
    def __init__(self, *args, **kwargs):
        """

        """
        silx_keys = ["parent", "backend", "resetzoom", "autoScale",
                     "logScale", "grid", "curveStyle", "colormap",
                     "aspectRatio", "yInverted", "copy", "save",
                     "print", "control", "position", "roi", "mask",
                     "fit"]
        silx_kwargs = {silx_key: kwargs[silx_key] for silx_key in silx_keys}

        unexpected_kwargs = set(kwargs.keys()) - set(silx_keys + ["legacy"])
        if unexpected_kwargs:
            _logger.warning('Unexpected arguments for PlotAdapter.__init__: ' +   # fixme: _logger.error? raise?
                            " ".join(unexpected_kwargs))

        SilxPlotWindow.__init__(self, *args, **silx_kwargs)

        # PyMca attributes
        self._plotType = None

        # Additional attributes
        self.legacy = kwargs.get("legacy", True)
        """This attribute must be set to *True* if you want to globally use
        the *PyMca* legacy API, or *False* to globally use the newer *silx*
        plot API.

        This global behavior can be overwritten in all overloaded
        methods by passing a legacy parameter to the method.

        :meth:`setLegacy` should be used to change the value.
        The default value is currently *True* (use PyMca API), but this
        is likely to change in the future.
        """

    def setLegacy(self, flag):
        """Set :attr:`legacy` to True or False.

        :param bool flag: True or False
        """
        self.legacy = flag

    # methods with different number of return values
    def getCurve(self, legend=None, legacy=None):
        """*silx*: returns ``[x, y, legend, info, parameters]``.
        *PyMca*: returns ``[x, y, legend, info]``"""
        curve = SilxPlotWindow.getCurve(self, legend)
        # legacy not specified, default to global attribute
        if legacy is None:
            legacy = self.legacy
            _logger.warning(
                    "getCurve: using default legacy=%s parameter" % legacy)

        if not legacy or curve is None:
            return curve

        x, y, legend, info, parameters = curve
        return x, y, legend, merge_info_params(info, parameters)

    def getActiveCurve(self, just_legend=False, legacy=None):
        """*silx*: returns ``[x, y, legend, info, parameters]``.
        *PyMca*: returns ``[x, y, legend, info]``"""
        curve = SilxPlotWindow.getActiveCurve(self, just_legend)

        if legacy is None:
            legacy = self.legacy
            _logger.warning(
                    "getActiveCurve: using default legacy=%s parameter" % legacy)

        if (not legacy and not just_legend) or curve is None:
            return curve

        x, y, legend,  info, params = curve
        if just_legend:
            return legend

        return x, y, legend, merge_info_params(info, params)

    def getAllCurves(self, just_legend=False, legacy=None):
        """*silx*: returns ``[[x, y, legend, info, parameters], ...]``.
        *PyMca*: returns ``[[x, y, legend, info], ...]``"""
        if legacy is None:
            legacy = self.legacy
            _logger.warning(
                    "getAllCurves: using default legacy=%s parameter" % legacy)

        all_curves = []
        for x, y, legend, info, params in SilxPlotWindow.getAllCurves(self, just_legend):
            if just_legend:
                all_curves.append(legend)
            else:
                if legacy:
                    all_curves.append([x, y, legend, merge_info_params(info, params)])
                else:
                    all_curves.append([x, y, legend, info, params])
        return all_curves

    def getActiveImage(self, just_legend=False, legacy=None):
        """*silx*: return ``[data, legend, info, pixmap, params]`` into
        *PyMca*: return ``[data, legend, dict, pixmap]``"""
        image = SilxPlotWindow.getActiveImage(self, just_legend)
        if image is None:
            return None
        if legacy is None:
            legacy = self.legacy
            _logger.warning(
                    "getActiveImage: using default legacy=%s parameter" % legacy)

        data, legend, info, pixmap, params = image
        if just_legend:
            return legend
        if not legacy:
            return image
        return data, legend, merge_info_params(info, params), pixmap

    def getImage(self, legend=None, legacy=None):
        """*silx*: return ``[data, legend, info, pixmap, params]`` into
        *PyMca*: return ``[data, legend, dict, pixmap]``"""
        image = SilxPlotWindow.getImage(self, legend)
        if image is None:
            return None
        if legacy is None:
            legacy = self.legacy
            _logger.warning(
                    "getImage: using default legacy=%s parameter" % legacy)
        if not legacy:
            return image
        data, legend, info, pixmap, params = image
        return data, legend, merge_info_params(info, params), pixmap

    # deprecated replot parameter
    @deprecateReplotIfNotLegacy
    def removeCurve(self, legend, replot=None):
        replot = self._deprecateReplotParameter(replot)
        ret = SilxPlotWindow.removeCurve(self, legend)
        if replot:
            self.replot()
        return ret

    @deprecateReplotIfNotLegacy
    def hideCurve(self, legend, flag=True, replot=None):
        ret = SilxPlotWindow.hideCurve(self, legend, flag)
        if replot:
            self.replot()
        return ret

    @deprecateReplotIfNotLegacy
    def setActiveCurve(self, legend, replot=None):
        ret = SilxPlotWindow.setActiveCurve(self, legend)
        if replot:
            self.replot()
        return ret

    # TODO: other deprecated parameters
    @deprecateReplotIfNotLegacy
    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True,
                 color=None, symbol=None, linestyle=None,
                 xlabel=None, ylabel=None, yaxis=None,
                 xerror=None, yerror=None, z=None, selectable=None, **kw):
        # replot was replaced by resetzoom
        fill = info.get("plot_fill", False) if info is not None else False
        fill = kw.get("fill", fill)
        linewidth = kw.get("linewidth", None)
        return SilxPlotWindow.addCurve(self,
                x, y, legend=legend, info=info, replace=replace,
                color=color, symbol=symbol, linewidth=linewidth,
                linestyle=linestyle, xlabel=xlabel, ylabel=ylabel,
                yaxis=yaxis, xerror=xerror, yerror=yerror, z=z,
                selectable=selectable, fill=fill, resetzoom=replot, **kw)

    @deprecateMethodIfNotLegacy(replacementMethod="isActiveCurveHandling")
    def isActiveCurveHandlingEnabled(self):
        # isActiveCurveHandlingEnabled method deprecated in silx
        return SilxPlotWindow.isActiveCurveHandling(self)

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
        if hasattr(self, "printGraph"):
            SilxPlotWindow.printGraph(self, *args, **kwargs)

        # case of plot window defining a plotAction
        if hasattr(self, "getPrintAction"):
            printAction = SilxPlotWindow.getPrintAction(self)
            printAction.trigger()
            return

        # general case: code based on plotAction, relies on plot.saveGraph
        printer = qt.QPrinter()
        dialog = qt.QPrintDialog(printer, self)
        dialog.setWindowTitle('Print Plot')
        if not dialog.exec_():
            return False

        # Save Plot as PNG and make a pixmap from it with default dpi
        pngData = _plotAsPNG(self)

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
        SilxPlotWindow.hideCurve(self, legend, flag=False)
        if replot:
            self.replot()

    def showImage(self, legend, replot=True):
        # TODO: find out if this can be done in silx
        # unlikely case of custom plot explicitly defining showImage
        if hasattr(self, "showImage"):
            SilxPlotWindow.showImage(self, legend)
            if replot:
                self.replot()
            return
        print("showImage not implemented")

    def hideImage(self, legend, replot=True):
        # TODO: find out if this can be done in silx
        # unlikely case of custom plot explicitly defining method
        if hasattr(self, "hideImage"):
            SilxPlotWindow.hideImage(self, legend)
            if replot:
                self.replot()
            return
        print("hideImage not implemented")

    def isImageHidden(self, legend):
        # TODO: find out if this can be done in silx
        # unlikely case of custom plot explicitly defining method
        if hasattr(self, "isImageHidden"):
            return SilxPlotWindow.isImageHidden(self, legend)
        print("isImageHidden not implemented, returning False")
        return False
