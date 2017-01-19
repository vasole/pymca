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
"""This module defines a proxy class that provides a legacy PyMca API for
a *silx* plot widget."""

# TODO: maybe explicitly redefine addCurve & co. to avoid
#       "deprecated replot argument" warning


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
        return getattr(self.silx_plot, attr)

    def getCurve(self, legend):
        curve = self.silx_plot.getCurve(legend)
        if curve is None:
            return None
        x, y, legend, info, parameters = curve
        return x, y, legend, merge_dicts(info, parameters)

    def getActiveCurve(self, just_legend=False):
        curve = self.silx_plot.getActiveCurve(just_legend)
        if curve is None:
            return None
        x, y, legend,  info, params = curve
        if just_legend:
            return legend
        return x, y, legend, merge_dicts(info, params)

    def getAllCurves(self, just_legend=False):
        all_curves = []
        for x, y, legend, info, params in self.silx_plot.getAllCurves(just_legend):
            if just_legend:
                all_curves.append(legend)
            else:
                all_curves.append([x, y, legend, merge_dicts(info, params)])
        return all_curves

    def removeCurve(self, legend, replot=True):
        ret = self.silx_plot.removeCurve(legend)
        if replot:
            self.replot()
        return ret

    # TODO: maybe redefine addCurve to avoid "deprecated replot argument" warning
    # def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True,
    #              color=None, symbol=None, linestyle=None,
    #              xlabel=None, ylabel=None, yaxis=None,
    #              xerror=None, yerror=None, z=None, selectable=None, **kw):
    #     fill = info.get("plot_fill", False) if info is not None else False
    #     fill = kw.get("fill", fill)
    #     linewidth = kw.get("linewidth", None)
    #     return self.silx_plot.addCurve(
    #             x, y, legend=legend, info=info, replace=replace,
    #             replot=replot, color=color, symbol=symbol,
    #             linewidth=linewidth, linestyle=linestyle,
    #             xlabel=xlabel, ylabel=ylabel, yaxis=yaxis,
    #             xerror=xerror, yerror=yerror, z=z, selectable=selectable,
    #             fill=fill, resetzoom=replot, **kw)



