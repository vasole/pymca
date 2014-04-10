#/*##########################################################################
# Copyright (C) 2004-2013 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
__author__ = "V.A. Sole - ESRF Software Group"
import numpy
from PyMca5 import Plugin1DBase
from PyMca5 import XASSelfAttenuationCorrection
from PyMca5 import XASSelfAttenuationWindow

class XASSelfAttenuationPlugin(Plugin1DBase.Plugin1DBase):
    def __init__(self, plotWindow, **kw):
        Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)
        self.methodDict = {}
        text = "Configure fluorescent XAS self-\n"
        text += "attenuation correction parameters.\n"
        text += "Input curves need to be normalized.\n"
        text += "For the time being thick sample is assumed."
        function = self.configure
        info = text
        icon = None
        self.methodDict["Configure"] =[function,
                                       info,
                                       icon]
        function = self.correctActive
        text = "Add corrected active curve."
        info = text
        icon = None
        self.methodDict["Correct Active"] =[function,
                                         info,
                                         icon]
        function = self.correctAll
        text = "Replace all curves by normalized ones."
        info = text
        icon = None
        self.methodDict["Correct All"] =[function,
                                         info,
                                         icon]
        self.widget = None
        self.instance = XASSelfAttenuationCorrection.XASSelfAttenuationCorrection()
        self.parameters = None
        
    #Methods to be implemented by the plugin
    def getMethods(self, plottype=None):
        """
        A list with the NAMES  associated to the callable methods
        that are applicable to the specified plot.

        Plot type can be "SCAN", "MCA", None, ...        
        """
        names = list(self.methodDict.keys())
        names.sort()
        return names

    def getMethodToolTip(self, name):
        """
        Returns the help associated to the particular method name or None.
        """
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        """
        Returns the pixmap associated to the particular method name or None.
        """
        return self.methodDict[name][2]

    def applyMethod(self, name):
        """
        The plugin is asked to apply the method associated to name.
        """
        self.methodDict[name][0]()
        return

    def configure(self):
        if self.widget is None:
            self.widget = XASSelfAttenuationWindow.XASSelfAttenuationDialog()
        ret = self.widget.exec_()
        if ret:
            self.configuration = self.widget.getConfiguration()
            self.instance.setConfiguration(self.configuration)

    def correctActive(self):
        #check we have a configuration
        if self.configuration is None:
            raise RuntimeError("Please configure the plugin")

        #get active curve
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            raise ValueError("Please select an active curve")
            return

        energy, spectrum, legend, info = activeCurve[0:4]
        spectrum = self.instance.correctNormalizedSpectrum(energy, spectrum)
        self.addCurve(energy, spectrum, legend="CORR"+legend, info=info, replace=False, replot=True)

    def correctAll(self):
        #check we have a configuration
        if self.configuration is None:
            raise RuntimeError("Please configure the plugin")

        curves = self.getAllCurves()
        nCurves = len(curves)
        if nCurves < 1:
            raise ValueError("At least one curve needed")
            return

        #get active curve
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            activeCurve = curves[0]

        for i in range(nCurves):
            energy, spectrum, legend, info = curves[i][0:4]
            if i == 0:
                replace = True
            else:
                replace = False
            if i == nCurves - 1:
                replot = True
            else:
                replot = False
            spectrum = self.instance.correctNormalizedSpectrum(energy, spectrum)
            self.addCurve(energy, spectrum, legend="CORR"+legend, info=info,
                          replot=replot, replace=replace)

MENU_TEXT = "XAS Self-Attenuation Correction"
def getPlugin1DInstance(plotWindow, **kw):
    ob = XASSelfAttenuationPlugin(plotWindow)
    return ob
