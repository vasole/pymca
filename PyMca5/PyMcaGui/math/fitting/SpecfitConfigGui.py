#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""Configuration widget for using Specfit functions in a
SimpleFitWindow."""


from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui.math.fitting import QScriptOption
from PyMca5.PyMcaMath.fitting import SpecfitFunctions


sheet1 = {'notetitle': 'Restrains',
          'fields': (["CheckField", 'HeightAreaFlag', 'Force positive Height/Area'],
                     ["CheckField", 'PositionFlag', 'Force position in interval'],
                     ["CheckField", 'PosFwhmFlag', 'Force positive FWHM'],
                     ["CheckField", 'SameFwhmFlag', 'Force same FWHM'],
                     ["CheckField", 'EtaFlag', 'Force Eta between 0 and 1'],
                     ["CheckField", 'NoConstrainsFlag', 'Ignore Restrains'])}


sheet2 = {'notetitle': 'Search',
          'fields': (["EntryField", 'FwhmPoints', 'Fwhm Points: '],
                     ["EntryField", 'Sensitivity', 'Sensitivity: '],
                     ["EntryField", 'Yscaling', 'Y Factor   : '],
                     ["CheckField", 'ForcePeakPresence', 'Force peak presence '])}

sheet3 = {'notetitle': 'Fit',
          'fields': (["CheckField", 'WeightFlag', 'Weight'],
                     ["CheckField", 'AutoFwhm', 'Auto FWHM'],
                     ["CheckField", 'AutoScaling', 'AutoScaling'])}


def getSpecfitConfigGui(parent, default=None):
    if default is None:
        default = SpecfitFunctions.SPECFITFUNCTIONS_DEFAULTS
    return QScriptOption.QScriptOption(
        parent, name='Fit Configuration',
        sheets=(sheet1, sheet2),
        default=default)

class SpecfitConfigGui(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        self.configWidget = getSpecfitConfigGui(parent=self)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.configWidget)

        self.setLayout(layout)

    def setConfiguration(self, ddict):
        self.configWidget.output.update(ddict)

    def getConfiguration(self):
        self.configWidget.output.update(self.configWidget.default)
        for name, sheet in self.configWidget.sheets.items():
            self.configWidget.output.update(sheet.get())
        return self.configWidget.output
