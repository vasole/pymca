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
from PyMca5.PyMcaGui.math.fitting.QScriptOption import FieldSheet
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


class SpecfitConfigGui(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        layout = qt.QHBoxLayout(self)

        self.default = SpecfitFunctions.SPECFITFUNCTIONS_DEFAULTS
        self.sheets = {}
        self.output = {}

        for sheet in [sheet1, sheet2, sheet3]:
            frame = qt.QFrame(self)
            sublayout = qt.QVBoxLayout()
            frame.setLayout(sublayout)
            name = sheet['notetitle']
            label = qt.QLabel(frame)
            label.setText(name)
            self.sheets[name] = FieldSheet(fields=sheet['fields'])
            self.sheets[name].setdefaults(self.default)
            sublayout.addWidget(label)
            sublayout.addWidget(self.sheets[name])
            frame.setLayout(sublayout)
            frame.setFrameStyle(qt.QFrame.StyledPanel | qt.QFrame.Raised)
            layout.addWidget(frame)

        self.setLayout(layout)

    def configure(self, ddict):
        self.setConfiguration(ddict)

    def setConfiguration(self, ddict):
        # None can show up when checkbox is unchecked
        # and FieldSheet cannot handle None for checkboxes
        for key, value in ddict.items():
            if value is None:
                ddict[key] = 0
        if "configuration" in ddict:
            for key, value in ddict["configuration"].items():
                if value is None:
                    ddict["configuration"][key] = 0
        for name, sheet in self.sheets.items():
            if "configuration" in ddict:
                sheet.setdefaults(ddict["configuration"])
            else:
                sheet.setdefaults(ddict)

        self.output.update(ddict)

    def getConfiguration(self):
        for name, sheet in self.sheets.items():
            if "configuration" in self.output:
                self.output["configuration"].update(sheet.get())
            else:
                self.output.update(sheet.get())
        return self.output
