#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2004-2020 V.A. Sole, European Synchrotron Radiation Facility
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
__author__ = "H. Payno- ESRF Data Analysis"
__contact__ = "henri.payno@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui.misc.SelectionTable import SelectionTable
from tomogui.gui.datasource.QDataSourceWidget import QDataSourceWidget
from tomogui.configuration.config import FBPConfig
try:
    from freeart.configuration.config import _ReconsConfig
except ImportError:
    from tomogui.third_party.configuration.config import _ReconsConfig


class TomoRecons(qt.QWidget):
    """Widget to select what are the I0 sinogram, sinograms to reconstruct and
    the kind od reconstruction we want to run
    """
    def __init__(self, parent=None, entries=None):
        qt.QWidget.__init__(self, parent)
        self.setLayout(qt.QVBoxLayout())
        self._cbReconsType = QDataSourceWidget.getReconsTypeCombobox(parent=self)
        self.layout().addWidget(self._cbReconsType)

        self._widgetGiveItI0 = qt.QWidget(parent=self)
        self._widgetGiveItI0.setLayout(qt.QHBoxLayout())
        self._giveItCheckBox = qt.QCheckBox('give It',
                                            parent=self._widgetGiveItI0)
        self._giveItCheckBox.setChecked(True)
        self._giveItCheckBox.setToolTip('It is needed in order to produce the '
                                        'absorption matrix and deduce the self'
                                        ' absorption matrices. If not '
                                        'available then you will have to give '
                                        'directly the absorption matrix and '
                                        'the self absorption matrices.')
        self._widgetGiveItI0.layout().addWidget(self._giveItCheckBox)
        self._giveI0CheckBox = qt.QCheckBox('give I0',
                                            parent=self._widgetGiveItI0)
        self._giveI0CheckBox.setChecked(True)
        self._giveI0CheckBox.setToolTip('I0 sinogram is used to normalized '
                                        'sinograms. If this is not available '
                                        'then you can enter a single value to '
                                        'normalize sinograms.')
        self._widgetGiveItI0.layout().addWidget(self._giveI0CheckBox)
        self.layout().addWidget(self._widgetGiveItI0)

        self._fbpSelectionTable = FBPSinoReconsSelectionTable(parent=self)
        self.layout().addWidget(self._fbpSelectionTable)
        self._txSelectionTable = TxSinoReconsSelectionTable(parent=self)
        self.layout().addWidget(self._txSelectionTable)
        self._fluoSelectionTable = FluoSinoReconsSelectionTable(parent=self)
        self.layout().addWidget(self._fluoSelectionTable)

        self._cbReconsType.currentIndexChanged.connect(self._updadeView)
        self._giveI0CheckBox.toggled.connect(self._txSelectionTable.setI0Enabled)
        self._giveI0CheckBox.toggled.connect(self._fluoSelectionTable.setI0Enabled)
        self._giveItCheckBox.toggled.connect(self._fluoSelectionTable.setItEnabled)

        self.setNames(entries)
        self._updadeView()

    def setNames(self, entries):
        if entries:
            assert(isinstance(entries, list) or isinstance(entries, tuple))
            nEntries = len(entries)
            self._fbpSelectionTable.setRowCount(nEntries)
            self._txSelectionTable.setRowCount(nEntries)
            self._fluoSelectionTable.setRowCount(nEntries)
            for iEntry, entry in enumerate(entries):
                self._fbpSelectionTable.fillLine(iEntry, [entry, "", ""])
                self._txSelectionTable.fillLine(iEntry, [entry, "", ""])
                self._fluoSelectionTable.fillLine(iEntry, [entry, "", "", ""])

            self._fbpSelectionTable.resizeColumnsToContents()
            self._txSelectionTable.resizeColumnsToContents()
            self._fluoSelectionTable.resizeColumnsToContents()

    def _updadeView(self):
        isFluoOrCompton = self.getReconsType() in (_ReconsConfig.FLUO_ID,
                                                   _ReconsConfig.COMPTON_ID)
        self._fluoSelectionTable.setVisible(isFluoOrCompton)
        self._giveI0CheckBox.setVisible(self.getReconsType() != FBPConfig.FBP_ID)
        self._giveItCheckBox.setVisible(isFluoOrCompton)
        self._txSelectionTable.setVisible(self.getReconsType() == _ReconsConfig.TX_ID)
        self._fbpSelectionTable.setVisible(self.getReconsType() == FBPConfig.FBP_ID)

    def getReconsType(self):
        """

        :return: the reconstruction type (FBP, fluo...)  requested by the user
        """
        return str(QDataSourceWidget.DICT_IDS[self._cbReconsType.currentText()])

    def getI0(self):
        """

        :return: the ID of the sinogram used as I0
        """
        if self.getReconsType() == FBPConfig.FBP_ID:
            return None
        elif self.getReconsType() == _ReconsConfig.TX_ID:
            return self._txSelectionTable.getI0Selection()
        elif self.getReconsType() in (_ReconsConfig.FLUO_ID, _ReconsConfig.COMPTON_ID):
            return self._fluoSelectionTable.getI0Selection()

    def getIt(self):
        """

        :return: the Id of the sinogram used for It
        """
        if self.getReconsType() in (FBPConfig.FBP_ID, _ReconsConfig.TX_ID):
            return None
        elif self.getReconsType() in (_ReconsConfig.FLUO_ID, _ReconsConfig.COMPTON_ID):
            return self._fluoSelectionTable.getItSelection()

    def getSinogramsToRecons(self):
        """

        :return: the list of ID of the sinograms to reconstruct
        """
        if self.getReconsType() == FBPConfig.FBP_ID:
            return self._fbpSelectionTable.getSinogramsToRecons()
        elif self.getReconsType() == _ReconsConfig.TX_ID:
            return self._txSelectionTable.getSinogramsToRecons()
        elif self.getReconsType() in (_ReconsConfig.FLUO_ID, _ReconsConfig.COMPTON_ID):
            return self._fluoSelectionTable.getSinogramsToRecons()

    def getMultipleRole(self):
        """
        Check entries to check if one is selected in a multiple role
        (requested for I0, It, sinogram...)

        :return: None or the entries assigned to multiple roles. Each entry is
                 associated to a list of role
         :rtype: dict
        """
        if self.getReconsType() == FBPConfig.FBP_ID:
            return None
        if self.getReconsType() == _ReconsConfig.TX_ID:
            sinograms = self._txSelectionTable.getSinogramsToRecons()
            if not self._giveI0CheckBox.isChecked():
                return None
            I0 = self._txSelectionTable.getI0Selection()
            if I0 and I0 in sinograms:
                return {I0: ['i0', 'sinogram to reconstruct']}
            else:
                return None
        elif self.getReconsType() in (_ReconsConfig.FLUO_ID, _ReconsConfig.COMPTON_ID):
            fluoSinograms = self._fluoSelectionTable.getSinogramsToRecons()
            I0 = self._fluoSelectionTable.getI0Selection()
            It = self._fluoSelectionTable.getItSelection()
            duplicated = {}

            if self._giveI0CheckBox.isChecked() and I0 in fluoSinograms:
                duplicated[I0] = ['i0', 'sinogram to reconstruct']
            if self._giveI0CheckBox.isChecked() and \
                    self._giveItCheckBox.isChecked() and I0 == It:
                if I0 in duplicated:
                    duplicated[I0].append('it')
                else:
                    duplicated[I0] = ['i0', 'it']
            elif self._giveItCheckBox.isChecked() and It in fluoSinograms:
                duplicated[It] = ['it', 'sinogram to reconstruct']
            return None if len(duplicated) == 0 else duplicated

    def sizeHint(self):
        return qt.QSize(400, 600)


class TomoReconsDialog(qt.QDialog):
    """Dialog to validate the sinogram selection for tomogui reconstruction
    """
    class SinogramHasMultipleRoleInfoMessage(qt.QMessageBox):
        def __init__(self, sinoName, roles):
            qt.QMessageBox.__init__(self)

            self.setIcon(qt.QMessageBox.Warning)
            self.setText('Multiple role for a sinogram')

            self.setInformativeText(
                'The sinogram %s is used in multiple roles (%s).'
                'This seems like an incorrect selection and might bring'
                'incoherent reconstruction. Continue ?' %(str(sinoName), "; ".join(roles)))

            self.yesButton = self.addButton(qt.QMessageBox.Ignore)
            self.noButton = self.addButton(qt.QMessageBox.Cancel)

    def __init__(self, parent=None, entries=None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle('Sinogram selection for reconstruction')
        self.mainWidget = TomoRecons(parent=self, entries=entries)

        types = qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel
        _buttons = qt.QDialogButtonBox(parent=self)
        _buttons.setStandardButtons(types)
        _buttons.button(qt.QDialogButtonBox.Ok).clicked.connect(
            self._okTriggered)
        _buttons.button(qt.QDialogButtonBox.Cancel).clicked.connect(
            self.reject)

        self.setLayout(qt.QVBoxLayout())
        self.layout().addWidget(self.mainWidget)
        self.layout().addWidget(_buttons)

    def getReconstructionType(self):
        return self.mainWidget.getReconsType()

    def getSinogramsToRecons(self):
        return self.mainWidget.getSinogramsToRecons()

    def getIt(self):
        return self.mainWidget.getIt()

    def hasIt(self):
        return self.getIt() is not None

    def getI0(self):
        return self.mainWidget.getI0()

    def hasI0(self):
        return self.mainWidget.getI0() is not None

    def _okTriggered(self):
        if self.checkMultipleRole():
            self.accept()

    def checkMultipleRole(self):
        """

        :return: True if the current validation is correct.
        """
        mSelections = self.mainWidget.getMultipleRole()
        if mSelections is not None:
            name = list(mSelections.keys())[0]
            diag = self.SinogramHasMultipleRoleInfoMessage(sinoName=name,
                                                           roles=mSelections[name])
            if diag.exec():
                return diag.result() == qt.QDialogButtonBox.Ignore
            else:
                return False
        return True


class FluoSinoReconsSelectionTable(SelectionTable):
    """Table to select the sinogram to reconstruct and in the case of
    the fluorescence reconstruction what are It, I0... sinograms"""
    LABELS = ["name", "sinogram to reconstruct", "I0", "It"]

    TYPES = ["Text", "CheckBox", "RadioButton", "RadioButton"]

    def __init__(self, parent=None):
        SelectionTable.__init__(self, parent)

    def getItSelection(self):
        nSelection = len(self.getSelection()['it'])
        if nSelection == 0:
            return None
        elif nSelection == 1:
            index = self.getSelection()['it']
            assert(len(index) == 1)
            return self.getSelection()['name'][index[0]]
        else:
            raise ValueError('multiple sinogram set as I0, shouldn\'t happen')

    def getI0Selection(self):
        nSelection = len(self.getSelection()['i0'])
        if nSelection == 0:
            return None
        elif nSelection == 1:
            index = self.getSelection()['i0']
            assert(len(index) == 1)
            return self.getSelection()['name'][index[0]]
        else:
            raise ValueError('multiple sinogram set as I0, shouldn\'t happen')

    def getSinogramsToRecons(self):
        sinograms = []
        selections = self.getSelection()
        for iSino in selections['sinogram to reconstruct']:
            sinograms.append(selections['name'][iSino])
        return sinograms

    def setI0Enabled(self, enabled):
        self.setColumnEnabled(index=2, enabled=enabled)

    def setItEnabled(self, enabled):
        self.setColumnEnabled(index=3, enabled=enabled)


class TxSinoReconsSelectionTable(SelectionTable):
    """Table to select the sinogram to reconstruct and in the case of
    the fluorescence reconstruction what are It, I0... sinograms"""
    LABELS = ["name", "sinogram to reconstruct", "I0"]

    TYPES = ["Text", "CheckBox", "RadioButton"]

    def __init__(self, parent=None):
        SelectionTable.__init__(self, parent)

    def getI0Selection(self):
        nSelection = len(self.getSelection()['i0'])
        if nSelection == 0:
            return None
        elif nSelection == 1:
            index = self.getSelection()['i0']
            assert(len(index) == 1)
            return self.getSelection()['name'][index[0]]
        else:
            raise ValueError('multiple sinogram set as I0, shouldn\'t happen')

    def getSinogramsToRecons(self):
        sinograms = []
        selections = self.getSelection()
        for iSino in selections['sinogram to reconstruct']:
            sinograms.append(selections['name'][iSino])
        return sinograms

    def setI0Enabled(self, enabled):
        self.setColumnEnabled(index=2, enabled=enabled)


class FBPSinoReconsSelectionTable(SelectionTable):
    LABELS = ["name", "sinogram to reconstruct"]

    TYPES = ["Text", "CheckBox"]

    def __init__(self, parent=None):
        SelectionTable.__init__(self, parent)

    def getSinogramsToRecons(self):
        sinograms = []
        selections = self.getSelection()
        for iSino in selections['sinogram to reconstruct']:
            sinograms.append(selections['name'][iSino])
        return sinograms


if __name__ == '__main__':
    app = qt.QApplication([])
    widget = TomoReconsDialog(entries=["Cnt1", "Cnt2", "Cnt3", "Cnt4", "Cnt5"])
    widget.show()
    app.exec()
