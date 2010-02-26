#/*##########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Software Group"
import sys
import ExternalImagesWindow
qt = ExternalImagesWindow.qt
from PyMca_Icons import IconDict

class StackROIWindow(ExternalImagesWindow.ExternalImagesWindow):
    def __init__(self, *var, **kw):
        ExternalImagesWindow.ExternalImagesWindow.__init__(self, *var, **kw)
        self.backgroundIcon = qt.QIcon(qt.QPixmap(IconDict["subtract"]))
        infotext  = 'Toggle background image subtraction from current image\n'
        infotext += 'No action if no background image available.'
        self.backgroundIcon = qt.QIcon(qt.QPixmap(IconDict["subtract"]))  
        self.backgroundButton = self.graphWidget._addToolButton(\
                                    self.backgroundIcon,
                                    self.subtractBackground,
                                    infotext,
                                    toggle = True,
                                    state = False,
                                    position = 6)
        self.buildAndConnectImageButtonBox()
        self._toggleSelectionMode()
        #self.graphWidget._yAutoScaleToggle()
        #self.graphWidget._xAutoScaleToggle()

    def subtractBackground(self):
        current = self.slider.value()
        self.showImage(current, moveslider=False)

    def showImage(self, index=0, moveslider=True):
        if self.imageList is None:
            return
        if len(self.imageList) == 0:
            return
        backgroundIndex = None
        if self.backgroundButton.isChecked():
            if self.imageNames is not None:
                i = -1
                for imageName in self.imageNames:
                    i += 1
                    if imageName.lower().endswith('background'):
                        backgroundIndex = i
                        break
        if backgroundIndex is None:
            self.setImageData(self.imageList[index])
            if self.imageNames is None:
                self.graphWidget.graph.setTitle("Image %d" % index)
            else:
                self.graphWidget.graph.setTitle(self.imageNames[index])
        else:
            self.setImageData(self.imageList[index]-\
                              self.imageList[backgroundIndex])
            if self.imageNames is None:
                self.graphWidget.graph.setTitle("Image %d Net" % index)
            else:
                self.graphWidget.graph.setTitle(self.imageNames[index]+ " Net")
        if moveslider:
            self.slider.setValue(index)
