#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
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
import sys
from PyMca import PyMcaQt as qt
QTVERSION = qt.qVersion()
if QTVERSION < '4.0':
    QT4 = False
else:
    QT4 = True

if QT4:
    MCAEVENT = qt.QEvent.User
    #MCAEVENT = 12345

    class McaCustomEvent(qt.QEvent):
        def __init__(self, ddict={}):
            self.dict = ddict
            qt.QEvent.__init__(self, MCAEVENT)

        def type(self):
            return MCAEVENT
else:
    #MCAEVENT = qt.QUserEvent + 1
    MCAEVENT = 12345

    class McaCustomEvent(qt.QCustomEvent):
        def __init__(self, dict={}):
            qt.QCustomEvent.__init__(self, CAEVENT)
            self.dict = dict
        
        
        
