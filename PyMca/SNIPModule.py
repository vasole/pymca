#/*##########################################################################
# Copyright (C) 2004-2009 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF BLISS Group, A. Mirone - ESRF SciSoft Group"
import numpy
try:
    import PyMca.SpecfitFuns as SpecfitFuns
except ImportError:
    print "Importing SpecfitFuns from somewhere else!"
    import SpecfitFuns

snip1d = SpecfitFuns.snip1d

def getSpectrumBackground(spectrum, width, chmin=None, chmax=None):
    if chmin is None:
        chmin = 0
    if chmax is None:
        chmax = len(spectrum)
    background = spectrum * 1
    background[chmin:chmax] = snip1d(spectrum[chmin:chmax], width)
    return background


def subtractBackgroundFromStack(stack, width, chmin=None, chmax=None):
    if chmin is None:
        chmin = 0
    if chmax is None:
        chmax = len(spectrum)
    if hasattr(stack, "info") and hasattr(stack, "data"):
        data = stack.data
    else:
        data = stack
    oldShape = data.shape
    data.shape = -1, oldShape[-1]
    if chmin > 0:
        data[:, 0:chmin] = 0
    if chmax < oldShape[-1]:
        data[:, chmax:] = 0
        
    for i in range(data.shape[0]):
        data[i,chmin:chmax] -= snip1d(data[i,chmin:chmax], width)
    data.shape = oldShape
    return

