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
__revision__ = "$Revision: 1.5 $"

import numpy.oldnumeric as Numeric
import specfile
import os

dirname   = os.path.dirname(__file__)
inputfile = os.path.join(dirname, "KShellRates.dat")
if not os.path.exists(inputfile):
    dirname = os.path.dirname(dirname)
    inputfile = os.path.join(dirname, "KShellRates.dat")
    if not os.path.exists(inputfile):
        if dirname.lower().endswith(".zip"):
            dirname = os.path.dirname(dirname)
            inputfile = os.path.join(dirname, "KShellRates.dat")
    if not os.path.exists(inputfile):
        print "Cannot find inputfile ",inputfile
        raise IOError("Cannot find KShellRates.dat file")

sf=specfile.Specfile(os.path.join(dirname, "KShellRates.dat"))
ElementKShellTransitions = sf[0].alllabels()
ElementKShellRates = Numeric.transpose(sf[0].data()).tolist()

ElementKAlphaTransitions = []
ElementKBetaTransitions = []
for transition in ElementKShellTransitions:
    if transition[0] == 'K':
        if transition[1] == 'L':
            ElementKAlphaTransitions.append(transition)
        else:
            ElementKBetaTransitions.append(transition)
    elif transition[0] == 'Z':
        ElementKAlphaTransitions.append(transition)
        ElementKBetaTransitions.append(transition)
    else:
        #TOTAL column meaningless
        pass

filedata = sf[0].data()
ndata    = sf[0].lines()
ElementKAlphaRates = filedata[0] * 1
ElementKAlphaRates.shape = [ndata, 1]
ElementKBetaRates = filedata[0] * 1
ElementKBetaRates.shape  = [ndata, 1]
for transition in ElementKAlphaTransitions:
    if transition[0] != 'Z':
        data = filedata[ElementKShellTransitions.index(transition)] * 1
        data.shape = [ndata, 1]
        ElementKAlphaRates = Numeric.concatenate((ElementKAlphaRates, data),
                                                 axis = 1)

for transition in ElementKBetaTransitions:
    if transition[0] != 'Z':
        data = filedata[ElementKShellTransitions.index(transition)] * 1
        data.shape = [ndata, 1] 
        ElementKBetaRates = Numeric.concatenate((ElementKBetaRates, data),
                                                axis = 1)
for i in range(len(ElementKAlphaTransitions)):
    if ElementKAlphaTransitions[i] != 'Z':
        ElementKAlphaTransitions[i] = ElementKAlphaTransitions[i] + "a"

for i in range(len(ElementKBetaTransitions)):
    if ElementKBetaTransitions[i] != 'Z':
        ElementKBetaTransitions[i] = ElementKBetaTransitions[i] + "b"

ElementKAlphaRates = ElementKAlphaRates.tolist()
ElementKBetaRates  = ElementKBetaRates.tolist()

sf=specfile.Specfile(os.path.join(dirname, "KShellConstants.dat"))
ElementKShellConstants = sf[0].alllabels()
ElementKShellValues = Numeric.transpose(sf[0].data()).tolist()
sf=None

Elements = ['H', 'He', 
            'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
            'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
            'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
            'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
            'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 
            'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 
            'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 
            'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 
            'Bh', 'Hs', 'Mt']


def getsymbol(z):
    return Elements[z-1]

def getz(ele):
    return Elements.index(ele)+1

#fluorescence yields
def getomegak(ele):
    index = ElementKShellConstants.index('omegaK')
    return ElementKShellValues[getz(ele)-1][index]

#Jump ratios following Veigele: Atomic Data Tables 5 (1973) 51-111. p 54 and 55
def getjk(z):
    return (125.0/z) + 3.5
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ele = sys.argv[1]
        if ele in Elements:
            z = getz(ele)
            print "Atomic  Number = ",z
            print "K-shell yield = ",getomegak(ele)
            print "K-shell  jump = ",getjk(z)

