#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
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
__revision__ = "$Revision: 1.6 $"
import os
import numpy.oldnumeric as Numeric
try:
    from PyMca import specfile
except ImportError:
    print("LShell.py is importing specfile from local directory")
    import specfile
from PyMca import PyMcaDataDir

dirname   = PyMcaDataDir.PYMCA_DATA_DIR
inputfile = os.path.join(dirname, "LShellRates.dat")
if not os.path.exists(inputfile):
    dirname = os.path.dirname(dirname)
    inputfile = os.path.join(dirname, "LShellRates.dat")
    if dirname.lower().endswith(".zip"):
        dirname = os.path.dirname(dirname)
        inputfile = os.path.join(dirname, "LShellRates.dat")
    if not os.path.exists(inputfile):
        print("Cannot find inputfile ",inputfile)
        raise IOError("Cannot find LShellRates.dat file")
sf=specfile.Specfile(os.path.join(dirname, "LShellRates.dat"))
ElementL1ShellTransitions = sf[0].alllabels()
ElementL2ShellTransitions = sf[1].alllabels()
ElementL3ShellTransitions = sf[2].alllabels()
ElementL1ShellRates = Numeric.transpose(sf[0].data()).tolist()
ElementL2ShellRates = Numeric.transpose(sf[1].data()).tolist()
ElementL3ShellRates = Numeric.transpose(sf[2].data()).tolist()

sf=specfile.Specfile(os.path.join(dirname, "LShellConstants.dat"))
ElementL1ShellConstants = sf[0].alllabels()
ElementL2ShellConstants = sf[1].alllabels()
ElementL3ShellConstants = sf[2].alllabels()
ElementL1ShellValues = Numeric.transpose(sf[0].data()).tolist()
ElementL2ShellValues = Numeric.transpose(sf[1].data()).tolist()
ElementL3ShellValues = Numeric.transpose(sf[2].data()).tolist()
sf=None

EADL97 = False
fname = os.path.join(dirname, "EADL97_LShellConstants.dat")
if os.path.exists(fname):
    sf = specfile.Specfile(fname)
    EADL97_ElementL1ShellConstants = sf[0].alllabels()
    EADL97_ElementL2ShellConstants = sf[1].alllabels()
    EADL97_ElementL3ShellConstants = sf[2].alllabels()
    EADL97_ElementL1ShellValues = Numeric.transpose(sf[0].data()).tolist()
    EADL97_ElementL2ShellValues = Numeric.transpose(sf[1].data()).tolist()
    EADL97_ElementL3ShellValues = Numeric.transpose(sf[2].data()).tolist()
    EADL97 = True
    sf = None

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
def getomegal1(ele):
    zEle = getz(ele)
    index = ElementL1ShellConstants.index('omegaL1')
    value = ElementL1ShellValues[zEle-1][index]
    if (value <= 0.0) and EADL97:
        #extend with EADL97 values
        if zEle > 99:
            #just to avoid a crash
            #I do not expect any fluorescent analysis of these elements ...
            zEle = 99
        index = EADL97_ElementL1ShellConstants.index('omegaL1')
        value = EADL97_ElementL1ShellValues[zEle-1][index]        
    return value

def getomegal2(ele):
    zEle = getz(ele)
    index = ElementL2ShellConstants.index('omegaL2')
    value = ElementL2ShellValues[zEle-1][index]
    if (value <= 0.0) and EADL97:
        #extend with EADL97 values
        if zEle > 99:
            #just to avoid a crash
            #I do not expect any fluorescent analysis of these elements ...
            zEle = 99
        index = EADL97_ElementL2ShellConstants.index('omegaL2')
        value = EADL97_ElementL2ShellValues[zEle-1][index]
    return value

def getomegal3(ele):
    zEle = getz(ele)
    index = ElementL3ShellConstants.index('omegaL3')
    value = ElementL3ShellValues[zEle-1][index]
    if (value <= 0.0) and EADL97:
        #extend with EADL97 values
        if zEle > 99:
            #just to avoid a crash
            #I do not expect any fluorescent analysis of these elements ...
            zEle = 99
        index = EADL97_ElementL3ShellConstants.index('omegaL3')
        value = EADL97_ElementL3ShellValues[zEle-1][index]        
    return value

def getCosterKronig(ele):
    ck = {}
    transitions = [ 'f12', 'f13', 'f23']
    zEle = getz(ele)
    if zEle > 99:
        #just to avoid a crash
        #I do not expect any fluorescent analysis of these elements ...
        EADL_z = 99
    else:
        EADL_z = zEle
    ckEADL = {}
    ckSum = 0.0
    for t in transitions:
        if   t in ElementL1ShellConstants:
             index   = ElementL1ShellConstants.index(t)
             ck[t]   = ElementL1ShellValues[zEle-1][index]
             if EADL97:
                 #extend with EADL97 values
                 index   = EADL97_ElementL1ShellConstants.index(t)
                 ckEADL[t]   = EADL97_ElementL1ShellValues[EADL_z-1][index]
        elif t in ElementL2ShellConstants:
             index   = ElementL2ShellConstants.index(t)
             ck[t]   = ElementL2ShellValues[zEle-1][index]
             if EADL97:
                 #extend with EADL97 values
                 index     = EADL97_ElementL2ShellConstants.index(t)
                 ckEADL[t] = EADL97_ElementL2ShellValues[EADL_z-1][index]                 
        elif t in ElementL3ShellConstants:
             index   = ElementL3ShellConstants.index(t)
             ck[t]   = ElementL3ShellValues[zEle-1][index]
             if EADL97:
                 #extend with EADL97 values
                 index   = EADL97_ElementL3ShellConstants.index(t)
                 ckEADL[t]   = EADL97_ElementL3ShellValues[EADL_z-1][index]
        else:
            print("%s not in L-Shell Coster-Kronig transitions" % t)
            continue
        ckSum += ck[t]
        
    if ckSum > 0.0:
        #I do not force EADL97 because of compatibility
        #with previous versions. I may offer forcing to
        #EADL97 in the future.
        return ck
    elif EADL97:
        #extended values if defined
        #for instance, the region from Mg to Cl
        return ckEADL
    else:
        return ck

#Jump ratios following Veigele: Atomic Data Tables 5 (1973) 51-111. p 54 and 55
def getjl1(z):
    return 1.2

def getjl2(z):
    return 1.4

def getjl3(z):
    return (80.0/z) + 1.5
    
def getwjump(ele,excitedshells=[1.0,1.0,1.0]):
    """
    wjump represents the probability for a vacancy to be created
    on the respective L-Shell by direct photoeffect on that shell
    """
    z = getz(ele)
    #weights due to photoeffect
    jl  = [getjl1(z), getjl2(z), getjl3(z)]
    wjump = []
    i = 0
    cum = 0.00
    for jump in jl:
        v = excitedshells[i]*(jump-1.0)/jump
        wjump.append(v)
        cum += v
        i+=1
    for i in range(len(wjump)):
        wjump[i] = wjump[i] / cum
    return wjump

def getweights(ele,excitedshells=None):
    if type(ele) == type(" "):
        pass
    else:
        ele = getsymbol(int(ele))
    if excitedshells == None:excitedshells=[1.0,1.0,1.0]
    w = getwjump(ele,excitedshells)
    #weights due to Coster Kronig transitions and fluorescence yields 
    ck= getCosterKronig(ele)
    w[0] *=  1.0
    w[1] *= (1.0 + ck['f12'] * w[0])
    w[2] *= (1.0 + ck['f13'] * w[0] + ck['f23'] * w[1])
    omega = [ getomegal1(ele), getomegal2(ele), getomegal3(ele)]
    for i in range(len(w)):
        w[i] *= omega[i]
    cum = sum(w)
    for i in range(len(w)):
        if cum > 0.0:
            w[i] /= cum        
    return w

if 'TOTAL' in  ElementL1ShellTransitions:
    labeloffset = 2
else:
    labeloffset = 1
ElementLShellTransitions = ElementL1ShellTransitions     +  \
                           ElementL2ShellTransitions[labeloffset:] +  \
                           ElementL3ShellTransitions[labeloffset:]
    
for i in range(len(ElementLShellTransitions)):
    ElementLShellTransitions[i]+="*"    
 
nele = len(ElementL1ShellRates)
elements = range(1,nele+1)
weights = []
for ele in elements:
    weights.append(getweights(ele))
weights = Numeric.array(weights).astype(Numeric.Float)
ElementLShellRates = Numeric.zeros((len(ElementL1ShellRates),len(ElementLShellTransitions)),Numeric.Float)
ElementLShellRates[:,0]     = Numeric.arange(len(ElementL1ShellRates)) + 1 
n1 = len(ElementL1ShellTransitions)
lo = labeloffset
ElementLShellRates[:,lo:n1] = Numeric.array(ElementL1ShellRates).astype(Numeric.Float)[:,lo:] * Numeric.resize(weights[:,0],(nele,1))
n2 = n1 + len(ElementL2ShellTransitions) - lo
ElementLShellRates[:,n1:n2] = Numeric.array(ElementL2ShellRates).astype(Numeric.Float)[:,lo:]* Numeric.resize(weights[:,1],(nele,1))
n1 = n2
n2 = n1 + len(ElementL3ShellTransitions) - lo
ElementLShellRates[:,n1:n2] = Numeric.array(ElementL3ShellRates).astype(Numeric.Float)[:,lo:]* Numeric.resize(weights[:,2],(nele,1))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ele = sys.argv[1]
        if ele in Elements:
            z = getz(ele)
            print("Atomic  Number = ",z)
            print("L1-shell yield = ",getomegal1(ele))
            print("L2-shell yield = ",getomegal2(ele))
            print("L3-shell yield = ",getomegal3(ele))
            print("L1-shell  jump = ",getjl1(z))
            print("L2-shell  jump = ",getjl2(z))
            print("L3-shell  jump = ",getjl3(z))
            print("Coster-Kronig  = ",getCosterKronig(ele))
