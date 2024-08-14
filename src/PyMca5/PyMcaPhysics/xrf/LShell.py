#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import os
import numpy
from PyMca5.PyMcaIO import specfile
from PyMca5 import getDataFile

sf=specfile.Specfile(getDataFile("LShellRates.dat"))
ElementL1ShellTransitions = sf[0].alllabels()
ElementL2ShellTransitions = sf[1].alllabels()
ElementL3ShellTransitions = sf[2].alllabels()
ElementL1ShellRates = numpy.transpose(sf[0].data()).tolist()
ElementL2ShellRates = numpy.transpose(sf[1].data()).tolist()
ElementL3ShellRates = numpy.transpose(sf[2].data()).tolist()

sf=specfile.Specfile(getDataFile("LShellConstants.dat"))
ElementL1ShellConstants = sf[0].alllabels()
ElementL2ShellConstants = sf[1].alllabels()
ElementL3ShellConstants = sf[2].alllabels()
ElementL1ShellValues = numpy.transpose(sf[0].data()).tolist()
ElementL2ShellValues = numpy.transpose(sf[1].data()).tolist()
ElementL3ShellValues = numpy.transpose(sf[2].data()).tolist()
sf=None

fname = getDataFile("EADL97_LShellConstants.dat")
sf = specfile.Specfile(fname)
EADL97_ElementL1ShellConstants = sf[0].alllabels()
EADL97_ElementL2ShellConstants = sf[1].alllabels()
EADL97_ElementL3ShellConstants = sf[2].alllabels()
EADL97_ElementL1ShellValues = numpy.transpose(sf[0].data()).tolist()
EADL97_ElementL2ShellValues = numpy.transpose(sf[1].data()).tolist()
EADL97_ElementL3ShellValues = numpy.transpose(sf[2].data()).tolist()
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
weights = numpy.array(weights).astype(numpy.float64)
ElementLShellRates = numpy.zeros((len(ElementL1ShellRates),
                                  len(ElementLShellTransitions)),
                                  numpy.float64)
ElementLShellRates[:,0]     = numpy.arange(len(ElementL1ShellRates)) + 1
n1 = len(ElementL1ShellTransitions)
lo = labeloffset
ElementLShellRates[:,lo:n1] = numpy.array(ElementL1ShellRates).astype(numpy.float64)[:,lo:] * \
                              numpy.resize(weights[:,0],(nele,1))
n2 = n1 + len(ElementL2ShellTransitions) - lo
ElementLShellRates[:,n1:n2] = numpy.array(ElementL2ShellRates).astype(numpy.float64)[:,lo:]* \
                              numpy.resize(weights[:,1],(nele,1))
n1 = n2
n2 = n1 + len(ElementL3ShellTransitions) - lo
ElementLShellRates[:,n1:n2] = numpy.array(ElementL3ShellRates).astype(numpy.float64)[:,lo:]* \
                              numpy.resize(weights[:,2],(nele,1))

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
