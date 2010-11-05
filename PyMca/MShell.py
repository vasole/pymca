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
import numpy.oldnumeric as Numeric
import specfile
import os
dirname   = os.path.dirname(__file__)
inputfile = os.path.join(dirname, "MShellRates.dat")
if not os.path.exists(inputfile):
    dirname = os.path.dirname(dirname)
    inputfile = os.path.join(dirname, "MShellRates.dat")
    if not os.path.exists(inputfile):
        if dirname.lower().endswith(".zip"):
            dirname = os.path.dirname(dirname)
            inputfile = os.path.join(dirname, "MShellRates.dat")
    if not os.path.exists(inputfile):
        print "Cannot find inputfile ",inputfile
        raise IOError("Cannot find MShellRates.dat file")

sf=specfile.Specfile(os.path.join(dirname, "MShellRates.dat"))
ElementM1ShellTransitions = sf[0].alllabels()
ElementM2ShellTransitions = sf[1].alllabels()
ElementM3ShellTransitions = sf[2].alllabels()
ElementM4ShellTransitions = sf[3].alllabels()
ElementM5ShellTransitions = sf[4].alllabels()
ElementM1ShellRates = Numeric.transpose(sf[0].data()).tolist()
ElementM2ShellRates = Numeric.transpose(sf[1].data()).tolist()
ElementM3ShellRates = Numeric.transpose(sf[2].data()).tolist()
ElementM4ShellRates = Numeric.transpose(sf[3].data()).tolist()
ElementM5ShellRates = Numeric.transpose(sf[4].data()).tolist()

sf=specfile.Specfile(os.path.join(dirname, "MShellConstants.dat"))
ElementM1ShellConstants = sf[0].alllabels()
ElementM2ShellConstants = sf[1].alllabels()
ElementM3ShellConstants = sf[2].alllabels()
ElementM4ShellConstants = sf[3].alllabels()
ElementM5ShellConstants = sf[4].alllabels()
ElementM1ShellValues = Numeric.transpose(sf[0].data()).tolist()
ElementM2ShellValues = Numeric.transpose(sf[1].data()).tolist()
ElementM3ShellValues = Numeric.transpose(sf[2].data()).tolist()
ElementM4ShellValues = Numeric.transpose(sf[3].data()).tolist()
ElementM5ShellValues = Numeric.transpose(sf[4].data()).tolist()
sf=None

EADL97 = False
fname = os.path.join(dirname, "EADL97_MShellConstants.dat")
if os.path.exists(fname):
    sf = specfile.Specfile(fname)
    EADL97_ElementM1ShellConstants = sf[0].alllabels()
    EADL97_ElementM2ShellConstants = sf[1].alllabels()
    EADL97_ElementM3ShellConstants = sf[2].alllabels()
    EADL97_ElementM4ShellConstants = sf[3].alllabels()
    EADL97_ElementM5ShellConstants = sf[4].alllabels()
    EADL97_ElementM1ShellValues = Numeric.transpose(sf[0].data()).tolist()
    EADL97_ElementM2ShellValues = Numeric.transpose(sf[1].data()).tolist()
    EADL97_ElementM3ShellValues = Numeric.transpose(sf[2].data()).tolist()
    EADL97_ElementM4ShellValues = Numeric.transpose(sf[3].data()).tolist()
    EADL97_ElementM5ShellValues = Numeric.transpose(sf[4].data()).tolist()
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
def getomegam1(ele):
    zEle = getz(ele)
    index = ElementM1ShellConstants.index('omegaM1')
    value = ElementM1ShellValues[zEle-1][index]
    if (value <= 0.0) and EADL97:
        if zEle > 99:
            #just to avoid a crash
            #I do not expect any fluorescent analysis of these elements ...
            zEle = 99        
        index = EADL97_ElementM1ShellConstants.index('omegaM1')
        value = EADL97_ElementM1ShellValues[zEle-1][index]
    return value

def getomegam2(ele):
    zEle = getz(ele)
    index = ElementM2ShellConstants.index('omegaM2')
    value = ElementM2ShellValues[zEle-1][index]
    if (value <= 0.0) and EADL97:
        if zEle > 99:
            #just to avoid a crash
            #I do not expect any fluorescent analysis of these elements ...
            zEle = 99        
        index = EADL97_ElementM2ShellConstants.index('omegaM2')
        value = EADL97_ElementM2ShellValues[zEle-1][index]
    return value

def getomegam3(ele):
    zEle = getz(ele)
    index = ElementM3ShellConstants.index('omegaM3')
    value = ElementM3ShellValues[zEle-1][index]
    if (value <= 0.0) and EADL97:
        if zEle > 99:
            #just to avoid a crash
            #I do not expect any fluorescent analysis of these elements ...
            zEle = 99        
        index = EADL97_ElementM3ShellConstants.index('omegaM3')
        value = EADL97_ElementM3ShellValues[zEle-1][index]
    return value

def getomegam4(ele):
    zEle = getz(ele)
    index = ElementM4ShellConstants.index('omegaM4')
    value = ElementM4ShellValues[zEle-1][index]
    if (value <= 0.0) and EADL97:
        if zEle > 99:
            #just to avoid a crash
            #I do not expect any fluorescent analysis of these elements ...
            zEle = 99        
        index = EADL97_ElementM4ShellConstants.index('omegaM4')
        value = EADL97_ElementM4ShellValues[zEle-1][index]
    return value

def getomegam5(ele):
    zEle = getz(ele)
    index = ElementM5ShellConstants.index('omegaM5')
    value = ElementM5ShellValues[zEle-1][index]
    if (value <= 0.0) and EADL97:
        if zEle > 99:
            #just to avoid a crash
            #I do not expect any fluorescent analysis of these elements ...
            zEle = 99        
        index = EADL97_ElementM5ShellConstants.index('omegaM5')
        value = EADL97_ElementM5ShellValues[zEle-1][index]
    return value

#Coster Kronig transitions
def getCosterKronig(ele):
    ck = {}
    transitions = [ 'f12', 'f13', 'f14', 'f15',
                           'f23', 'f24', 'f25',
                                  'f34', 'f35',
                                         'f45']
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
        if   t in ElementM1ShellConstants:
             index   = ElementM1ShellConstants.index(t)
             ck[t]   = ElementM1ShellValues[zEle-1][index]
             if EADL97:
                 #try to extend with EADL97 values
                 index   = EADL97_ElementM1ShellConstants.index(t)
                 ckEADL[t]   = EADL97_ElementM1ShellValues[EADL_z-1][index]
        elif t in ElementM2ShellConstants:
             index   = ElementM2ShellConstants.index(t)
             ck[t]   = ElementM2ShellValues[zEle-1][index]
             if EADL97:
                 #try to extend with EADL97 values
                 index   = EADL97_ElementM2ShellConstants.index(t)
                 ckEADL[t]   = EADL97_ElementM2ShellValues[EADL_z-1][index]
        elif t in ElementM3ShellConstants:
             index   = ElementM3ShellConstants.index(t)
             ck[t]   = ElementM3ShellValues[zEle-1][index]
             if EADL97:
                 #try to extend with EADL97 values
                 index   = EADL97_ElementM3ShellConstants.index(t)
                 ckEADL[t]   = EADL97_ElementM3ShellValues[EADL_z-1][index]
        elif t in ElementM4ShellConstants:
             index   = ElementM4ShellConstants.index(t)
             ck[t]   = ElementM4ShellValues[zEle-1][index]
             if EADL97:
                 #try to extend with EADL97 values
                 index   = EADL97_ElementM4ShellConstants.index(t)
                 ckEADL[t]   = EADL97_ElementM4ShellValues[EADL_z-1][index]
        else:
            print "%s not in M-Shell Coster-Kronig transitions" % t
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
def getjm1(z):
    return 1.1
    
def getjm2(z):
    return 1.1
    
def getjm3(z):
    return 1.2
    
def getjm4(z):
    return 1.5
    
def getjm5(z):
    return (225.0/z) - 0.35

def getwjump(ele,excitedshells=[1.0,1.0,1.0,1.0,1.0]):
    """
    wjump represents the probability for a vacancy to be created
    on the respective M-Shell by direct photoeffect on that shell
    """
    z = getz(ele)
    #weights due to photoeffect
    jm  = [getjm1(z), getjm2(z), getjm3(z), getjm4(z), getjm5(z)]
    wjump = []
    i = 0
    cum = 0.00
    for jump in jm:
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
    if excitedshells == None:excitedshells=[1.0,1.0,1.0,1.0,1.0]
    w = getwjump(ele,excitedshells)
    #weights due to Coster Kronig transitions and fluorescence yields 
    ck= getCosterKronig(ele)
    w[0] *=  1.0
    w[1] *= (1.0 + ck['f12'] * w[0])
    w[2] *= (1.0 + ck['f13'] * w[0] + ck['f23'] * w[1])
    w[3] *= (1.0 + ck['f14'] * w[0] + ck['f24'] * w[1] + ck['f34'] * w[2])
    w[4] *= (1.0 + ck['f15'] * w[0] + ck['f25'] * w[1] + ck['f35'] * w[2] + ck['f45'] * w[3])
    omega = [ getomegam1(ele), getomegam2(ele), getomegam3(ele), getomegam4(ele), getomegam5(ele)]
    for i in range(len(w)):
        w[i] *= omega[i]
    cum = sum(w)
    for i in range(len(w)):
        if cum > 0.0:
            w[i] /= cum        
    return w

ElementMShellTransitions = ElementM1ShellTransitions     +  \
                           ElementM2ShellTransitions[2:] +  \
                           ElementM3ShellTransitions[2:] +  \
                           ElementM4ShellTransitions[2:] +  \
                           ElementM5ShellTransitions[2:]
nele = len(ElementM1ShellRates)
elements = range(1,nele+1)
weights = []
for ele in elements:
    weights.append(getweights(ele))
weights = Numeric.array(weights).astype(Numeric.Float)
ElementMShellRates = Numeric.zeros((len(ElementM1ShellRates),len(ElementMShellTransitions)),Numeric.Float)
ElementMShellRates[:,0]     = Numeric.arange(len(ElementM1ShellRates)) + 1 
n1 = len(ElementM1ShellTransitions)
ElementMShellRates[:,2:n1] = Numeric.array(ElementM1ShellRates).astype(Numeric.Float)[:,2:] * Numeric.resize(weights[:,0],(nele,1))
n2 = n1 + len(ElementM2ShellTransitions) - 2
ElementMShellRates[:,n1:n2] = Numeric.array(ElementM2ShellRates).astype(Numeric.Float)[:,2:]* Numeric.resize(weights[:,1],(nele,1))
n1 = n2
n2 = n1 + len(ElementM3ShellTransitions) - 2
ElementMShellRates[:,n1:n2] = Numeric.array(ElementM3ShellRates).astype(Numeric.Float)[:,2:]* Numeric.resize(weights[:,2],(nele,1))
n1 = n2
n2 = n1 + len(ElementM4ShellTransitions) - 2
ElementMShellRates[:,n1:n2] = Numeric.array(ElementM4ShellRates).astype(Numeric.Float)[:,2:]* Numeric.resize(weights[:,3],(nele,1))
n1 = n2
n2 = n1 + len(ElementM5ShellTransitions) - 2
ElementMShellRates[:,n1:n2] = Numeric.array(ElementM5ShellRates).astype(Numeric.Float)[:,2:]* Numeric.resize(weights[:,4],(nele,1))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ele = sys.argv[1]
        if ele in Elements:
            z = getz(ele)
            print "Atomic  Number = ",z
            print "M1-shell yield = ",getomegam1(ele)
            print "M2-shell yield = ",getomegam2(ele)
            print "M3-shell yield = ",getomegam3(ele)
            print "M4-shell yield = ",getomegam4(ele)
            print "M5-shell yield = ",getomegam5(ele)
            print "M1-shell  jump = ",getjm1(z)
            print "M2-shell  jump = ",getjm2(z)
            print "M3-shell  jump = ",getjm3(z)
            print "M4-shell  jump = ",getjm4(z)
            print "M5-shell  jump = ",getjm5(z)
            print "Coster-Kronig  = ",getCosterKronig(ele)
            EADL97 = False
            print "Coster-Kronig no EADL97 = ",getCosterKronig(ele)
            
