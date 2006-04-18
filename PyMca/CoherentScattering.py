#/*##########################################################################
# Copyright (C) 2004-2006 European Synchrotron Radiation Facility
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
# is a problem to you.
#############################################################################*/
import ConfigDict
import os
import Numeric
import Scofield1973

dirmod = os.path.dirname(Scofield1973.__file__) 
file   = os.path.join(dirmod,"attdata")
file   = os.path.join(file,"atomsf.dict")
if not os.path.exists(file):
    #freeze does bad things with the path ...
    file = os.path.dirname(dirmod)
    file = os.path.join(file,"attdata")
    file = os.path.join(file,"atomsf.dict")
    if not os.path.exists(file):
        print "Cannot find file ",file
        raise "IOError","Cannot find file %s" % file

COEFFICIENTS = ConfigDict.ConfigDict()
COEFFICIENTS.read(file)    
KEVTOANG = 12.39852000
R0 = 2.82E-13 #electron radius in cm

def getElementFormFactor(ele, theta, energy):
    """
    Usage: 
        getFormFactor(ele,theta, energy):
    
    ele   - Element
    theta - Scattering angle in degrees
    energy- Photon Energy in keV
    
    This routine calculates the atomic form factor in electron units using 
    a four gaussians approximation
    """
    wavelength = KEVTOANG / energy
    x = Numeric.sin(theta*(Numeric.pi/360.0)) /wavelength
    x = x * x
    c0= COEFFICIENTS[ele]['c'][0]
    c = COEFFICIENTS[ele]['c'][1:]
    b = COEFFICIENTS[ele]['b']
    return c0 + (c[0] * Numeric.exp(-b[0]*x)) + \
                  (c[1] * Numeric.exp(-b[1]*x)) + \
                  (c[2] * Numeric.exp(-b[2]*x)) + \
                  (c[3] * Numeric.exp(-b[3]*x))

def getElementCoherentDifferentialCrossSection(ele, theta, energy, p1=None):
    if p1 is None:p1=0.0
    if (p1 > 1.0) or (p1 < -1):
        raise "ValueError",\
        "Invalid degree of linear polarization respect to the scattering plane"
    thetasin2 = pow(Numeric.sin(theta*Numeric.pi/180.0),2)
    return (1.0+ 0.5 *(p1-1.0) * thetasin2) * \
           pow(R0*getElementFormFactor(ele, theta, energy),2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >  3:
        ele   = sys.argv[1]
        theta = float(sys.argv[2])
        energy= float(sys.argv[3])
        print getElementFormFactor(ele, theta, energy)
        import Tkinter
        import SimplePlot
        root = Tkinter.Tk()
        e=range(int(energy+1))[1:]
        y=[getElementFormFactor(ele, theta, x) for x in e]
        SimplePlot.plot([e,y])        
        root.mainloop()
    else:
        print "Usage:"
        print "python FormFactor.py Element Theta(deg) Energy(kev)"
    
