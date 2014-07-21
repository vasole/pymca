#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
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
from PyMca5.PyMcaIO import ConfigDict
from PyMca5 import PyMcaDataDir

dirmod = PyMcaDataDir.PYMCA_DATA_DIR
ffile = os.path.join(dirmod, "attdata")
ffile = os.path.join(ffile, "atomsf.dict")
if not os.path.exists(ffile):
    #freeze does bad things with the path ...
    dirmod = os.path.dirname(dirmod)
    ffile = os.path.join(dirmod, "attdata")
    ffile = os.path.join(ffile, "atomsf.dict")
    if not os.path.exists(ffile):
        if dirmod.lower().endswith(".zip"):
            dirmod = os.path.dirname(dirmod)
            ffile = os.path.join(dirmod, "attdata")
            ffile = os.path.join(ffile, "atomsf.dict")
    if not os.path.exists(ffile):
        print("Cannot find file ", ffile)
        raise IOError("Cannot find file %s" % ffile)
COEFFICIENTS = ConfigDict.ConfigDict()
COEFFICIENTS.read(ffile)
KEVTOANG = 12.39852000
R0 = 2.82E-13 #electron radius in cm

def getElementFormFactor(ele, theta, energy):
    """
    Usage:
        getFormFactor(ele,theta, energy):

    ele   - Element
    theta - Scattering angle or array of scattering angles in degrees
    energy- Photon Energy in keV

    This routine calculates the atomic form factor in electron units using
    a four gaussians approximation
    """
    wavelength = KEVTOANG / energy
    x = numpy.sin(theta*(numpy.pi/360.0)) / wavelength
    x = x * x
    c0= COEFFICIENTS[ele]['c'][0]
    c = COEFFICIENTS[ele]['c'][1:]
    b = COEFFICIENTS[ele]['b']
    return c0 + (c[0] * numpy.exp(-b[0]*x)) + \
                  (c[1] * numpy.exp(-b[1]*x)) + \
                  (c[2] * numpy.exp(-b[2]*x)) + \
                  (c[3] * numpy.exp(-b[3]*x))

def getElementCoherentDifferentialCrossSection(ele, theta, energy, p1=None):
    if p1 is None:
        p1=0.0
    if (p1 > 1.0) or (p1 < -1):
        raise ValueError(\
        "Invalid degree of linear polarization respect to the scattering plane")
    thetasin2 = pow(numpy.sin(theta*numpy.pi/180.0),2)
    return (1.0+ 0.5 *(p1-1.0) * thetasin2) * \
           pow(R0*getElementFormFactor(ele, theta, energy),2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >  3:
        ele   = sys.argv[1]
        theta = float(sys.argv[2])
        energy= float(sys.argv[3])
        print(getElementFormFactor(ele, theta, energy))
    else:
        print("Usage:")
        print("python CoherentScattering.py Element Theta(deg) Energy(kev)")
