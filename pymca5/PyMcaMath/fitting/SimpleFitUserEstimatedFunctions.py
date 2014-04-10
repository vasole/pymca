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
import numpy
from PyMca import SpecfitFuns
arctan = numpy.arctan
pi = numpy.pi

class SimpleFitDefaultFunctions(object):
    def __init__(self):
        self.gaussian = SpecfitFuns.gauss
        self.areaGaussian = SpecfitFuns.agauss
        self.lorentzian = SpecfitFuns.lorentz
        self.areaLorentzian = SpecfitFuns.alorentz
        self.pseudoVoigt = SpecfitFuns.pvoigt
        self.areaPseudoVoigt = SpecfitFuns.apvoigt
        
    def hypermet(self,pars,x):
        """
        Default hypermet function
        """
        return SpecfitFuns.ahypermet(pars, x, 15, 0)

    def stepDown(self,pars,x):
        """
        Complementary error function like.
        """
        return 0.5*SpecfitFuns.downstep(pars,x)

    def stepUp(self,pars,x):
        """
        Error function like.
        """
        return 0.5*SpecfitFuns.upstep(pars,x)

    def slit(self,pars,x):
        """
        Function to calulate slit width and knife edge cut
        """
        return 0.5*SpecfitFuns.slit(pars,x)

    def atan(self, pars, x):
        return pars[0] * (0.5 + (arctan((1.0*x-pars[1])/pars[2])/pi))
    
    def polynomial(self, pars, x):
        result = numpy.zeros(x.shape, numpy.float) + pars[0]
        if len(pars) == 1:
            return result
        d = x * 1.0
        for p in pars[1:]:
            result += p * d
            d *= d
        return result

fitfuns=SimpleFitDefaultFunctions()

FUNCTION=[fitfuns.gaussian,
          fitfuns.lorentzian,
          fitfuns.pseudoVoigt,
          fitfuns.areaGaussian,
          fitfuns.areaLorentzian,
          fitfuns.areaPseudoVoigt,
          fitfuns.stepDown,
          fitfuns.stepUp,
          fitfuns.slit,
          fitfuns.atan,
          fitfuns.hypermet,
          fitfuns.polynomial,
          fitfuns.polynomial,
          fitfuns.polynomial,
          fitfuns.polynomial,
          fitfuns.polynomial,
          fitfuns.polynomial]
          
PARAMETERS=[['Height','Position','Fwhm'],
            ['Height','Position','Fwhm'],
            ['Height','Position','Fwhm','Eta'],
            ['Area','Position','Fwhm'],
            ['Area','Position','Fwhm'],
            ['Area','Position','Fwhm','Eta'],
            ['Height','Position','FWHM'],
            ['Height','Position','FWHM'],
            ['Height','Position','FWHM','BeamFWHM'],
            ['Height','Position','Width'],
            ['G_Area','Position','FWHM',
             'ST_Area','ST_Slope','LT_Area','LT_Slope','Step_H'],
            ['Constant'],
            ['Constant', 'Slope'],
            ['a(0)', ' a(1)', 'a(2)'],
            ['a(0)', ' a(1)', 'a(2)', 'a(3)'],
            ['a(0)', ' a(1)', 'a(2)', 'a(3)','a(4)'],
            ['a(0)', ' a(1)', 'a(2)', 'a(3)','a(4)', 'a(5)']]
            

THEORY=['User Estimated Gaussians',
        'User Estimated Lorentzians',
        'User Estimated Pseudo-Voigt',
        'User Estimated Area Gaussians',
        'User Estimated Area Lorentz',
        'User Estimated Area Pseudo-Voigt',
        'User Estimated Step Down',
        'User Estimated Step Up',
        'User Estimated Slit',
        'User Estimated Atan',
        'User Estimated Hypermet',
        'User Estimated Constant',
        'User Estimated First Order Polynomial',
        'User Estimated Second Order Polynomial',
        'User Estimated Third Order Polynomial',
        'User Estimated Fourth Order Polynomial',
        'User Estimated Fifth Order Polynomial']

ESTIMATE = []
CONFIGURE = []
WIDGET = []
DERIVATIVE = []
for t in THEORY:
    ESTIMATE.append(None)
    CONFIGURE.append(None)
    WIDGET.append(None)
    DERIVATIVE.append(None)



