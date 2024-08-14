#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2019 European Synchrotron Radiation Facility
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
import numpy
from PyMca5.PyMcaMath.fitting import SpecfitFuns
arctan = numpy.arctan
exp = numpy.exp
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
        result = 0.0
        for i in range(len(pars) // 3):
            result += pars[3 * i + 0] * \
                    (0.5 + (arctan((1.0*x-pars[3 * i + 1])/pars[3 * i + 2])/pi))
        return result

    def polynomial(self, pars, x):
        result = numpy.zeros(x.shape, numpy.float64) + pars[0]
        if len(pars) == 1:
            return result
        d = x * 1.0
        for p in pars[1:]:
            result += p * d
            d *= d
        return result

    def exponential(self, pars, x):
        result = 0.0
        for i in range(len(pars) // 2):
            result += pars[2 * i + 0] * exp(float(pars[2 * i + 1]) * x)
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
          fitfuns.polynomial,
          fitfuns.exponential]

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
            ['a(0)', 'a(1)', 'a(2)'],
            ['a(0)', 'a(1)', 'a(2)', 'a(3)'],
            ['a(0)', 'a(1)', 'a(2)', 'a(3)','a(4)'],
            ['a(0)', 'a(1)', 'a(2)', 'a(3)','a(4)', 'a(5)'],
            ['Factor', 'Slope'],
            ]


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
        'User Estimated Fifth Order Polynomial',
        'User Estimated Exponential',
        ]

ESTIMATE = []
CONFIGURE = []
WIDGET = []
DERIVATIVE = []
for t in THEORY:
    ESTIMATE.append(None)
    CONFIGURE.append(None)
    WIDGET.append(None)
    DERIVATIVE.append(None)



