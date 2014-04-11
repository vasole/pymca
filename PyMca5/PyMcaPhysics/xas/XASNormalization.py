#/*##########################################################################
# Copyright (C) 2004-2014 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Software Group"
__doc__ = """This set of routines performs normalization of X-ray absorption
spectra for qualitative/preliminary analysis. For state-of-the-art XAS you
should take a look at dedicated and well-tested packages like IFEFFIT or
Viper/XANES dactyloscope """

import numpy
from PyMca5.PyMcaMath.fitting import SpecfitFuns
from PyMca5.PyMcaMath import SGModule
from PyMca.Gefit import LeastSquaresFit
DEBUG = 0
if DEBUG:
    from pylab import *

def polynom(parameter_list, x):
    if hasattr(x, 'shape'):
        output = numpy.zeros(x.shape)
    else:
        output = 0.0
    for i in range(len(parameter_list)):
        output += parameter_list[i] * pow(x, i)
    return output

def polynomDerivative(parameter_list, parameter_index, x):
    return pow(x, parameter_index)

def victoreen(parameter_list, x):
    return parameter_list[0] * pow(x, -3) + parameter_list[1] * pow(x, -4)

def victoreenDerivative(parameter_list, parameter_index, x):
    if parameter_index == 0:
        return pow(x, -3)
    else:
        return pow(x, -4)

def modifiedVictoreen(parameter_list, x):
    return parameter_list[0] * pow(x, -3) + parameter_list[1]

def modifiedVictoreenDerivative(parameter_list, parameter_index, x):
    if parameter_index == 0:
        return pow(x, -3)
    else:
        return numpy.ones(x.shape, dtype=numpy.float)

def estimateXANESEdge(spectrum, energy=None, full=False):
    if energy is None:
        x = numpy.arange(len(spectrum)).astype(numpy.float)
    else:
        x = numpy.array(energy, dtype=numpy.float, copy=True)
    y = numpy.array(spectrum, dtype=numpy.float, copy=True)

    # make sure data are sorted
    idx = energy.argsort(kind='mergesort')
    x = numpy.take(energy, idx)
    y = numpy.take(spectrum, idx)

    # make sure data are strictly increasing
    delta = x[1:] - x[:-1]
    dmin = delta.min()
    dmax = delta.max()
    if delta.min() <= 1.0e-10:
        # force data are strictly increasing
        # although we do not consider last point
        idx = numpy.nonzero(delta>0)[0]
        x = numpy.take(x, idx)
        y = numpy.take(y, idx)
        delta = None

    sortedX = x
    sortedY = y

    # use a regularly spaced spectrum
    if dmax != dmin:
        # choose the number of points or deduce it from
        # the input data length?
        nchannels = 2 * len(spectrum)
        xi = numpy.linspace(x[1], x[-2], nchannels).reshape(-1, 1)
        x.shape = -1
        y.shape = -1
        y = SpecfitFuns.interpol([x], y, xi, y.min())
        x = xi
        x.shape = -1
        y.shape = -1

    # take the first derivative
    npoints = 7
    xPrime = x[npoints:-npoints]
    yPrime = SGModule.getSavitzkyGolay(y, npoints=npoints, degree=2, order=1)
    
    # get the index at maximum value
    iMax = numpy.argmax(yPrime)

    # get the center of mass
    w = 2 * npoints
    selection = yPrime[iMax-w:iMax+w+1]
    edge = (selection * xPrime[iMax-w:iMax+w+1]).sum(dtype=numpy.float)/\
           selection.sum(dtype=numpy.float)

    if full:
        # return intermediate information
        return edge, sortedX, sortedY, xPrime, yPrime
    else:
        # return the corresponding x value
        return edge
        
def getRegionsData(x0, y0, regions, edge=0.0):
    """
    x - 1D array
    y - 1D array of the same dimension as x
    regions - List of (xmin, xmax) values defining the regions.
    edge - Supplied edge energy
           The default is 0. That means regions are absolute energies.
           The actual regions are defined as (xmin + edge, xmin + edge)
    """
    # take a view of the data
    x = x0[:]
    y = y0[:]
    
    x.shape = -1
    y.shape = -1
    
    i = 0
    for region in regions:
        xmin = region[0] + edge
        xmax = region[1] + edge
        toidx = numpy.nonzero((x >= xmin) & (x <= xmax))[0]
        if i == 0:
            i = 1
            idx = toidx
        else:
            idx = numpy.concatenate((idx, toidx), axis=0)

    xOut = numpy.take(x, idx)
    yOut = numpy.take(y, idx)

    if len(x0.shape) == 1:
        xOut.shape = -1
        yOut.shape = -1
    elif x0.shape[0] == 1:
        xOut.shape = 1, -1
        yOut.shape = 1, -1
    else:
        xOut.shape = -1, 1
        yOut.shape = -1, 1

    return xOut, yOut

def XASNormalization(spectrum,
                     energy=None,
                     edge=None,
                     pre_edge_regions=None,
                     post_edge_regions=None,
                     algorithm='polynomial',
                     algorithm_parameters=None):
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError("Unsupported algorithm %s" % algorithm)
    if energy is None:
        energy = numpy.arange(len(spectrum))
    if edge in [None, 'Auto']:
        edge = estimateXANESEdge(spectrum, energy=energy)

    if pre_edge_regions is None:
        # divide pre-edge zone in 4 regions and take the 3rd?
        if edge < 200:
            # data assumed to be in keV
            pre_edge_regions = [[-0.4, -0.05]]
        else:
            # data assumend to be in eV
            pre_edge_regions = [[-400., -50.]]

    if post_edge_regions is None:
        #divide post-edge by 20 and leave out the first one?
        if edge < 200:
            # data assumed to be in keV
            post_edge_regions = [[0.020, energy.max()-edge]]
        else:
            # data assumend to be in eV
            post_edge_regions = [[20., energy.max()-edge]]

    return SUPPORTED_ALGORITHMS[algorithm](spectrum,
                                           energy,
                                           edge,
                                           pre_edge_regions,
                                           post_edge_regions,
                                           parameters=algorithm_parameters)

def XASPolynomialNormalization(spectrum,
                             energy,
                             edge=None,
                             pre_edge_regions=None,
                             post_edge_regions=None,
                             parameters=None):
    if edge in [None, 'Auto']:
        edge = estimateXANESEdge(spectrum, energy=energy)
    if parameters is None:
        parameters = {}
    pre_edge_order = parameters.get('pre_edge_order', 1)
    post_edge_order = parameters.get('post_edge_order', 3)

    xPre, yPre = getRegionsData(energy, spectrum, pre_edge_regions, edge=edge)
    xPost, yPost = getRegionsData(energy, spectrum, post_edge_regions, edge=edge)

    # get the proper pre-edge function to be used
    pre_edge_function = polynom
    if pre_edge_order in [0, 'Constant']:
        pre_edge_order = 0
    elif pre_edge_order in [1, 'Linear']:
        pre_edge_order = 1
    elif pre_edge_order in [2, 'Parabolic']:
        pre_edge_order = 2
    elif pre_edge_order in [3, 'Cubic']:
        pre_edge_order = 3
    elif pre_edge_order in [-1, 'Victoreen']:
        pre_edge_order = -1
        pre_edge_function = victoreen
    elif pre_edge_order in [-2, 'Modif. Victoreen']:
        pre_edge_order = -2
        pre_edge_function = modifiedVictoreen
    else:
        # case of arriving with a 4th order polynom, for instance
        pass

    # calculate pre-edge
    if pre_edge_order == 0:
        prePol = [yPre.mean()]
    elif pre_edge_order > 0:
        p = numpy.arange(pre_edge_order + 1).astype(numpy.float)
        prePol = LeastSquaresFit(pre_edge_function, p,
                                 xdata=xPre, ydata=yPre,
                                 model_deriv=polynomDerivative,
                                 weightflag=0, linear=1)[0]
    elif pre_edge_order == -1:
        p = numpy.array([1.0, 1.0])
        prePol = LeastSquaresFit(pre_edge_function, p,
                                 xdata=xPre, ydata=yPre,
                                 model_deriv=victoreenDerivative,
                                 weightflag=0, linear=1)[0]
    elif pre_edge_order == -2:
        p = numpy.array([1.0, 1.0])
        prePol = LeastSquaresFit(pre_edge_function, p,
                                 xdata=xPre, ydata=yPre,
                                 model_deriv=modifiedVictoreenDerivative,
                                 weightflag=0, linear=1)[0]

    # get the proper post-edge function to be used
    post_edge_function = polynom
    if post_edge_order in [0, 'Constant']:
        post_edge_order = 0
    elif post_edge_order in [1, 'Linear']:
        post_edge_order = 1
    elif post_edge_order in [2, 'Parabolic']:
        post_edge_order = 2
    elif post_edge_order in [3, 'Cubic']:
        post_edge_order = 3
    elif post_edge_order in [-1, 'Victoreen']:
        post_edge_order = -1
        post_edge_function = victoreen
    elif post_edge_order in [-2, 'Modif. Victoreen']:
        post_edge_order = -2
        post_edge_function = modifiedVictoreen
    else:
        # case of arriving with a 4th order polynom, for instance
        pass

    # calculate post-edge
    baseLine = pre_edge_function(prePol, xPost)
    if post_edge_order == 0:
        # just take the average
        postPol = [(yPost-baseLine).mean()]
        normalizedSpectrum = (spectrum - pre_edge_function(prePol, energy))/postPol[0]
    elif post_edge_order > 0:
        p = numpy.arange(post_edge_order + 1).astype(numpy.float)
        postPol = LeastSquaresFit(post_edge_function, p,
                                  xdata=xPost,
                                  ydata=yPost-baseLine,
                                  model_deriv=polynomDerivative,
                                  weightflag=0, linear=1)[0]
        normalizedSpectrum = (spectrum - pre_edge_function(prePol, energy))\
                             /post_edge_function(postPol, energy)
    elif post_edge_order == -1:
        p = numpy.array([1.0, 1.0])
        postPol = LeastSquaresFit(post_edge_function, p,
                                  xdata=xPost,
                                  ydata=yPost-baseLine,
                                  model_deriv=victoreenDerivative,
                                  weightflag=0, linear=1)[0]
        normalizedSpectrum = (spectrum - pre_edge_function(prePol, energy))\
                             /post_edge_function(postPol, energy)
    elif post_edge_order == -2:
        p = numpy.array([1.0, 1.0])
        postPol = LeastSquaresFit(post_edge_function, p,
                                  xdata=xPost,
                                  ydata=yPost-baseLine,
                                  model_deriv=modifiedVictoreenDerivative,
                                  weightflag=0, linear=1)[0]
        normalizedSpectrum = (spectrum - pre_edge_function(prePol, energy))\
                             /post_edge_function(postPol, energy)
    jump = post_edge_function(postPol, edge)
    if DEBUG:
        plot(energy, spectrum, 'o')
        plot(xPre, pre_edge_function(prePol, xPre), 'r')
        plot(xPost, post_edge_function(postPol, xPost)+pre_edge_function(prePol, xPost), 'y')
        show()
    return energy, normalizedSpectrum, edge, jump, pre_edge_function, prePol, post_edge_function, postPol

def XASVictoreenNormalization(spectrum,
                              energy,
                              edge=None,
                              pre_edge_regions=None,
                              post_edge_regions=None,
                              parameters=None):

    if edge in [None, 'Auto']:
        edge = estimateXANESEdge(spectrum, energy=energy)

    if parameters is None:
        parameters = {}

    xPre, yPre = getRegionsData(energy, spectrum, pre_edge_regions)
    xPost, yPost = getRegionsData(energy, spectrum, post_edge_regions)


    pre_edge_order = parameters.get('pre_edge_order', 1)
    post_edge_order = parameters.get('post_edge_order', 1)
    if pre_edge_order in [1, -1, 'Victoreen']:
        pre_edge_function = victoreen
    else:
        pre_edge_function = modifiedVictoreen

    if post_edge_order in [1, -1, 'Victoreen']:
        post_edge_function = victoreen
    else:
        post_edge_function = modifiedVictoreen

    p = numpy.array([1.0, 1.0])
    prePol = LeastSquaresFit(pre_edge_function, p, xdata=xPre, ydata=yPre,
                                 weightflag=0, linear=1)[0]    
    postPol = LeastSquaresFit(post_edge_function, p,
                              xdata=xPost,
                              ydata=yPost-pre_edge_function(prePol, xPost),
                              weightflag=0, linear=1)[0]
    normalizedSpectrum = (spectrum - pre_edge_function(prePol, energy))\
                         /post_edge_function(postPol, energy)
    if DEBUG:
        print("VICTOREEN")
        plot(energy, spectrum, 'o')
        plot(xPre, pre_edge_function(prePol, xPre), 'r')
        plot(xPost,
             post_edge_function(postPol, xPost)+pre_edge_function(prePol, xPost), 'y')
        show()
    return energy, normalizedSpectrum, edge       

SUPPORTED_ALGORITHMS = {"polynomial":XASPolynomialNormalization,
                        "victoreen": XASVictoreenNormalization}

if __name__ == "__main__":
    import sys
    from PyMca.PyMcaIO import specfilewrapper as specfile
    import time
    sf = specfile.Specfile(sys.argv[1])
    scan = sf[0]
    data = scan.data()
    energy = data[0, :]
    spectrum = data[1, :]
    n = 100
    t0 = time.time()
    for i in range(n):
        edge = estimateXANESEdge(spectrum+i, energy=energy)
    print("EDGE ELAPSED = ", (time.time() - t0)/float(n))
    print("EDGE = %f"  % edge)
    if DEBUG:
        n = 1
    else:
        n = 100
    t0 = time.time()
    for i in range(n):
        nEne0, nSpe0 = XASNormalization(spectrum+i, energy,
                                        edge=edge,
                                        algorithm='polynomial',
                                        algorithm_parameters={'pre_edge_order':0,
                                                              'post_edge_order':0})[0:2]
    print("ELAPSED 0 = ", (time.time() - t0)/float(n))
    t0 = time.time()
    for i in range(n):
        nEneP, nSpeP = XASNormalization(spectrum+i,
                                        energy,
                                        edge=edge,
                                        algorithm='polynomial',
                                        algorithm_parameters={'pre_edge_order':1,
                                                              'post_edge_order':2})[0:2]
    print("ELAPSED Poly = ", (time.time() - t0)/float(n))
    t0 = time.time()
    for i in range(n):
        nEneV, nSpeV = XASNormalization(spectrum+i,
                                        energy,
                                        edge=edge,
                                        algorithm='polynomial',
                                        algorithm_parameters={'pre_edge_order':'Victoreen',
                                                              'post_edge_order':'Victoreen'})[0:2]
    print("ELAPSED Victoreen = ", (time.time() - t0)/float(n))
    if DEBUG:
        #plot(energy, spectrum, 'b')
        plot(nEne0, nSpe0, 'k', label='Polynomial')
        plot(nEneP, nSpeP, 'b', label='Polynomial')
        plot(nEneV, nSpeV, 'r', label='Victoreen')
        show()
