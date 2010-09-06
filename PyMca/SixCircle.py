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
__author__ = "V.A. Sole - ESRF Data Analysis"
import numpy
cos = numpy.cos
sin = numpy.sin

class SixCircle(object):
    def __init__(self):
        self._energy = None
        self._lambda = None
        self._K      = 1.0
        self.setLambda(1.0)
        self.setUB([1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0])

    def setUB(self, ublist):
        self._ub = numpy.array(ublist).astype(numpy.float)
        self._ub.shape = 3, 3

    def getUB(self):
        a = self._ub * 1
        a.shape = -1
        return a.tolist()

    def setEnergy(self, energy):
        """
        setEnergy(self, energy)
        The energy has to be given in keV
        """
        self._lambda = 12.39842 / energy
        self._energy = energy
        self.update()

    def getEnergy(self):
        return self._energy

    def setLambda(self, value):
        """
        setLamabda(self, lambda)
        The wavelength has to be given in Angstroms
        """
        self._lambda = value
        self._energy = 12.39842 / value
        self.update()

    def getLambda(self):
        return self._lambda

    def update(self):
        self._K = (2*numpy.pi)/self._lambda

    def getPhiMatrix(self, phi):
        angle = phi * numpy.pi/180.
        cphi = cos(angle)
        sphi = sin(angle)
        return numpy.array([[ cphi, sphi, 0.0],
                            [-sphi, cphi, 0.0],
                            [  0.0,  0.0, 1.0]],numpy.float)

    def getChiMatrix(self, chi):
        angle = chi * numpy.pi/180.
        cchi = cos(angle)
        schi = sin(angle)
        return numpy.array([[ cchi, 0.0, schi],
                            [  0.0, 1.0,  0.0],
                            [-schi, 0.0, cchi]], numpy.float)

    def getThetaMatrix(self, th):
        angle = th * numpy.pi/180.
        cth = cos(angle)
        sth = sin(angle)
        return numpy.array([[ cth,      sth,    0],
                            [-sth,      cth,    0],
                            [   0,      0,      1]], numpy.float)

    def getDeltaMatrix(self, delta):
        angle = delta * numpy.pi/180.
        cdel = cos(angle)
        sdel = sin(angle)
        return numpy.array([[ cdel,     sdel,   0],
                            [-sdel,     cdel,   0],
                            [   0,      0,      1]], numpy.float)

    def getGammaMatrix(self, gamma):
        angle = gamma * numpy.pi/180.
        cgam = cos(angle)
        sgam = sin(angle)
        return numpy.array([[1.0,   0.0,   0.0],
                            [0.0,  cgam, -sgam],
                            [0.0,  sgam,  cgam]], numpy.float)

    def getMuMatrix(self, mu):
        angle = mu * numpy.pi/180.
        cmu = cos(angle)
        smu = sin(angle)
        return numpy.array([[1.0,   0.0,    0.0],
                            [0.0,   cmu,   -smu],
                            [0.0,   smu,    cmu]], numpy.float)


    def _getDeltaDotGammaMatrix(self, delta, gamma):
        """
        Given a 1D array of delta values and a 1D array of gamma values
        returns an array of dimension (3, 3, ndelta_values * n_gamma_values)
        """
        delr = delta * numpy.pi/180.
        gamr = gamma * numpy.pi/180.
        if 0:
            cgam, cdel = numpy.meshgrid(numpy.cos(gamr), numpy.cos(delr))
            sgam, sdel = numpy.meshgrid(numpy.sin(gamr), numpy.sin(delr))
        else:
            #this is to give the same result as Didier and not the transpose
            cdel, cgam = numpy.meshgrid(numpy.cos(delr), numpy.cos(gamr))
            sdel, sgam = numpy.meshgrid(numpy.sin(delr), numpy.sin(gamr))
        deltaDotGamma = numpy.zeros((3, 3, len(delta), len(gamma)), numpy.float)
        #1st row of dot(deltamatrix, gammaMatrix)
        deltaDotGamma[0, 0, :] =  cdel
        deltaDotGamma[0, 1, :] =  (sdel * cgam)[:]
        deltaDotGamma[0, 2, :] = -sdel * sgam
        #2nd row of dot(deltaMatrix, gammaMatrix)
        deltaDotGamma[1, 0, :] = -sdel
        deltaDotGamma[1, 1, :] =  cdel * cgam
        deltaDotGamma[1, 2, :] = -cdel * sgam
        #3rd row of dot(deltaMatrix, gammaMatrix)
        deltaDotGamma[2, 0, :] =  0.0
        deltaDotGamma[2, 1, :] =  sgam
        deltaDotGamma[2, 2, :] =  cgam
        deltaDotGamma.shape = 3, 3, len(delta)*len(gamma)
        return deltaDotGamma

    def getQMu(self, phi=0., chi=0., theta=0., mu=0., delta=0., gamma=0.):
        """
        getQMu(self, phi=0., chi=0., theta=0., mu=0., delta=0., gamma=0.)

        Angles given in degrees
        
        """
        PHIi = self.getPhiMatrix(phi).T
        CHIi = self.getChiMatrix(chi).T
        THi  = self.getThetaMatrix(theta).T
        MUi   = self.getMuMatrix(mu).T
        tmpArray = numpy.dot(PHIi,numpy.dot(CHIi,numpy.dot(THi, MUi)))
        Q = self.getQLab(mu=mu, delta=delta, gamma=gamma)
        Q.shape = 3, -1
        Q = numpy.transpose(numpy.dot(tmpArray, Q))
        if type(delta) in [type(1.0), type(1)]:
            lendelta = 1
        else:
            lendelta = len(delta)
        if type(gamma) in [type(1.0), type(1)]:
            lengamma = 1
        else:
            lengamma = len(gamma)
        Q.shape = lengamma, lendelta, 3
        return Q


    def getQSurface(self, phi=0., chi=0., theta=0., mu=0., delta=0., gamma=0.):
        """
        getQSurface(self, phi=0., chi=0., theta=0., mu=0., delta=0., gamma=0.)

        Angles given in degrees

        This is only true if the diffractometer has been properly aligned.
        
        """
        PHIi = self.getPhiMatrix(phi).T
        CHIi = self.getChiMatrix(chi).T
        THi  = self.getThetaMatrix(theta).T
        MUi   = self.getMuMatrix(mu).T
        tmpArray = numpy.dot(PHIi,numpy.dot(CHIi,numpy.dot(THi, MUi)))
        Q = self.getQLab(mu=mu, delta=delta, gamma=gamma)
        Q.shape = 3, -1
        return (numpy.dot(tmpArray, Q))

    def getQLab(self, mu=0.0, delta=0.0, gamma=0.0):
        """
        getQLab(self, mu=0.0, delta=0.0, gamma=0.0)

        Angles are given in degrees.
        
        The momentum transfer in the Lab system is

        Q = Kf - Ki = (2 * pi / lambda) * (MU DELTA GAMMA - I) * (0, 1, 0)

        This gives (transforming angles to radians):


        (2*pi/lambda) * (        sin(delta) cos(gamma),
                         cos(mu) cos(delta) cos(gamma) - sin(mu) sin(gamma) - 1,
                         sin(mu) cos(delta) cos(gamma) + cos(mu) sin(gamma))

        or, in terms of DG = numpy.dot(DELTA, GAMMA):

        (2*pi/lambda) * (         DG[0,1],
                         cos(mu)* DG[1,1] - sin(mu) * DG[2,1] - 1
                         sin(mu)* DG[1,1] + cos(mu) * DG[2,1])


        """
        alpha = mu * (numpy.pi/180.)
        cmu = cos(alpha)
        smu = sin(alpha)
        alpha = delta * (numpy.pi/180.)
        cdel = cos(alpha)
        sdel = sin(alpha)
        alpha = gamma * (numpy.pi/180.)
        cgam = cos(alpha)
        sgam = sin(alpha)
        if isinstance(delta, numpy.ndarray) and isinstance(gamma, numpy.ndarray):
            if 0:
                cgam, cdel = numpy.meshgrid(cgam, cdelr)
                sgam, sdel = numpy.meshgrid(sgamr, sdel)
            else:
                #this is to give the same result as Didier and not the transpose
                cdel, cgam = numpy.meshgrid(cdel, cgam)
                sdel, sgam = numpy.meshgrid(sdel, sgam)
            Q = numpy.zeros((3, sdel.shape[0], sdel.shape[1]), numpy.float)
            Q[0, :, :] = sdel * cgam
            Q[1, :, :] = cmu * cdel * cgam - smu * sgam - 1
            Q[2, :, :] = smu * cdel * cgam + cmu * sgam
        else:
            Q = numpy.zeros((3,1), numpy.float)
            Q[0,0] = sdel * cgam
            Q[1,0] = cmu * cdel * cgam - smu * sgam - 1
            Q[2,0] = smu * cdel * cgam + cmu * sgam
        return Q * self._K

    def getHKL(self, phi=0., chi=0., theta=0., mu=0., delta=0., gamma=0.):
        PHIi = self.getPhiMatrix(phi).T
        CHIi = self.getChiMatrix(chi).T
        THi  = self.getThetaMatrix(theta).T
        MUi   = self.getMuMatrix(mu).T
        UBi = numpy.linalg.inv(self._ub)
        tmpArray = numpy.dot(UBi,numpy.dot(PHIi,numpy.dot(CHIi,numpy.dot(THi, MUi))))
        Q = self.getQLab(mu=mu, delta=delta, gamma=gamma)
        Q.shape = 3, -1
        return (numpy.dot(tmpArray, Q))

def getHKL(wavelength, ub, phi=0., chi=0., theta=0., mu=0., delta=0., gamma=0.):
    """
    getHKL(wavelength, ub, phi=0., chi=0., theta=0., mu=0., delta=0., gamma=0.):
    A convenience function that takes the whole input in one go
    """
    a = SixCircle()
    a.setLambda(wavelength)
    a.setUB(ub)
    return a.getHKL(delta=delta, theta=theta, chi=chi, phi=phi, mu=mu, gamma=gamma)

if __name__ == "__main__":
    wavelength = 0.363504
    UB = [1.0, 0.0, 0.0, 
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0]
    UB[0] = -4.080
    UB[1] =  0.000
    UB[2] =  0.000
    UB[3] =  0.000
    UB[4] =  4.080
    UB[5] =  0.000
    UB[6] =  0.000
    UB[7] =  0.000
    UB[8] = -4.080    
    d = SixCircle()
    d.setLambda(wavelength)
    d.setUB(UB)
    print "H = 0 K = 0 L = 1"
    delta, theta, chi, phi, mu, gamma = 13.5558, 6.77779, -90, 0.0, 0.0, 0.0
    print d.getHKL(delta=delta, theta=theta, chi=chi, phi=phi, mu=mu, gamma=gamma)
    print "H = 0 K = 1 L = 0"
    delta, theta, chi, phi, mu, gamma = 13.5558, 96.77779, -90, 0.0, 0.0, 0.0
    print d.getHKL(delta=delta, theta=theta, chi=chi, phi=phi, mu=mu, gamma=gamma)
    print "H = 1 K = 1 L = 1"
    delta, theta, chi, phi, mu, gamma = 23.5910, 47.0595, -135., 0.0, 0.0, 0.0
    print d.getHKL(delta=delta, theta=theta, chi=chi, phi=phi, mu=mu, gamma=gamma)
    print "H = 2 K = -1 L = 0"
    delta, theta, chi, phi, mu, gamma = 30.6035, -11.2635, 180.0, 0.0, 0.0, 0.0
    print d.getHKL(delta=delta, theta=theta, chi=chi, phi=phi, mu=mu, gamma=gamma)

    print "H = 2 K = -1 L = 0"
    print getHKL(wavelength, UB, delta=delta, theta=theta, chi=chi, phi=phi, mu=mu, gamma=gamma)

    if 0:
        print "DIDIER Image"
        wavelength = 1.12711884437
        UB = [1.99593e-16, 2.73682e-16, -1.54, -1.08894, 1.08894, 1.6083e-16, 1.08894, 1.08894, 9.28619e-17]
        chi = 90.
        phi = -13.3
        theta = -5.53
        mu = 0.0
        gamma = 12.3
        delta = 23.23
        print getHKL(wavelength, UB, delta=delta, theta=theta, chi=chi, phi=phi, mu=mu, gamma=gamma)
    

