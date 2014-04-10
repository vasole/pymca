#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
import unittest
import numpy
import numpy.linalg
try:
    import mdp
    MDP = True
except:
    # MDP can give very weird errors
    MDP = False

class testPCATools(unittest.TestCase):
    def testPCAToolsImport(self):
        from PyMca import PCATools

    def testPCAToolsCovariance(self):
        from PyMca5.PCATools import getCovarianceMatrix
        x = numpy.array([[0.0,  2.0,  3.0],
                         [3.0,  0.0, -1.0],
                         [4.0, -4.0,  4.0],
                         [4.0,  4.0,  4.0]])
        nSpectra = x.shape[0]

        # test just multiplication
        tmpArray = numpy.dot(x.T, x)
        for force in [True, False]:
            pymcaCov, pymcaAvg, nData = getCovarianceMatrix(x,
                                                            force=force,
                                                            center=False)
            self.assertTrue(numpy.allclose(tmpArray, pymcaCov * (nData - 1)))
        
        # calculate covariance using numpy
        numpyCov = numpy.cov(x.T)
        numpyAvg = x.sum(axis=0).reshape(-1, 1) / nSpectra
        tmpArray = x.T - numpyAvg
        numpyCov2 = numpy.dot(tmpArray, tmpArray.T) / nSpectra
        numpyAvg = numpyAvg.reshape(1, -1)

        # calculate covariance using PCATools and 2D stack
        # directly and dynamically loading data
        for force in [False, True]:
            pymcaCov, pymcaAvg, nData = getCovarianceMatrix(x,
                                                            force=force,
                                                            center=True)
        
            self.assertTrue(numpy.allclose(numpyCov, pymcaCov))
            self.assertTrue(numpy.allclose(numpyAvg, pymcaAvg))
            self.assertTrue(nData == nSpectra)

        # calculate covariance using PCATools and 3D stack
        # directly and dynamically loading data
        x.shape = 2, 2, -1
        for force in [False, True]:
            pymcaCov, pymcaAvg, nData = getCovarianceMatrix(x,
                                                            force=force,
                                                            center=True)

            self.assertTrue(numpy.allclose(numpyCov, pymcaCov))
            self.assertTrue(numpy.allclose(numpyAvg, pymcaAvg))
            self.assertTrue(nData == nSpectra)

    def testPCAToolsPCA(self):
        from PyMca5.PCATools import numpyPCA
        x = numpy.array([[0.0,  2.0,  3.0],
                         [3.0,  0.0, -1.0],
                         [4.0, -4.0,  4.0],
                         [4.0,  4.0,  4.0]])

        # that corresponds to 4 spectra of 3 channels
        nSpectra = x.shape[0]

        # calculate eigenvalues and eigenvectors with numpy
        tmpArray = numpy.dot(x.T, x)/(nSpectra - 1)
        numpyEigenvalues, numpyEigenvectors = numpy.linalg.eigh(tmpArray)

        # sort from higher to lower
        idx = list(range(numpyEigenvalues.shape[0]-1, -1 , -1))
        numpyEigenvalues = numpy.take(numpyEigenvalues, idx)
        numpyEigenvectors = numpyEigenvectors[:, ::-1].T

        # now use PyMca
        # centering has to be false to obtain the same results
        for force in [True, False]:
            images, eigenvalues, eigenvectors = numpyPCA(x,
                                                         ncomponents=x.shape[1],
                                                         force=force,
                                                         center=False,
                                                         scale=False)
            self.assertTrue(numpy.allclose(eigenvalues, numpyEigenvalues))
            self.assertTrue(numpy.allclose(eigenvectors, numpyEigenvectors))

        # test with a different shape
        x.shape = 2, 2, -1
        for force in [True, False]:
            images, eigenvalues, eigenvectors = numpyPCA(x,
                                                         ncomponents=3,
                                                         force=force,
                                                         center=False,
                                                         scale=False)
            self.assertTrue(numpy.allclose(eigenvalues, numpyEigenvalues))
            self.assertTrue(numpy.allclose(eigenvectors, numpyEigenvectors))

    if MDP:
        def testPCAToolsMDP(self):
            from PyMca5.PCATools import getCovarianceMatrix, numpyPCA
            x = numpy.array([[0.0,  2.0,  3.0],
                             [3.0,  0.0, -1.0],
                             [4.0, -4.0,  4.0],
                             [4.0,  4.0,  4.0]])

            # use mdp
            pcaNode = mdp.nodes.PCANode()
            pcaNode.train(x)
            pcaNode.stop_training()
            pcaEigenvectors = pcaNode.v.T

            # and compare with PyMca
            for force in [True, False]:
                images, eigenvalues, eigenvectors = numpyPCA(x,
                                                        ncomponents=x.shape[1],
                                                        force=force,
                                                        center=True,
                                                        scale=False)

                # the eigenvalues must be the same
                self.assertTrue(numpy.allclose(eigenvalues, pcaNode.d))
                # the eigenvectors can be multiplied by -1
                for i in range(x.shape[1]):
                    if (eigenvectors[i,0] >= 0 and pcaEigenvectors[i,0] >=0) or\
                       (eigenvectors[i,0] <= 0 and pcaEigenvectors[i,0] <=0):
                        # both same sign
                        self.assertTrue(numpy.allclose(eigenvectors[i],
                                                       pcaEigenvectors[i]))
                    else:
                        self.assertTrue(numpy.allclose(-eigenvectors[i],
                                                       pcaEigenvectors[i]))

def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(\
            unittest.TestLoader().loadTestsFromTestCase(testPCATools))
    else:
        # use a predefined order
        testSuite.addTest(testPCATools("testPCAToolsImport"))
        testSuite.addTest(testPCATools("testPCAToolsCovariance"))
        testSuite.addTest(testPCATools("testPCAToolsPCA"))
        if MDP:
            testSuite.addTest(testPCATools("testPCAToolsMDP"))
    return testSuite

def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    test()
