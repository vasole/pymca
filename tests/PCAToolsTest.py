import unittest
import numpy

class testPCATools(unittest.TestCase):
    def testPCAToolsImport(self):
        from PyMca import PCATools

    def testPCAToolsCovariance(self):
        from PyMca.PCATools import getCovarianceMatrix
        x = numpy.array([[0.0,  2.0,  3.0],
                         [3.0,  0.0, -1.0],
                         [4.0, -4.0,  4.0],
                         [4.0,  4.0,  4.0]])
        nSpectra = x.shape[0]
        
        # calculate covariance using numpy
        numpyCov = numpy.cov(x.T)
        numpyAvg = x.sum(axis=0).reshape(-1, 1) / nSpectra
        tmpArray = x.T - numpyAvg
        numpyCov2 = numpy.dot(tmpArray, tmpArray.T) / nSpectra
        numpyAvg = numpyAvg.reshape(1, -1)

        # calculate covariance using PCATools and 2D stack
        pymcaCov, pymcaAvg, nData = getCovarianceMatrix(x,
                                                        force=False,
                                                        center=True)
        
        self.assertTrue(numpy.allclose(numpyCov, pymcaCov))
        self.assertTrue(numpy.allclose(numpyAvg, pymcaAvg))
        self.assertTrue(nData == nSpectra)

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

def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(\
            unittest.TestLoader().loadTestsFromTestCase(testPCATools))
    else:
        # use a predefined order
        testSuite.addTest(testPCATools("testPCAToolsImport"))
        testSuite.addTest(testPCATools("testPCAToolsCovariance"))
    return testSuite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=False))
