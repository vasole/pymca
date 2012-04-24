import unittest
import os
import sys
import numpy

DEBUG = 0

class testElements(unittest.TestCase):
    ELEMENTS = ['H', 'He', 
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

    def setUp(self):
        """
        Get the data directory
        """
        try:
            from PyMca import PyMcaDataDir
            self.dataDir = PyMcaDataDir.PYMCA_DATA_DIR
        except:
            self.dataDir = None
        from PyMca import Elements
        self._elements = Elements

    def testDataDirectoryPresence(self):
        # Testing directory presence
        try:
            self.assertIsNotNone(self.dataDir)
            self.assertTrue(os.path.exists(self.dataDir))
            self.assertTrue(os.path.isdir(self.dataDir))
        except:
            print("\n Cannot find PyMcaData directory: %s" % self.dataDir)
            raise

    def testPeakIdentification(self):
        # energy in keV
        energy = 5.9
        # 10 eV threshold
        threshold = 0.010
        lines = self._elements.getcandidates(energy,
                                             threshold=threshold,
                                             targetrays=['K'])
        self.assertTrue(len(lines[0]['elements']) == 1)
        self.assertTrue(lines[0]['energy'] == energy)
        self.assertTrue(lines[0]['elements'][0] == 'Mn')

        energy = 10.550
        threshold = 0.030
        lines = self._elements.getcandidates(energy,
                                             threshold=threshold,
                                             targetrays=['K'])
        self.assertTrue(len(lines[0]['elements']) == 1)
        self.assertTrue(lines[0]['energy'] == energy)
        self.assertTrue('As' in lines[0]['elements'])
        self.assertTrue('Pb' not in lines[0]['elements'])

        # Test K and L lines
        lines = self._elements.getcandidates(energy,
                                             threshold=threshold,
                                             targetrays=['K', 'L'])
        self.assertTrue(len(lines[0]['elements']) > 1)
        self.assertTrue('As' in lines[0]['elements'])
        self.assertTrue('Pb' in lines[0]['elements'])

        # Test all
        energy = 2.280
        threshold = 0.030
        lines = self._elements.getcandidates(energy,
                                             threshold=threshold)
        self.assertTrue(len(lines[0]['elements']) > 1)
        self.assertTrue('As' not in lines[0]['elements'])
        self.assertTrue('Pb' not in lines[0]['elements'])
        self.assertTrue('S' in lines[0]['elements'])
        self.assertTrue('Hg' in lines[0]['elements'])

    def testElementCrossSectionsReadout(self):
        if DEBUG:
            print()
            print("Test XCOM Cross Sections Readout")
        from PyMca import specfile
        xcomFile = os.path.join(self.dataDir, 'XCOM_CrossSections.dat')
        sf = specfile.Specfile(xcomFile)
        for ele in ['Si', 'Fe', 'Pb', 'U']:
            if DEBUG:
                print("Testing element %s" % ele)
            z = self._elements.getz(ele)
            scan = sf[z-1]
            xcomLabels = scan.alllabels()
            self.assertTrue('ENERGY' in xcomLabels[0].upper())
            self.assertTrue('COHERENT' in xcomLabels[1].upper())
            self.assertTrue('COMPTON' in xcomLabels[2].upper())
            self.assertTrue('PHOTO' in xcomLabels[-3].upper())
            self.assertTrue('PAIR' in xcomLabels[-2].upper())
            self.assertTrue('TOTAL' in xcomLabels[-1].upper())
            xcomData = scan.data()
            
            # WARNING: This call is to read XCOM data
            # only in case energy is None the data are the same as
            # those found later on in the 'xcom' key of the element.
            data = self._elements.getelementmassattcoef(ele, energy=None)

            # The original data are in the xcom key
            data = self._elements.Element[ele]['xcom']

            # Energy grid
            self.assertTrue(numpy.allclose(data['energy'],
                                           xcomData[0, :]))

            # Test the different cross sections
            self.assertTrue(numpy.allclose(data['coherent'],
                                           xcomData[1, :]))
            self.assertTrue(numpy.allclose(data['compton'],
                                           xcomData[2, :]))
            print((numpy.array(data['photo'])-xcomData[-3,:]).max())
            print((numpy.array(data['photo'])-xcomData[-3,:]).min())
            self.assertTrue(numpy.allclose(data['photo'],
                                           xcomData[-3, :]))
            self.assertTrue(numpy.allclose(data['pair'],
                                           xcomData[-2, :]))
            self.assertTrue(numpy.allclose(data['total'],
                                           xcomData[-1, :]))
            total = xcomData[1, :] + xcomData[2, :] +\
                    xcomData[-3, :] + xcomData[-2, :]

            # Check the total is self-consistent
            self.assertTrue(numpy.allclose(total, xcomData[-1, :]))

    def testElementCrossSectionsCalculation(self):
        if DEBUG:
            print()
            print("Testing Element Mass Attenuation Cross Sections Calculation")

        log = numpy.log10
        for ele in ['Ge', 'Mn', 'Au', 'U']:
            if DEBUG:
                print("Testing element = %s" % ele)
            # take a set of energies not present in the grid
            energyList = [1.0533, 2.03166, 5.82353, 10.3123, 24.7431]
            data = self._elements.getelementmassattcoef(ele,
                                                        energy=energyList)

            # now, perform log-log interpolation in the read data
            # to see if we get the same results
            # now perform a log-log interpolation when needed
            # lin-lin interpolation:
            #
            #              y0 (x1-x) + y1 (x-x0)
            #        y = -------------------------
            #                     x1 - x0
            #
            # log-log interpolation:
            #
            #                  log(y0) * log(x1/x) + log(y1) * log(x/x0)
            #        log(y) = ------------------------------------------
            #                                  log (x1/x0)
            #
            
            xcomData = self._elements.Element[ele]['xcom']
            energyArray = numpy.array(energyList)
            i = 0
            for x in energyList:
                if DEBUG:
                    print("Testing energy %f" % x)
                i0 = numpy.nonzero(xcomData['energy'] <= x)[0].max()
                i1 = numpy.nonzero(xcomData['energy'] >= x)[0].min()            
                x0 = xcomData['energy'][i0]
                x1 = xcomData['energy'][i1]
                total = 0.0
                for key in ['coherent', 'compton', 'photo']:
                    if DEBUG:
                        print("Testing key = %s" % key)
                    y0 = xcomData[key][i0]
                    y1 = xcomData[key][i1]
                    logy = (log(y0) * log(x1/x) + log(y1) * log(x/x0))\
                                       /log(x1/x0)
                    y = pow(10.0, logy)
                    total += y
                    self.assertTrue((100.0 * abs(data[key][i]-y)/y) < 0.01)
                y = total
                key = 'total'
                self.assertTrue((100.0 * abs(data[key][i]-y)/y) < 0.01)
                i += 1

def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(\
            unittest.TestLoader().loadTestsFromTestCase(testElements))
    else:
        testSuite.addTest(testElements("testDataDirectoryPresence"))
        testSuite.addTest(testElements("testPeakIdentification"))
        testSuite.addTest(testElements("testElementCrossSectionsReadout"))
        testSuite.addTest(testElements("testElementCrossSectionsCalculation"))
    return testSuite

if __name__ == '__main__':
    DEBUG = 1
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=False))
