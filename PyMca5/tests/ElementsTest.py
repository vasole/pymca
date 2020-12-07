#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
__author__ = "V. Armando Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import unittest
import os
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
            from PyMca5 import PyMcaDataDir
            self.dataDir = PyMcaDataDir.PYMCA_DATA_DIR
        except:
            self.dataDir = None
        from PyMca5.PyMcaPhysics import Elements
        self._elements = Elements

    def testDataDirectoryPresence(self):
        # Testing directory presence
        try:
            self.assertTrue(self.dataDir is not None)
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
        from PyMca5 import getDataFile
        from PyMca5.PyMcaIO import specfile
        xcomFile = getDataFile('XCOM_CrossSections.dat')
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

    def getCrossSections(self, element, energy):
        # perform log-log interpolation in the read data
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

        log = numpy.log10

        # make sure data for the element are loaded
        # the test for proper loading is made somewhere else
        self._elements.getelementmassattcoef(element)

        # and work with them
        xcomData = self._elements.Element[element]['xcom']

        i0 = numpy.nonzero(xcomData['energy'] <= energy)[0].max()
        i1 = numpy.nonzero(xcomData['energy'] >= energy)[0].min()
        x = numpy.array(energy)
        x0 = xcomData['energy'][i0]
        x1 = xcomData['energy'][i1]
        ddict = {}
        total = 0.0
        for key in ['coherent', 'compton', 'photo']:
            y0 = xcomData[key][i0]
            y1 = xcomData[key][i1]
            if x1 != x0:
                logy = (log(y0) * log(x1/x) + log(y1) * log(x/x0))\
                               /log(x1/x0)
                y = pow(10.0, logy)
            else:
                y = y1
            ddict[key] = y
            total += y
        ddict['total'] = total
        return ddict

    def testElementCrossSectionsCalculation(self):
        if DEBUG:
            print()
            print("Testing Element Mass Attenuation Cross Sections Calculation")

        for ele in ['Ge', 'Mn', 'Au', 'U']:
            if DEBUG:
                print("Testing element = %s" % ele)
            # take a set of energies not present in the grid
            energyList = [1.0533, 2.03166, 5.82353, 10.3123, 24.7431]
            data = self._elements.getelementmassattcoef(ele,
                                                        energy=energyList)
            energyIndex = 0
            for x in energyList:
                if DEBUG:
                    print("Testing energy %f" % x)
                refData = self.getCrossSections(ele, x)
                for key in ['coherent', 'compton', 'photo', 'total']:
                    if DEBUG:
                        print("Testing key = %s" % key)
                    yRef = refData[key]
                    yTest = data[key][energyIndex]
                    self.assertTrue((100.0 * abs(yTest-yRef)/yRef) < 0.01)
                energyIndex += 1

    def testMaterialCrossSectionsCalculation(self):
        if DEBUG:
            print()
            print("Testing Material Mass Attenuation Cross Sections Calculation")

        formulae = ['H2O1', 'Hg1S1', 'Ca1C1O3']
        unpackedFormulae = [(('H', 2), ('O', 1)),
                            (('Hg', 1), ('S', 1)),
                            (('Ca', 1), ('C', 1.0), ('O', 3.0))]

        for i in range(len(unpackedFormulae)):
            if DEBUG:
                print("Testing formula %s" % formulae[i])
            # calculate mass fractions
            totalMass = 0.0
            massFractions = numpy.zeros((len(unpackedFormulae[i]),),
                                            numpy.float64)
            j = 0
            for ele, amount in unpackedFormulae[i]:
                tmpValue = amount * self._elements.Element[ele]['mass']
                totalMass += tmpValue
                massFractions[j] = tmpValue
                j += 1
            massFractions /= totalMass

            # the list of energies
            energyList = [1.5, 3.33, 10., 20.4, 30.6, 90.33]

            # get the data to be checked
            data = self._elements.getmassattcoef(formulae[i], energyList)

            energyIndex = 0
            for energy in energyList:
                if DEBUG:
                    print("Testing energy %f" % energy)
                # initialize reference data
                refData = {}
                for key in ['coherent', 'compton', 'photo', 'total']:
                    refData[key] = 0.0

                # calculate reference data
                for j in range(len(unpackedFormulae[i])):
                    ele = unpackedFormulae[i][j][0]
                    tmpData = self.getCrossSections(ele, energy)
                    for key in ['coherent', 'compton', 'photo', 'total']:
                        refData[key] += tmpData[key] * massFractions[j]

                # test
                for key in ['coherent', 'compton', 'photo', 'total']:
                    if DEBUG:
                        print("Testing key %s" % key)
                    yRef = refData[key]
                    yTest = data[key][energyIndex]
                    self.assertTrue((100.0 * abs(yTest-yRef)/yRef) < 0.01)
                energyIndex += 1

    def testMaterialCompositionCalculation(self):
        if DEBUG:
            print()
            print("Testing Material Composition Calculation (issue #298)")
        mat1 = "Kapton"
        mat2 = "Mylar"

        c1 = self._elements.getMaterialMassFractions([mat1, mat2], [0.5, 0.5])
        c2 = self._elements.getMaterialMassFractions([mat2, mat1], [0.5, 0.5])

        for key in c1:
            self.assertTrue(abs(c1[key] - c2[key]) < 1.0e-7,
                            "Inconsistent calculation for element %s" % key)

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
        testSuite.addTest(testElements("testMaterialCrossSectionsCalculation"))
        testSuite.addTest(testElements("testMaterialCompositionCalculation"))
    return testSuite

def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    DEBUG = 1
    test()
