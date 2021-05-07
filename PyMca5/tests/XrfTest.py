#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
import sys
import copy
import numpy
if sys.version_info < (3,):
    from StringIO import StringIO
else:
    from io import StringIO

cfg = """[attenuators]
kapton = 0, -, 0.0, 0.0, 1.0
atmosphere = 1, Air, 0.00120479, 0.14, 1.0
Matrix = 1, Sample, 1.0, 0.01, 0.1, 90.0, 0, 90.1
deadlayer = 0, Si1, 2.33, 4.5e-06, 1.0
BeamFilter1 = 0, -, 0.0, 0.0, 1.0
BeamFilter0 = 0, -, 0.0, 0.0, 1.0
absorber = 0, -, 0.0, 0.0, 1.0
window = 1, Be1, 1.85, 0.0008, 1.0
contact = 0, Al1, 2.72, 3e-06, 1.0
Filter 6 = 0, -, 0.0, 0.0, 1.0
Filter 7 = 0, -, 0.0, 0.0, 1.0
Detector = 1, Si1, 2.33, 0.045, 1.0

[peaks]
Ni = K
Zn = K, L
Co = K
Sr = K, L
Ca = K
Mn = K
As = K, L
Cd = L
Pb = L, M
Tl = L, M
Ar = K
Ti = K
Fe = K
V = K
Sb = L
Cu = K, L
Se = K, L
Cr = K

[fit]
stripwidth = 10
linearfitflag = 1
xmin = 290
scatterflag = 0
snipwidth = 20
stripfilterwidth = 4
escapeflag = 1
exppolorder = 6
fitweight = 1
stripflag = 1
stripanchorsflag = 0
use_limit = 1
maxiter = 10
stripiterations = 6000
sumflag = 0
linpolorder = 5
stripalgorithm = 0
deltaonepeak = 0.01
deltachi = 0.001
continuum = 0
hypermetflag = 1
stripconstant = 1.0
xmax = 3400
fitfunction = 0
energy = 17.5, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
stripanchorslist = 3400, 290, 0, 0
energyscatter = 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
energyweight = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
energyflag = 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

[multilayer]
Layer3 = 0, -, 0.0, 0.0
Layer2 = 0, -, 0.0, 0.0
Layer1 = 0, -, 0.0, 0.0
Layer0 = 1, Water, 1.0, 0.01
Layer7 = 0, -, 0.0, 0.0
Layer6 = 0, -, 0.0, 0.0
Layer5 = 0, -, 0.0, 0.0
Layer4 = 0, -, 0.0, 0.0
Layer9 = 0, -, 0.0, 0.0
Layer8 = 0, -, 0.0, 0.0

[tube]
windowdensity = 1.848
anodedensity = 10.5
windowthickness = 0.0125
anodethickness = 0.0002
transmission = 0
alphax = 90.0
deltaplotting = 0.1
window = Be
filter1thickness = 0.0
anode = Ag
voltage = 30.0
filter1density = 0.000118
alphae = 90.0
filter1 = He

[materials]

[materials.Kapton]
Comment = Kapton 100 HN 25 micron density=1.42 g/cm3
Thickness = 0.0025
Density = 1.42
CompoundFraction = 0.628772, 0.066659, 0.304569
CompoundList = C1, N1, O1

[materials.Teflon]
Comment = Teflon density=2.2 g/cm3
Density = 2.2
CompoundFraction = 0.240183, 0.759817
CompoundList = C1, F1

[materials.Gold]
Comment = Gold
CompoundFraction = 1.0
Thickness = 1e-06
Density = 19.37
CompoundList = Au

[materials.Water]
Comment = Water density=1.0 g/cm3
CompoundFraction = 1.0
Density = 1.0
CompoundList = H2O1

[materials.Sample]
Comment = Water with 500 ppm Co
Thickness = 0.01
Density = 0.1
CompoundFraction = 0.9995, 0.0005
CompoundList = H2O1, Co

[materials.Air]
Comment = Dry Air (Near sea level) density=0.001204790 g/cm3
Thickness = 1.0
Density = 0.0012048
CompoundFraction = 0.000124, 0.75527, 0.23178, 0.012827, 3.2e-06
CompoundList = C1, N1, O1, Ar1, Kr1

[materials.Mylar]
Comment = Mylar (Polyethylene Terephthalate) density=1.40 g/cm3
Density = 1.4
CompoundFraction = 0.041959, 0.625017, 0.333025
CompoundList = H1, C1, O1

[materials.Viton]
Comment = Viton Fluoroelastomer density=1.8 g/cm3
Density = 1.8
CompoundFraction = 0.009417, 0.280555, 0.710028
CompoundList = H1, C1, F1

[concentrations]
usemultilayersecondary = 0
reference = Co
area = 0.10
flux = 190000.0
time = 600.0
useattenuators = 1
usematrix = 1
mmolarflag = 0
distance = 0.3

[detector]
noise = 0.0781703
fixednoise = 0
fixedgain = 0
deltafano = 0.114
fixedfano = 0
sum = 0.0
deltasum = 1e-08
fano = 0.120159
fixedsum = 0
fixedzero = 0
zero = -0.492773
deltazero = 0.1
deltanoise = 0.05
deltagain = 0.001
detele = Si
nthreshold = 4
gain = 0.00502883

[peakshape]
lt_arearatio = 0.2
fixedlt_arearatio = 0
fixedeta_factor = 0
st_arearatio = 0.04
deltalt_arearatio = 0.015
deltaeta_factor = 0.2
deltalt_sloperatio = 7.0
deltastep_heightratio = 5e-05
st_sloperatio = 0.6
lt_sloperatio = 10.0
fixedlt_sloperatio = 0
deltast_arearatio = 0.03
eta_factor = 0.2
fixedst_sloperatio = 0
fixedst_arearatio = 0
deltast_sloperatio = 0.49
step_heightratio = 0.0005
fixedstep_heightratio = 0"""


class testXrf(unittest.TestCase):
    def setUp(self):
        """
        Get the data directory
        """
        self._importSuccess = False
        try:
            from PyMca5 import PyMcaDataDir
            self._importSuccess = True
            self.dataDir = PyMcaDataDir.PYMCA_DATA_DIR
        except:
            self.dataDir = None

    def testTrainingDataDirectoryPresence(self):
        self.assertTrue(self._importSuccess,
                        'Unsuccessful PyMca5.PyMcaDataDir import')
        self.assertTrue(self.dataDir is not None,
                        'Unassigned PyMca5.PyMcaDataDir.PYMCA_DATA_DIR')
        self.assertTrue(os.path.exists(self.dataDir),
                        'Directory "%s" does not exist' % self.dataDir)
        self.assertTrue(os.path.isdir(self.dataDir),
                        '"%s" expected to be a directory' % self.dataDir)

    def testTrainingDataFilePresence(self):
        trainingDataFile = os.path.join(self.dataDir, "XRFSpectrum.mca")
        self.assertTrue(os.path.exists(trainingDataFile),
                        "File %s does not exists" % trainingDataFile)
        self.assertTrue(os.path.isfile(trainingDataFile),
                        "File %s is not an actual file" % trainingDataFile)

    def _readTrainingData(self):
        from PyMca5.PyMcaIO import specfilewrapper as specfile
        from PyMca5.PyMcaIO import ConfigDict
        trainingDataFile = os.path.join(self.dataDir, "XRFSpectrum.mca")
        self.assertTrue(os.path.isfile(trainingDataFile),
                        "File %s is not an actual file" % trainingDataFile)

        sf = specfile.Specfile(trainingDataFile)
        self.assertTrue(len(sf) == 2,
                        "Training data not interpreted as two scans")
        self.assertTrue(sf[0].nbmca() == 0,
                        "Training data 1st scan should contain no MCAs")
        self.assertTrue(sf[1].nbmca() == 1,
                        "Training data 1st scan should contain no MCAs")
        y = mcaData = sf[1].mca(1)
        sf = None
        x = numpy.arange(y.size).astype(numpy.float64)

        # perform the actual XRF analysis
        configuration = ConfigDict.ConfigDict()
        configuration.readfp(StringIO(cfg))

        return x, y, configuration

    def _readStainlessSteelData(self):
        from PyMca5.PyMcaIO import specfilewrapper as specfile
        from PyMca5.PyMcaIO import ConfigDict

        # read the data
        dataFile = os.path.join(self.dataDir, "Steel.spe")
        self.assertTrue(os.path.isfile(dataFile),
                        "File %s is not an actual file" % dataFile)
        sf = specfile.Specfile(dataFile)
        self.assertTrue(len(sf) == 1, "File %s cannot be read" % dataFile)
        self.assertTrue(sf[0].nbmca() == 1,
                        "Spe file should contain MCA data")
        y = counts = sf[0].mca(1)
        x = channels = numpy.arange(y.size).astype(numpy.float64)
        sf = None

        # read the fit configuration
        configFile = os.path.join(self.dataDir, "Steel.cfg")
        self.assertTrue(os.path.isfile(configFile),
                        "File %s is not an actual file" % configFile)
        configuration = ConfigDict.ConfigDict()
        configuration.read(configFile)
        # configure the fit
        # make sure no secondary excitations are used
        configuration["concentrations"]["usemultilayersecondary"] = 0

        return x, y, configuration

    def testTrainingDataFit(self):
        from PyMca5.PyMcaIO import specfilewrapper as specfile
        from PyMca5.PyMcaPhysics.xrf import LegacyMcaTheory
        from PyMca5.PyMcaPhysics.xrf import ConcentrationsTool

        x, y, configuration = self._readTrainingData()

        # perform the actual XRF analysis
        mcaFit = LegacyMcaTheory.LegacyMcaTheory()
        configuration, fitResult, result = self._configAndFit(x, y, configuration, mcaFit)

        # fit is already done, calculate the concentrations
        concentrationsConfiguration = configuration["concentrations"]
        cTool = ConcentrationsTool.ConcentrationsTool()
        cToolConfiguration = cTool.configure()
        cToolConfiguration.update(configuration['concentrations'])
        # make sure we are using Co as internal standard
        cToolConfiguration["usematrix"] = 1
        cToolConfiguration["reference"] = "Co"
        concentrationsResult, addInfo = cTool.processFitResult( \
                    config=cToolConfiguration,
                    fitresult={"result":result},
                    elementsfrommatrix=False,
                    fluorates = mcaFit._fluoRates,
                    addinfo=True)
        referenceElement = addInfo['ReferenceElement']
        referenceTransitions = addInfo['ReferenceTransitions']
        self.assertTrue(referenceElement == "Co",
               "referenceElement is <%s> instead of <Co>" % referenceElement)
        cobalt = concentrationsResult["mass fraction"]["Co K"]
        self.assertTrue( abs(cobalt-0.0005) < 1.0E-7,
                        "Wrong Co concentration %f expected 0.0005" % cobalt)

        # we should get the same result with internal parameters
        cTool = ConcentrationsTool.ConcentrationsTool()
        cToolConfiguration = cTool.configure()
        cToolConfiguration.update(configuration['concentrations'])

        # make sure we are not using an internal standard
        cToolConfiguration['usematrix'] = 0
        cToolConfiguration['flux'] = addInfo["Flux"]
        cToolConfiguration['time'] = addInfo["Time"]
        cToolConfiguration['area'] = addInfo["DetectorArea"]
        cToolConfiguration['distance'] = addInfo["DetectorDistance"]
        concentrationsResult2, addInfo = cTool.processFitResult( \
                    config=cToolConfiguration,
                    fitresult={"result":result},
                    elementsfrommatrix=False,
                    fluorates = mcaFit._fluoRates,
                    addinfo=True)
        referenceElement = addInfo['ReferenceElement']
        referenceTransitions = addInfo['ReferenceTransitions']
        self.assertTrue(referenceElement in ["None", "", None],
               "referenceElement is <%s> instead of <None>" % referenceElement)

        for key in concentrationsResult["mass fraction"]:
            internal = concentrationsResult["mass fraction"][key]
            fp = concentrationsResult2["mass fraction"][key]
            delta = 100 * (abs(internal - fp) / internal)
            self.assertTrue( delta < 1.0e-5,
                "Error for <%s> concentration %g != %g" % (key, internal, fp))

    def testStainlessSteelDataFit(self):
        from PyMca5.PyMcaPhysics.xrf import LegacyMcaTheory
        from PyMca5.PyMcaPhysics.xrf import ConcentrationsTool

        x, y, configuration = self._readStainlessSteelData()

        # configure the fit
        mcaFit = LegacyMcaTheory.LegacyMcaTheory()
        configuration, fitResult, result = self._configAndFit(x, y, configuration, mcaFit)

        # concentrations
        # fit is already done, calculate the concentrations
        concentrationsConfiguration = configuration["concentrations"]
        cTool = ConcentrationsTool.ConcentrationsTool()
        cToolConfiguration = cTool.configure()
        cToolConfiguration.update(configuration['concentrations'])

        # make sure we are using Fe as internal standard
        matrix = configuration["attenuators"]["Matrix"]
        self.assertTrue(matrix[1] == "SRM_1155",
                "Invalid matrix. Expected <SRM_1155> got <%s>" % matrix[1])
        cToolConfiguration["usematrix"] = 1
        cToolConfiguration["reference"] = "Fe"
        concentrationsResult, addInfo = cTool.processFitResult( \
                    config=cToolConfiguration,
                    fitresult={"result":result},
                    elementsfrommatrix=False,
                    fluorates = mcaFit._fluoRates,
                    addinfo=True)
        referenceElement = addInfo['ReferenceElement']
        referenceTransitions = addInfo['ReferenceTransitions']
        self.assertTrue(referenceElement == "Fe",
               "referenceElement is <%s> instead of <Fe>" % referenceElement)

        # check the Fe concentration is 0.65 +/ 5 %
        self.assertTrue( \
            abs(concentrationsResult["mass fraction"]["Fe Ka"] - 0.65) < 0.03,
            "Invalid Fe Concentration")
        # check the Cr concentration is overestimated (more than 30 %) %
        testValue = concentrationsResult["mass fraction"]["Cr K"]
        self.assertTrue( testValue > 0.30,
            "Expected Cr concentration above 0.30 got %.3f" % testValue)

        # chek the sum of concentration of main components is above 1
        # because of neglecting higher order excitations
        elements = ["Cr K", "V K", "Mn K", "Fe Ka", "Ni K"]
        total = 0.0
        for element in elements:
            total += concentrationsResult["mass fraction"][element]
        self.assertTrue(total > 1,
                    "Sum of concentrations should be above 1 got %.3f" % total)

        # correct for tertiary excitation without a new fit
        cToolConfiguration["usemultilayersecondary"] = 2
        concentrationsResult, addInfo = cTool.processFitResult( \
                    config=cToolConfiguration,
                    fitresult={"result":result},
                    elementsfrommatrix=False,
                    fluorates = mcaFit._fluoRates,
                    addinfo=True)

        # check the Fe concentration is 0.65 +/ 5 %
        self.assertTrue( \
            abs(concentrationsResult["mass fraction"]["Fe Ka"] - 0.65) < 0.03,
            "Invalid Fe Concentration Using Tertiary Excitation")

        # chek the sum of concentration of main components is above 1
        elements = ["Cr K", "Mn K", "Fe Ka", "Ni K"]
        total = 0.0
        for element in elements:
            total += concentrationsResult["mass fraction"][element]
        self.assertTrue(total < 1,
                   "Sum of concentrations should be below 1 got %.3f" % total)
        # check the Cr concentration is not overestimated (more than 30 %) %
        testValue = concentrationsResult["mass fraction"]["Cr K"]
        self.assertTrue( (testValue > 0.18) and (testValue < 0.20),
            "Expected Cr between 0.18 and 0.20 got %.3f" % testValue)

        # perform the fit already accounting for tertiary excitation
        # in order to get the good fundamental parameters
        configuration["concentrations"]['usematrix'] = 1
        configuration["concentrations"]["usemultilayersecondary"] = 2
        configuration, fitResult, result = self._configAndFit(x, y, configuration, mcaFit)

        # concentrations
        # fit is already done, calculate the concentrations
        concentrationsConfiguration = configuration["concentrations"]
        cTool = ConcentrationsTool.ConcentrationsTool()
        cToolConfiguration = cTool.configure()
        cToolConfiguration.update(configuration['concentrations'])
        matrix = configuration["attenuators"]["Matrix"]
        self.assertTrue(matrix[1] == "SRM_1155",
                "Invalid matrix. Expected <SRM_1155> got <%s>" % matrix[1])
        cToolConfiguration["usematrix"] = 1
        cToolConfiguration["reference"] = "Fe"
        concentrationsResult, addInfo = cTool.processFitResult( \
                    config=cToolConfiguration,
                    fitresult={"result":result},
                    elementsfrommatrix=False,
                    fluorates = mcaFit._fluoRates,
                    addinfo=True)

        # make sure we are not using an internal standard
        # repeat everything using a single layer strategy
        configuration["concentrations"]['usematrix'] = 0
        configuration["concentrations"]['flux'] = addInfo["Flux"]
        configuration["concentrations"]['time'] = addInfo["Time"]
        configuration["concentrations"]['area'] = addInfo["DetectorArea"]
        configuration["concentrations"]['distance'] = \
                                                    addInfo["DetectorDistance"]
        configuration["concentrations"]["usemultilayersecondary"] = 2

        # setup the strategy starting with Fe as matrix
        matrix[1] = "Fe"
        configuration["attenuators"]["Matrix"] = matrix
        configuration["fit"]["strategyflag"] = 1
        configuration["fit"]["strategy"] = "SingleLayerStrategy"
        configuration["SingleLayerStrategy"] = {}
        configuration["SingleLayerStrategy"]["layer"] = "Auto"
        configuration["SingleLayerStrategy"]["iterations"] = 3
        configuration["SingleLayerStrategy"]["completer"] = "-"
        configuration["SingleLayerStrategy"]["flags"] = [1, 1, 1, 1, 0,
                                                         0, 0, 0, 0, 0]
        configuration["SingleLayerStrategy"]["peaks"] = [ "Cr K",
                                                         "Mn K", "Fe Ka",
                                                         "Ni K", "-", "-",
                                                         "-","-","-","-"]
        configuration["SingleLayerStrategy"]["materials"] = ["Cr",
                                                         "Mn", "Fe",
                                                         "Ni", "-", "-",
                                                         "-","-","-"]
        mcaFit = LegacyMcaTheory.LegacyMcaTheory()
        configuration, fitResult, result = self._configAndFit(x, y, configuration, mcaFit)

        # concentrations
        # fit is already done, calculate the concentrations
        concentrationsConfiguration = configuration["concentrations"]
        cTool = ConcentrationsTool.ConcentrationsTool()
        cToolConfiguration = cTool.configure()
        cToolConfiguration.update(configuration['concentrations'])
        concentrationsResult2, addInfo = cTool.processFitResult( \
                    config=cToolConfiguration,
                    fitresult={"result":result},
                    elementsfrommatrix=False,
                    fluorates = mcaFit._fluoRates,
                    addinfo=True)

        # chek the sum of concentration of main components is above 1
        elements = ["Cr K", "Mn K", "Fe Ka", "Ni K"]
        total = 0.0
        for element in elements:
            if element == "Cr K":
                tolerance = 6 # 6 %
            else:
                tolerance = 5 # 5 %
            previous = concentrationsResult["mass fraction"][element]
            current = concentrationsResult2["mass fraction"][element]
            delta = 100 * (abs(previous - current) / previous)
            self.assertTrue(delta < tolerance,
                "Strategy: Element %s discrepancy too large %.1f %%" % \
                  (element.split()[0], delta))

    def testLegacyMcaTheory(self):
        x, y, configuration = self._readTrainingData()
        self._testLegacyMcaTheory(x, y, configuration)

        x, y, configuration = self._readStainlessSteelData()

        configuration["concentrations"]['usematrix'] = 0
        configuration["concentrations"]["usemultilayersecondary"] = 0
        self._testLegacyMcaTheory(x, y, configuration)

        configuration["concentrations"]['usematrix'] = 1
        configuration["concentrations"]["usemultilayersecondary"] = 2
        self._testLegacyMcaTheory(x, y, configuration)

        configuration["concentrations"]['usematrix'] = 0
        configuration["concentrations"]["usemultilayersecondary"] = 2
        configuration["attenuators"]["Matrix"] = [1, 'Fe', 1.0, 1.0, 45.0, 45.0]
        configuration["fit"]["strategyflag"] = 1
        configuration["fit"]["strategy"] = "SingleLayerStrategy"
        configuration["SingleLayerStrategy"] = {}
        configuration["SingleLayerStrategy"]["layer"] = "Auto"
        configuration["SingleLayerStrategy"]["iterations"] = 3
        configuration["SingleLayerStrategy"]["completer"] = "-"
        configuration["SingleLayerStrategy"]["flags"] = [1, 1, 1, 1, 0,
                                                         0, 0, 0, 0, 0]
        configuration["SingleLayerStrategy"]["peaks"] = [ "Cr K",
                                                         "Mn K", "Fe Ka",
                                                         "Ni K", "-", "-",
                                                         "-","-","-","-"]
        configuration["SingleLayerStrategy"]["materials"] = ["Cr",
                                                         "Mn", "Fe",
                                                         "Ni", "-", "-",
                                                         "-","-","-"]
        self._testLegacyMcaTheory(x, y, configuration)

    def _testLegacyMcaTheory(self, x, y, configuration):
        from PyMca5.PyMcaPhysics.xrf import LegacyMcaTheory
        from PyMca5.PyMcaPhysics.xrf import NewClassMcaTheory

        import time
        t0 = time.time()

        mcaFitLegacy = LegacyMcaTheory.LegacyMcaTheory()
        _, fitResult1, result1 = self._configAndFit(
            x, y, copy.deepcopy(configuration), mcaFitLegacy, tmpflag=True)

        t1 = time.time()

        mcaFit = NewClassMcaTheory.McaTheory()
        _, fitResult2, result2 = self._configAndFit(
            x, y, copy.deepcopy(configuration), mcaFit, tmpflag=True)

        t2 = time.time()

        print("\nLEGACY TIME", t1-t0)
        print("NEW TIME", t2-t1)

        # Compare data
        numpy.testing.assert_array_equal(mcaFitLegacy.xdata.flat, mcaFit.xdata)
        numpy.testing.assert_array_equal(mcaFitLegacy.ydata.flat, mcaFit.ydata)
        numpy.testing.assert_array_equal(mcaFitLegacy.sigmay.flat, mcaFit.ystd)
        numpy.testing.assert_array_equal(mcaFitLegacy.zz.flat, mcaFit.ynumbkg())

        # Compare configuration
        config1 = copy.deepcopy(mcaFitLegacy.config)
        config2 = copy.deepcopy(mcaFit.config)

        # Remove expected differences
        n = len(config2["fit"]["energy"])
        for name in ["energy", "energyweight", "energyflag", "energyscatter"]:
            if config1["fit"][name] is None:
                config1["fit"][name] = []
            else:
                config1["fit"][name] = config1["fit"][name][:n]
        if len(config1["attenuators"]["Matrix"]) == 6:
            lst = config1["attenuators"]["Matrix"]
            lst.extend([0, lst[-1]+lst[-2]])
        config1["fit"]["continuum_name"] = config2["fit"]["continuum_name"]

        self._assertDeepEqual(config1, config2)

        # Compare fluo rate dictionaries
        self.assertEqual(mcaFitLegacy._fluoRates, mcaFit._fluoRates)

        # Compare line groups
        linegroups1 = mcaFitLegacy.PEAKS0
        linegroups2 = mcaFit._lineGroups
        linegroups1 = [[[line[1], line[0], name]
                        for name, line in zip(names, lines)] 
                        for names, lines in zip(mcaFitLegacy.PEAKS0NAMES, linegroups1)]

        self.assertEqual(len(linegroups1), len(linegroups2))
        for lg1, lg2 in zip(linegroups1, linegroups2):
            self.assertEqual(len(lg1), len(lg2))
            for line1, line2 in zip(lg1, lg2):
                self.assertEqual(line1[0], line2[0])
                numpy.testing.assert_allclose(line1[1], line2[1], rtol=1e-9)
                self.assertEqual(line1[2], line2[2])

        # Compare escape line groups
        linegroups1 = mcaFitLegacy.PEAKS0ESCAPE
        linegroups2 = mcaFit._escapeLineGroups
        self.assertEqual(len(linegroups1), len(linegroups2))
        for lg1, lg2 in zip(linegroups1, linegroups2):
            self.assertEqual(len(lg1), len(lg2))
            for elines1, elines2 in zip(lg1, lg2):
                self.assertEqual(len(elines1), len(elines2))
                for line1, line2 in zip(elines1, elines2):
                    self.assertEqual(line1[0], line2[0])
                    numpy.testing.assert_allclose(line1[1], line2[1], rtol=1e-9)
                    self.assertEqual(line1[2], line2[2])

        # Compare fit results
        self.assertEqual(fitResult1, fitResult2)
        self.assertEqual(result1, result2)

    def _configAndFit(self, x, y, configuration, mcaFit, tmpflag=False):
        configuration = mcaFit.configure(configuration)
        mcaFit.setData(x, y,
                       xmin=configuration["fit"]["xmin"],
                       xmax=configuration["fit"]["xmax"])

        mcaFit.estimate()

        if tmpflag:
            return configuration, None, None

        fitResult1, result1 = mcaFit.startFit(digest=1)
        return configuration, fitResult1, result1

    def _assertDeepEqual(self, obj1, obj2):
        """Better verbosity than assertEqual for deep structures
        """
        if isinstance(obj1, dict):
            self.assertEqual(set(obj1.keys()), set(obj2.keys()))
            for k in obj1:
                self._assertDeepEqual(obj1[k], obj2[k])
        elif isinstance(obj1, (list, tuple)):
            if isinstance(obj1[0], (list, tuple, numpy.ndarray)):
                self._assertDeepEqual(obj1, obj2)
            else:
                self.assertEqual(obj1, obj2)
        elif isinstance(obj1, numpy.ndarray):
            numpy.testing.assert_allclose(obj1, obj2, rtol=0)
        else:
            self.assertEqual(obj1, obj2)


def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(unittest.TestLoader().loadTestsFromTestCase(testXrf))
    else:
        # use a predefined order
        testSuite.addTest(testXrf("testTrainingDataDirectoryPresence"))
        testSuite.addTest(testXrf("testTrainingDataFilePresence"))
        testSuite.addTest(testXrf("testTrainingDataFit"))
        testSuite.addTest(testXrf("testStainlessSteelDataFit"))
    return testSuite

def test(auto=False):
    return unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        auto = False
    else:
        auto = True
    result = test(auto)
    sys.exit(not result.wasSuccessful())
