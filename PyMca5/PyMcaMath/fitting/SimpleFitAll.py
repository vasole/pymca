#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
import sys
import os
import numpy
from PyMca5.PyMcaIO import ConfigDict
from PyMca5.PyMcaIO import ArraySave
from PyMca5 import PyMcaDirs

DEBUG = False


class SimpleFitAll(object):
    def __init__(self, fit):
        self.fit = fit
        self.curves_y = None
        self.outputDir = PyMcaDirs.outputDir
        self.outputFile = None
        self.fixedLenghtOutput = True
        self._progress = 0.0
        self._status = "Ready"
        self._currentFitIndex = None
        self._nSpectra = None
        self.progressCallback = None
        # optimization variables
        self.__ALWAYS_ESTIMATE = True

    def setProgressCallback(self, method):
        """
        The method will be called as method(current_fit_index, total_fit_index)
        """
        self.progressCallback = method

    def progressUpdate(self):
        """
        This method returns a dictionnary with the keys
        progress: A number between 0 and 100 indicating the fit progress
        status: Status of the calculation thread.
        """
        ddict = {
            'progress': self._progress,
            'status': self._status}
        return ddict

    def setOutputDirectory(self, outputdir):
        self.outputDir = outputdir

    def setOutputFileName(self, outputfile):
        self.outputFile = outputfile

    def setData(self, curves_x, curves_y, sigma=None, xmin=None, xmax=None):
        """

        :param curves_x: List of 1D arrays, one per curve, or single 1D array
        :param curves_y: List of 1D arrays, one per curve
        :param sigma: List of 1D arrays, one per curve, or single 1D array
        :param float xmin:
        :param float xmax:
        :return:
        """
        self.curves_x = curves_x
        self.curves_y = curves_y
        self.curves_sigma = sigma
        self.xMin = xmin
        self.xMax = xmax

    # TODO
    def setConfigurationFile(self, fname):
        if not os.path.exists(fname):
            raise IOError("File %s does not exist" % fname)
        w = ConfigDict.ConfigDict()
        w.read(fname)
        self.setConfiguration(w)

    def setConfiguration(self, ddict):
        self.fit.setConfiguration(ddict, try_import=True)

    def processAll(self):
        data = self.curves_y

        # get the total number of fits to be performed
        self._nSpectra = len(data)

        # optimization
        self.__ALWAYS_ESTIMATE = True
        backgroundPolicy = self.fit._fitConfiguration['fit']['background_estimation_policy']
        functionPolicy = self.fit._fitConfiguration['fit']['function_estimation_policy']
        if "Estimate always" not in [functionPolicy, backgroundPolicy]:
            self.__ALWAYS_ESTIMATE = False

        # initialize control variables
        self._parameters = None
        self._progress = 0
        self._status = "Fitting"
        for i in range(self._nSpectra):
            self._progress = (i * 100.) / self._nSpectra
            try:
                self.processSpectrum(i)
            except:
                print("Error %s processing index = %d" %
                      (sys.exc_info()[1], i))
                if DEBUG:
                    raise
        self.onProcessSpectraFinished()
        self._status = "Ready"
        if self.progressCallback is not None:
            self.progressCallback(self._nSpectra, self._nSpectra)

    def processSpectrum(self, i):
        self.aboutToGetSpectrum(i)
        x, y, sigma, xmin, xmax = self.getFitInputValues(i)
        self.fit.setData(x, y, sigma=sigma, xmin=xmin, xmax=xmax)
        if self._parameters is None:
            if DEBUG:
                print("First estimation")
            self.fit.estimate()
        elif self.__ALWAYS_ESTIMATE:
            if DEBUG:
                print("Estimation due to settings")
            self.fit.estimate()
        self.estimateFinished()
        values, chisq, sigma, niter, lastdeltachi = self.fit.startFit()
        self.fitOneSpectrumFinished()

    def getFitInputValues(self, index):
        """
        Returns the fit parameters x, y, sigma, xmin, xmax
        """
        # get y (always a list of 1D arrays)
        y = self.curves_y[index]

        # get x
        if self.curves_x is None:
            nValues = y.size
            x = numpy.arange(float(nValues))
            x.shape = y.shape
            self.curves_x = x
        elif hasattr(self.curves_x, "ndim") and self.curves_x.ndim == 1:
            # same x array for all curves
            x = self.curves_x
        else:
            # list of abscissas, one per curve
            x = self.curves_x[index]

        assert x.shape == y.shape

        if self.curves_sigma is None:
            return x, y, None, self.xMin, self.xMax

        # get sigma
        if hasattr(self.curves_sigma, "ndim") and self.curves_sigma.ndim == 1:
            # only one sigma for all the y values
            sigma = self.curves_sigma
        else:
            sigma = self.curves_sigma[index]
        assert sigma.shape == y.shape

        return x, y, sigma, self.xMin, self.xMax

    def estimateFinished(self):
        if DEBUG:
            print("Estimate finished")

    def aboutToGetSpectrum(self, idx):
        if DEBUG:
            print("New spectrum %d" % idx)
        self._currentFitIndex = idx
        if self.progressCallback is not None:
            self.progressCallback(idx, self._nSpectra)

    def fitOneSpectrumFinished(self):
        if DEBUG:
            print("fit finished")

        # get parameter results
        fitOutput = self.fit.getResult(configuration=False)
        result = fitOutput['result']
        idx = self._currentFitIndex
        if result is None:
            print("result not valid for index %d" % idx)
            return

        self.getOutputFileName()
        self._appendOneResultToHdf5(self.getOutputFileName(),
                                    result=fitOutput)

    def _appendOneResultToHdf5(self, filename, result):   # TODO
        print(result)   # PK debugging
        return
        scanNumber = self._currentFitIndex

        #open file in append mode
        fitResult = result['result']
        fittedValues = fitResult['fittedvalues']
        fittedParameters = fitResult['parameters']
        chisq = fitResult['chisq']
        text = "\n#S %d %s\n" % (scanNumber, "PyMca Stack Simple F"
                                             "it")
        text += "#N %d\n" % (len(fittedParameters)+2)
        text += "#L N  Chisq"
        for parName in fittedParameters:
            text += '  %s' % parName
        text += "\n"
        text += "1 %f" % chisq
        for parValue in fittedValues:
            text += "% .7E" % parValue
        text += "\n"
        sf = open(filename, 'ab')
        sf.write(text)
        sf.close()

    def getOutputFileName(self):
        return os.path.join(self.outputDir,
                            self.outputFile)

    def onProcessSpectraFinished(self):
        if DEBUG:
            print("All curves processed")
        self._status = "Curves Fitting finished"
        if self.fixedLenghtOutput:
            self._status = "Writing output file"
            nParameters = len(self._parameters)
            datalist = [None] * (2*len(self._sigmas.keys())+1)
            labels = []
            for i in range(nParameters):
                parameter = self._parameters[i]
                datalist[2*i] = self._images[parameter]
                datalist[2*i + 1] = self._sigmas[parameter]
                labels.append(parameter)
                labels.append('s(%s)' % parameter)
            datalist[-1] = self._images['chisq']
            labels.append('chisq')

            # filenames = self.getOutputFileNames()
            # csvName = filenames['csv']
            # edfName = filenames['edf']
            # ArraySave.save2DArrayListAsASCII(datalist,
            #                                  csvName,
            #                                  labels=labels,
            #                                  csv=True,
            #                                  csvseparator=";")
            # ArraySave.save2DArrayListAsEDF(datalist,
            #                                edfName,
            #                                labels = labels,
            #                                dtype=numpy.float32)
