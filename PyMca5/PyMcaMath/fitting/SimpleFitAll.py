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
import h5py
import datetime

from PyMca5.PyMcaIO import ConfigDict
import PyMca5

DEBUG = False

if sys.version_info < (3, ):
    text_dtype = h5py.special_dtype(vlen=unicode)
else:
    text_dtype = h5py.special_dtype(vlen=str)


def to_h5py_utf8(str_list):
    """Convert a string or a list of strings to a numpy array of
    unicode strings that can be written to HDF5 as utf-8.
    """
    return numpy.array(str_list, dtype=text_dtype)


class SimpleFitAll(object):
    """Fit module designed to fit a number of curves, and save its
    output to HDF5 - nexus."""
    def __init__(self, fit):
        self.fit = fit
        self.curves_x = None
        self.curves_y = None
        self.curves_sigma = None
        self.legends = None
        self.xMin = None
        self.xMax = None
        self.outputDir = PyMca5.PyMcaDirs.outputDir
        self.outputFileName = None
        self.outputFile = None
        self._progress = 0.0
        self._status = "Ready"
        self._currentFitIndex = None
        self._nSpectra = None
        self.progressCallback = None
        # optimization variables
        self.__ALWAYS_ESTIMATE = True
        self._startTime = ""
        self._endTime = ""

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
        self.outputFileName = outputfile

    def setData(self, curves_x, curves_y, sigma=None, xmin=None, xmax=None,
                legends=None):
        """

        :param curves_x: List of 1D arrays, one per curve, or single 1D array
        :param curves_y: List of 1D arrays, one per curve
        :param sigma: List of 1D arrays, one per curve, or single 1D array
        :param float xmin:
        :param float xmax:
        :param List[str] legends: List of curve legends. If None, defaults to
            ``["curve0", "curve1"...]``
        """
        self.curves_x = curves_x
        self.curves_y = curves_y
        self.curves_sigma = sigma
        self.xMin = xmin
        self.xMax = xmax
        self.legends = legends or ["curve%d" % i for i in range(len(curves_y))]

    def setConfigurationFile(self, fname):
        if not os.path.exists(fname):
            raise IOError("File %s does not exist" % fname)
        w = ConfigDict.ConfigDict()
        w.read(fname)
        self.setConfiguration(w)

    def setConfiguration(self, ddict):
        self.fit.setConfiguration(ddict, try_import=True)

    def processAll(self):
        assert self.curves_y is not None, "You must first call setData()!"
        data = self.curves_y

        assert self.outputFile is None
        # self.outputFile = h5py.File(self.getOutputFileName(), mode="w-")
        # self.outputFile.attrs["NX_class"] = "NXroot"

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
        # self.outputFile.close()
        # self.outputFile = None
        self._status = "Ready"
        if self.progressCallback is not None:
            self.progressCallback(self._nSpectra, self._nSpectra)

    def processSpectrum(self, i):
        self._startTime = datetime.datetime.now().isoformat()
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
        self._endTime = datetime.datetime.now().isoformat()
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
        elif hasattr(self.curves_x, "shape") and len(self.curves_x.shape) == 1:
            # same x array for all curves
            x = self.curves_x
        else:
            # list of abscissas, one per curve
            x = self.curves_x[index]

        assert x.shape == y.shape

        if self.curves_sigma is None:
            return x, y, None, self.xMin, self.xMax

        # get sigma
        if hasattr(self.curves_sigma, "shape") and len(self.curves_sigma.shape) == 1:
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

        self._appendOneResultToHdf5(result=fitOutput["result"])

    def _appendOneResultToHdf5(self, result):

        idx = self._currentFitIndex        # TODO
        print(idx, "\n", result)

        configFitFunction = result["fit_function"]
        configBackgroundFunction = result["background_function"],
        configIni = ConfigDict.ConfigDict(self.fit.getConfiguration()).tostring()

        entry = self.outputFile.create_group("fit_curve_%d" % idx)
        entry.attrs["NX_class"] = to_h5py_utf8("NXentry")
        entry.attrs["title"] = to_h5py_utf8("Fit of curve '%s'" % self.legends[idx])
        entry.attrs["default"] = to_h5py_utf8("fit_process/results/plot")
        entry.create_dataset("start_time", data=to_h5py_utf8(self._startTime))
        entry.create_dataset("end_time", data=to_h5py_utf8(self._endTime))
        entry.create_dataset("curve_legend", data=to_h5py_utf8(self.legends[idx]))

        process = entry.create_group("fit_process")
        process.create_dataset("program", data=to_h5py_utf8("pymca"))
        process.create_dataset("version", data=to_h5py_utf8(PyMca5.version()))
        process.create_dataset("date", data=to_h5py_utf8(self._endTime))

        configuration = process.create_group("configuration")
        configuration.attrs["NX_class"] = to_h5py_utf8("NXnote")
        # not sure if these attrs are needed, they are repeated in the INI
        # configuration.create_dataset("function", configFitFunction)
        # configuration.create_dataset("background_function", configBackgroundFunction)
        # ...
        configuration.create_dataset("type", data=to_h5py_utf8("text/ini"))
        configuration.create_dataset("data", data=to_h5py_utf8(configIni))

        results = entry.create_group("results")
        for key, value in result.items():
            if not numpy.issubdtype(type(key), numpy.character):
                print("skipping key %s (not a text string)" % key)
                continue
            value_dtype = numpy.array(value).dtype
            if numpy.issubdtype(value_dtype, numpy.number) or\
                    numpy.issubdtype(value_dtype, numpy.bool_):
                # straightforward conversion to HDF5
                results.create_dataset(key, value)
            elif numpy.issubdtype(value_dtype, numpy.character):
                results.create_dataset(key, to_h5py_utf8(value))



        results.create_dataset("chisq", data=result["chisq"])
        results.create_dataset("niter", data=result["niter"])
        results.create_dataset("lastdeltachi", data=result["lastdeltachi"])
        results.create_dataset("parameters", data=to_h5py_utf8(result["parameters"]))
        results.create_dataset("fittedvalues", data=to_h5py_utf8(result["fittedvalues"]))
        ...

        return

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
                            self.outputFileName)

    def onProcessSpectraFinished(self):
        if DEBUG:
            print("All curves processed")
        self._status = "Curves Fitting finished"

        # if self.fixedLenghtOutput:
        #     self._status = "Writing output file"
        #     nParameters = len(self._parameters)
        #     datalist = [None] * (2*len(self._sigmas.keys())+1)
        #     labels = []
        #     for i in range(nParameters):
        #         parameter = self._parameters[i]
        #         datalist[2*i] = self._images[parameter]
        #         datalist[2*i + 1] = self._sigmas[parameter]
        #         labels.append(parameter)
        #         labels.append('s(%s)' % parameter)
        #     datalist[-1] = self._images['chisq']
        #     labels.append('chisq')

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
