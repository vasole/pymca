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
import os
import numpy
try:
    from PyMca import ConfigDict
    from PyMca import SimpleFitModule
    from PyMca import ArraySave
    from PyMca import PyMcaDirs
except ImportError:
    print("StackSimpleFit is importing from somewhere else")
    import ConfigDict
    import SimpleFitModule
    import ArraySave
    import PyMcaDirs
    

DEBUG = 0

class StackSimpleFit(object):
    def __init__(self, fit=None):
        if fit is None:
            fit = SimpleFitModule.SimpleFit()
        self.fit = fit    
        self.stack_y = None
        #self.configuration = None
        self.outputDir = PyMcaDirs.outputDir
        self.outputFile = None
        self.fixedLenghtOutput = True
        self.progressCallback = None
        self.dataIndex = None

    def setProgressCallback(self, method):
        """
        The method will be called as method(current_fit_index, total_fit_index)
        """
        self.progressCallback = method

    def setOutputDirectory(self, outputdir):
        self.outputDir = outputdir

    def setOutputFileBaseName(self, outputfile):
        self.outputFile = outputfile

    def setData(self, stack_x, stack_y, sigma=None, xmin=None, xmax=None):
        self.stack_x = stack_x
        self.stack_y = stack_y
        self.stack_sigma = sigma
        self.xMin = xmin
        self.xMax = xmax

    def setDataIndex(self, data_index=None):
        self.data_index = data_index

    def setConfigurationFile(self, fname):
        if not os.path.exists(fname):
            raise IOError("File %s does not exist" % fname)
        w = ConfigDict.ConfigDict()
        w.read(fname)
        self.configuration = w
        self.setConfiguration(w)        

    def setConfiguration(self, ddict):
        self.configuration = ddict
        self.fit.setConfiguration(ddict, try_import=True)
        
    def processStack(self):
        data_index = self.dataIndex
        if data_index == None:
            data_index = -1
        if type(data_index) == type(1):
            data_index = [data_index]

        if len(data_index) > 1:
            raise IndexError("Only 1D fitting implemented for the time being")
            
        #this leaves the possibility to fit images by giving
        #two indices specifying the image dimensions
        self.stackDataIndexList = data_index
        
        stack = self.stack_y 
        if stack is None:
            raise ValueError("No data to be processed")

        if hasattr(stack, "info") and hasattr(stack, "data"):
            data = stack.data
        else:
            data = stack

        #make sure all the indices are positive
        for i in range(len(data_index)):
            if data_index[i] < 0:
                data_index[i] = range(len(data.shape))[data_index[i]]

        #get the total number of fits to be performed
        outputDimension = []
        nPixels = 1
        for i in range(len(data.shape)):
            if not (i in data_index):
                nPixels *= data.shape[i]
                outputDimension.append(data.shape[i])

        lenOutput = len(outputDimension) 
        if lenOutput > 2:
            raise ValueError("Rank of  output greater than 2")
        elif lenOutput == 2:
            self._nRows = outputDimension[0]
            self._nColumns = outputDimension[1]
        else:    
            self._nRows = outputDimension[0]
            self._nColumns = 1

        #self.fit.setConfiguration(self.configuration, try_import=True)
        self._parameters = None

        self._row = 0
        self._column = -1
        for i in range(nPixels):
            if (self._column+1) == self._nColumns:
                self._column = 0
                self._row   += 1
            else:
                self._column += 1
            try:
                self.processStackData(i)
            except:
                print("Error processing index = %d, row = %d column = %d" %\
                          (i, self._row, self._column))
                if DEBUG:
                    raise
        self.onProcessStackFinished()
        if self.progressCallback is not None:
            self.progressCallback(nPixels, nPixels)

    def processStackData(self, i):
        self.aboutToGetStackData(i)
        x, y, sigma, xmin, xmax = self.getFitInputValues(i)
        self.fit.setData(x, y, sigma=sigma, xmin=xmin, xmax=xmax)
        self.fit.estimate()
        self.estimateFinished()
        values, chisq, sigma, niter, lastdeltachi = self.fit.startFit()
        self.fitFinished()

    def getFitInputValues(self, index):
        """
        Returns the fit parameters x, y, sigma, xmin, xmax
        """


        row    = self._row
        column = self._column
        data_index = self.stackDataIndexList[0]

        #get y
        yShape = self.stack_y.shape
        if len(yShape) == 3:
            if data_index == 0:
                y = self.stack_y[:, row, column]
            elif data_index == 1:    
                y = self.stack_y[row, :, column]
            else:
                y = self.stack_y[row, column]
        elif len(yShape) == 2:
            if column > 0:
                raise ValueError("Column index > 0 on a single column stack")
            y = self.stack_y[row]
        else:
            raise TypeError("Unsupported y data shape lenght")

        #get x
        if self.stack_x is None:
            nValues = y.size
            x = numpy.arange(float(nValues))
            x.shape = y.shape
            self.stack_x = x

        xShape = self.stack_x.shape
        xSize  = self.stack_x.size
        sigma = None
        if xShape == yShape:
            #as many x as y, follow same criterium
            if len(xShape) == 3:
                if data_index == 0:
                    x = self.stack_x[:, row, column]
                elif data_index == 1:    
                    x = self.stack_x[row, :, column]
                else:
                    x = self.stack_x[row, column]
            elif len(xShape) == 2:
                if column > 0:
                    raise ValueError("Column index > 0 on a single column stack")
                x = self.stack_x[row]
            else:
                raise TypeError("Unsupported x data shape lenght")
        elif xSize == y.size:
            #only one x for all the y values
            x = numpy.zeros(y.size, numpy.float)
            x[:] = self.stack_x[:]
            x.shape = y.shape
        else:
            raise ValueError("Cannot handle incompatible X and Y values")

        if self.stack_sigma is None:
            return x, y, sigma, self.xMin, self.xMax

        # get sigma
        sigmaShape = self.stack_sigma.shape
        sigmaSize  = self.stack_sigma.size

        if sigmaShape == yShape:
            #as many sigma as y, follow same criterium
            if len(sigmaShape) == 3:
                if data_index == 0:
                    sigma = self.stack_sigma[:, row, column]
                elif data_index == 1:    
                    sigma = self.stack_sigma[row, :, column]
                else:
                    sigma = self.stack_sigma[row, column]
            elif len(sigmaShape) == 2:
                if column > 0:
                    raise ValueError("Column index > 0 on a single column stack")
                sigma = self.stack_sigma[row]
            else:
                raise TypeError("Unsupported sigma data shape lenght")
        elif sigmaSize == y.size:
            #only one sigma for all the y values
            sigma = numpy.zeros(y.size, numpy.float)
            sgima[:] = self.stack_sigma[:]
            sigma.shape = y.shape
        else:
            raise ValueError("Cannot handle incompatible sigma and y values")

        return x, y, sigma, self.xMin, self.xMax

    def estimateFinished(self):
        if DEBUG:
            print("Estimate finished")

    def aboutToGetStackData(self, idx):
        if DEBUG:
            print("New spectrum %d" % idx)
        self._currentFitIndex = idx
        if self.progressCallback is not None:
            self.progressCallback(idx, self._nRows * self._nColumns)

        if idx == 0:
            specfile = os.path.join(self.outputDir,
                                    self.outputFile+".spec")
            if os.path.exists(self.outputFile):
                os.remove(self.outputFile)

    def fitFinished(self):
        if DEBUG:
            print("fit finished")

        #get parameter results
        fitOutput = self.fit.getResult(configuration=False)
        result = fitOutput['result']
        row= self._row
        column = self._column
        if result is None:
            print("result not valid for row %d, column %d" % (row, column))
            return

        if self.fixedLenghtOutput and (self._parameters is None):
            #If it is the first fit, initialize results array
            imgdir = os.path.join(self.outputDir, "IMAGES")
            if not os.path.exists(imgdir):
                os.mkdir(imgdir)
            if not os.path.isdir(imgdir):
                msg= "%s does not seem to be a valid directory" % imgdir
                raise IOError(msg)
            self.imgDir = imgdir
            self._parameters  = []
            self._images      = {}
            self._sigmas      = {}
            for parameter in result['parameters']:
                self._parameters.append(parameter)
                self._images[parameter] = numpy.zeros((self._nRows,
                                                       self._nColumns),
                                                       numpy.float32)
                self._sigmas[parameter] = numpy.zeros((self._nRows,
                                                       self._nColumns),
                                                       numpy.float32)
            self._images['chisq'] = numpy.zeros((self._nRows,
                                                       self._nColumns),
                                                       numpy.float32)

        if self.fixedLenghtOutput:
            i = 0
            for parameter in self._parameters:
                self._images[parameter] [row, column] =\
                                        result['fittedvalues'][i]
                self._sigmas[parameter] [row, column] =\
                                        result['sigma_values'][i]
                i += 1
            self._images['chisq'][row, column] = result['chisq']

        #specfile output always available
        specfile = self.getOutputFileNames()['specfile']
        
        self._appendOneResultToSpecfile(specfile, result=fitOutput)

    def _appendOneResultToSpecfile(self, filename, result=None):
        if result is None:
            result = self.fit.getResult(configuration=False)

        scanNumber = self._currentFitIndex

        #open file in append mode
        fitResult = result['result']
        fittedValues = fitResult['fittedvalues']
        fittedParameters = fitResult['parameters']
        chisq = fitResult['chisq']
        text = "\n#S %d %s\n" % (scanNumber, "PyMca Stack Simple Fit")
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

    def getOutputFileNames(self):
        specfile = os.path.join(self.outputDir,
                                self.outputFile+".spec")
        imgDir = os.path.join(self.outputDir, "IMAGES")
        filename = os.path.join(imgDir, self.outputFile)
        csv = filename + ".csv"
        edf = filename + ".edf"
        ddict = {}
        ddict['specfile'] = specfile
        ddict['csv'] = csv
        ddict['edf'] = edf
        return ddict
        
    def onProcessStackFinished(self):
        if DEBUG:
            print("Stack proccessed")
        if self.fixedLenghtOutput:
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
            filenames = self.getOutputFileNames()
            csvName = filenames['csv']
            edfName = filenames['edf']
            ArraySave.save2DArrayListAsASCII(datalist,
                                             csvName,
                                             labels=labels,
                                             csv=True,
                                             csvseparator=";")
            ArraySave.save2DArrayListAsEDF(datalist,
                                           edfName,
                                           labels = labels,
                                           dtype=numpy.float32)


def test():
    import numpy
    from PyMca import SpecfitFuns
    x = numpy.arange(1000.)
    data = numpy.zeros((50, 1000), numpy.float)

    #the peaks to be fitted
    p0 = [100., 300., 50.,
          200., 500., 30.,
          300., 800., 65]

    #generate the data to be fitted
    for i in range(data.shape[0]):
        nPeaks = 3 - i % 3
        data[i,:] = SpecfitFuns.gauss(p0[:3*nPeaks],x)

    oldShape = data.shape
    data.shape = 1,oldShape[0], oldShape[1]

    instance = StackSimpleFit()
    instance.setData(x, data)
    instance.setConfigurationFile("C:\StackSimpleFit.cfg")
    instance.processStack()

if __name__=="__main__":
    DEBUG = 0
    test()
