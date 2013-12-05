#/*##########################################################################
# Copyright (C) 2013 European Synchrotron Radiation Facility
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
import numpy
from PyMca.linalg import lstsq
from PyMca import ClassMcaTheory
from PyMca import ConfigDict

DEBUG = 0

class McaFastLinearFit(object):
    def __init__(self, mcafit=None):
        self._config = None
        if mcafit is None:
            self._mcaTheory = ClassMcaTheory.McaTheory()
        else:
            self._mcaTheory = mcafit

    def setFitConfiguration(self, configuration):
        self._mcaTheory.setConfiguration(configuration)

    def setFitConfigurationFile(self, ffile):
        configuration = ConfigDict.ConfigDict()
        configuration.read(ffile)
        self.setFitConfiguration(configuration)

    def fitMultipleSpectra(self, x=None, y=None, xmin=None, xmax=None,
                           configuration=None, concentrations=False):
        if y is None:
            raise RuntimeError("y keyword argument is mandatory!")

        if configuration is not None:
            self._mcaTheory.setConfiguration(configuration)

        # read the current configuration
        config = self._mcaTheory.getConfiguration()

        #make sure we force a linear fit
        config['fit']['linearfitflag'] = 1
        weight = config['fit']['fitweight']

        # background
        if config['fit']['stripflag']:
            if config['fit']['stripalgorithm'] == 1:
                if DEBUG:
                    print("SNIP")
            else:
                raise RuntimeError("Please use the faster SNIP background")

            #loop for anchors
            anchorslist = []
            if config['fit']['stripanchorsflag']:
                if config['fit']['stripanchorslist'] is not None:
                    ravelled = numpy.ravel(mcaTheory.xdata)
                    for channel in config['fit']['stripanchorslist']:
                        if channel <= ravelled[0]:continue
                        index = numpy.nonzero(ravelled >= channel)[0]
                        if len(index):
                            index = min(index)
                            if index > 0:
                                anchorslist.append(index)
            if len(anchorslist) == 0:
                anchorlist = [0, mcaTheory.ydata.size - 1]
            anchorslist.sort()

        # and now configure the fit
        self._mcaTheory.setConfiguration(config)

        if hasattr(y, "info") and hasattr(y, "data"):
            data = y.data
            mcaIndex = y.info.get("McaIndex", -1)
        else:
            data = y

        if len(data.shape) != 3:
            txt = "For the time being only three dimensional arrays supported"
            raise IndexError(txt)
        else:
            # if the cumulated spectrum is present it should be better
            nRows = data.shape[0]
            nColumns = data.shape[1]
            nPixels =  nRows * nColumns 
            firstSpectrum = data[0, 0, :]

        # make sure we calculate the matrix of the contributions
        self._mcaTheory.enableOptimizedLinearFit()

        # initialize the fit
        self._mcaTheory.setData(x=x, y=firstSpectrum, xmin=xmin, xmax=xmax)
        self._mcaTheory.estimate()
        
        # now we can get the derivatives respect to the free parameters
        # These are the "derivatives" respect to the peaks
        # linearMatrix = self._mcaTheory.linearMatrix

        # but we are still missing the derivatives from the background
        nFree = 0
        freeNames = []
        for i, param in enumerate(self._mcaTheory.PARAMETERS):
            if self._mcaTheory.codes[0][i] != ClassMcaTheory.Gefit.CFIXED:
                nFree += 1
                freeNames.append(param)

        #build the matrix of derivatives
        derivatives = None
        idx = 0
        for i, param in enumerate(self._mcaTheory.PARAMETERS):
            if self._mcaTheory.codes[0][i] == ClassMcaTheory.Gefit.CFIXED:
                continue
            deriv= self._mcaTheory.linearMcaTheoryDerivative(self._mcaTheory.parameters[i],
                                                             i,
                                                             self._mcaTheory.xdata)
            if derivatives is None:
                derivatives = numpy.zeros((deriv.shape[0], nFree), numpy.float)
            derivatives[:, idx] = deriv
            idx += 1


        #loop for anchors
        xdata = self._mcaTheory.xdata

        if config['fit']['stripflag']:
            anchorslist = []
            if config['fit']['stripanchorsflag']:
                if config['fit']['stripanchorslist'] is not None:
                    ravelled = numpy.ravel(xdata)
                    for channel in config['fit']['stripanchorslist']:
                        if channel <= ravelled[0]:continue
                        index = numpy.nonzero(ravelled >= channel)[0]
                        if len(index):
                            index = min(index)
                            if index > 0:
                                anchorslist.append(index)
            if len(anchorslist) == 0:
                anchorlist = [0, self._mcaTheory.ydata.size - 1]
            anchorslist.sort()
            
        # find the indices to be used for selecting the appropriate data
        # if the original x data were nor ordered we have a problem
        # TODO: check for original ordering.
        if x is None:
            # we have an enumerated channels axis
            iXMin = xdata[0]
            iXMax = xdata[-1]
        else:
            iXMin = numpy.nonzero(x <= xdata[0])[0][0]
            iXMax = numpy.nonzero(x >= xdata[-1])[0][0]

        dummySpectrum = firstSpectrum[iXMin:iXMax+1].reshape(-1, 1)

        # allocate the output buffer
        results = numpy.zeros((nFree, nRows, nColumns), numpy.float32)

        #perform the inital fit
        for i in range(data.shape[0]):
            #chunks of nColumns spectra
            if i == 0:
                chunk = numpy.zeros((dummySpectrum.shape[0], data.shape[1]),
                                    numpy.float)
                
            chunk[:,:] = data[i, :, iXMin:iXMax+1].T
            if config['fit']['stripflag']:
                for k in range(chunk.shape[0]):
                    # obtain the smoothed spectrum
                    background=SpecfitFuns.SavitskyGolay(chunk[k], 
                                            config['fit']['stripfilterwidth'])
                    lastAnchor = 0
                    for anchor in anchorslist:
                        if (anchor > lastAnchor) and (anchor < background.size):
                            background[lastAnchor:anchor] =\
                                    SpecfitFuns.snip1d(background[lastAnchor:anchor],
                                                       config['fit']['snipwidth'],
                                                       0)
                            lastAnchor = anchor
                    if lastAnchor < background.size:
                        background[lastAnchor:] =\
                                SpecfitFuns.snip1d(background[lastAnchor:],
                                                   config['fit']['snipwidth'],
                                                   0)
                    chunk[k] -= background

            # perform the multiple fit to all the spectra in the chunk
            ddict=lstsq(derivatives, chunk, weight=weight, digested_output=True)
            parameters = ddict['parameters'] 
            results[:, i, :] = parameters

        # cleanup zeros
        # start with the parameter with the largest amount of negative values
        negativePresent = True
        badParameters = []
        while negativePresent:
            zeroList = []
            for i in range(nFree):
                #we have to skip the background parameters
                t = results[i] < 0
                zeroList.append((t.sum(), i, t))
            zeroList.sort()
            zeroList.reverse()
            if zeroList[0][0] == 0:
                negativePresent = False
            else:
                badParameters.append(zeroList[0][1])
                badMask = zeroList[0][2]
                results[zeroList[0][1]][badMask] = 0.0
                A = derivatives[:, [i for i in range(nFree) if i not in badParameters]]
                #assume we'll not have too many spectra
                spectra = data[badMask, iXMin:iXMax+1]
                spectra.shape = badMask.sum(), -1
                spectra = spectra.T
                # 
                if config['fit']['stripflag']:
                    for k in range(spectra.shape[0]):
                        # obtain the smoothed spectrum
                        background=SpecfitFuns.SavitskyGolay(spectra[k], 
                                                config['fit']['stripfilterwidth'])
                        lastAnchor = 0
                        for anchor in anchorslist:
                            if (anchor > lastAnchor) and (anchor < background.size):
                                background[lastAnchor:anchor] =\
                                        SpecfitFuns.snip1d(background[lastAnchor:anchor],
                                                           config['fit']['snipwidth'],
                                                           0)
                                lastAnchor = anchor
                        if lastAnchor < background.size:
                            background[lastAnchor:] =\
                                    SpecfitFuns.snip1d(background[lastAnchor:],
                                                       config['fit']['snipwidth'],
                                                       0)
                    spectra[k] -= background
                ddict = lstsq(A, spectra, weight=weight, digested_output=True)
                idx = 0
                for i in range(nFree):
                    if i in badParameters:
                        continue
                    results[i][badMask] = ddict['parameters'][idx]
                    idx += 1

        if concentrations:
            raise NotImplemented("Fast concentrations calculation not implemented yet")

        return results
        
if __name__ == "__main__":
    import time
    import glob
    from PyMca import EDFStack
    if 1:
        configurationFile = "G4-4720eV-NOWEIGHT-Constant-batch.cfg"
        fileList = glob.glob("E:\DATA\PyMca-Training\G4-4720eV\G4_mca_0012*.edf")
    else:
        configurationFile = "E2_line.cfg"
        fileList = glob.glob("E:\DATA\PyMca-Data\FDM55\AS_EDF\E2_line*.edf")
    dataStack = EDFStack.EDFStack(filelist=fileList)

    t0 = time.time()
    fastFit = McaFastLinearFit()
    fastFit.setFitConfigurationFile(configurationFile)
    print("Configuring Elapsed = % s " % (time.time() - t0))
    results = fastFit.fitMultipleSpectra(y=dataStack)
    print("Total Elapsed = % s " % (time.time() - t0))

