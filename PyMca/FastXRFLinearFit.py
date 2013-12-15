#/*##########################################################################
# Copyright (C) 2013-2014 European Synchrotron Radiation Facility
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
__license__ = "LGPL"
import numpy
from PyMca.linalg import lstsq
from PyMca import ClassMcaTheory
from PyMca import Gefit
from PyMca import ConcentrationsTool
from PyMca import SpecfitFuns
from PyMca import ConfigDict

DEBUG = 0

class FastXRFLinearFit(object):
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
                           configuration=None, concentrations=False,
                           ysum=None):
        if y is None:
            raise RuntimeError("y keyword argument is mandatory!")

        #if concentrations:
        #    txt = "Fast concentration calculation not implemented yet"
        #    raise NotImplemented(txt)

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

        # and now configure the fit
        self._mcaTheory.setConfiguration(config)

        if hasattr(y, "info") and hasattr(y, "data"):
            data = y.data
            mcaIndex = y.info.get("McaIndex", -1)
        else:
            data = y
            mcaIndex = -1

        if len(data.shape) != 3:
            txt = "For the time being only three dimensional arrays supported"
            raise IndexError(txt)
        elif mcaIndex not in [-1, 2]:
            txt = "For the time being only mca arrays supported"
            raise IndexError(txt)
        else:
            # if the cumulated spectrum is present it should be better
            nRows = data.shape[0]
            nColumns = data.shape[1]
            nPixels =  nRows * nColumns
            if ysum is not None:
                firstSpectrum = ysum
            elif not concentrations:
                # just one spectrum is enough for the setup
                firstSpectrum = data[0, 0, :]
            else:
                firstSpectrum = data[0, :, :].sum(axis=0, dtype=numpy.float)

        # make sure we calculate the matrix of the contributions
        self._mcaTheory.enableOptimizedLinearFit()

        # initialize the fit
        # print("xmin = ", xmin)
        # print("xmax = ", xmax)
        # print("firstShape = ", firstSpectrum.shape)
        self._mcaTheory.setData(x=x, y=firstSpectrum, xmin=xmin, xmax=xmax)

        #loop for anchors
        if config['fit']['stripflag']:
            anchorslist = []
            if config['fit']['stripanchorsflag']:
                if config['fit']['stripanchorslist'] is not None:
                    ravelled = numpy.ravel(self._mcaTheory.xdata)
                    for channel in config['fit']['stripanchorslist']:
                        if channel <= ravelled[0]:
                            continue
                        index = numpy.nonzero(ravelled >= channel)[0]
                        if len(index):
                            index = min(index)
                            if index > 0:
                                anchorslist.append(index)
            if len(anchorslist) == 0:
                anchorlist = [0, self._mcaTheory.ydata.size - 1]
            anchorslist.sort()

        # and initialize the derivatives
        self._mcaTheory.estimate()
        
        # now we can get the derivatives respect to the free parameters
        # These are the "derivatives" respect to the peaks
        # linearMatrix = self._mcaTheory.linearMatrix

        # but we are still missing the derivatives from the background
        nFree = 0
        freeNames = []
        nFreeBackgroundParameters = 0
        for i, param in enumerate(self._mcaTheory.PARAMETERS):
            if self._mcaTheory.codes[0][i] != ClassMcaTheory.Gefit.CFIXED:
                nFree += 1
                freeNames.append(param)
                if i < self._mcaTheory.NGLOBAL:
                    nFreeBackgroundParameters += 1

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
        # print("dummy = ", dummySpectrum.shape)

        # allocate the output buffer
        results = numpy.zeros((nFree, nRows, nColumns), numpy.float32)
        uncertainties = numpy.zeros((nFree, nRows, nColumns), numpy.float32)

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
            # print("SHAPES")
            # print(derivatives.shape)
            # print(chunk.shape)
            ddict=lstsq(derivatives, chunk, weight=weight, digested_output=True)
            parameters = ddict['parameters'] 
            results[:, i, :] = parameters
            uncertainties[:, i, :] = ddict['uncertainties']

        # cleanup zeros
        # start with the parameter with the largest amount of negative values
        negativePresent = True
        nFits = 0
        while negativePresent:
            zeroList = []
            for i in range(nFree):
                #we have to skip the background parameters
                if i >= nFreeBackgroundParameters:
                    t = results[i] < 0
                    if t.sum() > 0:
                        zeroList.append((t.sum(), i, t))

            if len(zeroList) == 0:
                negativePresent = False
                continue

            if nFits > (2 * (nFree - nFreeBackgroundParameters)):
                # we are probably in an endless loop
                # force negative pixels
                for item in zeroList:
                    i = item[1]
                    badMask = item[2]
                    results[i][badMask] = 0.0
                    print("WARNING: %d pixels of parameter %s set to zero" % (item[0], freeNames[i]))
                continue
            zeroList.sort()
            zeroList.reverse()
            
            badParameters = []
            badParameters.append(zeroList[0][1])
            badMask = zeroList[0][2]
            if 1:
                # prevent and endless loop if two or more parameters have common pixels where they are
                # negative and one of them remains negative when forcing other one to zero
                for i, item in enumerate(zeroList):
                    if item[1] not in badParameters:
                        if item[0] > 0:
                            #check if they have common negative pixels
                            t = badMask * item[-1]
                            if t.sum() > 0:
                                badParameters.append(item[1])
                                badMask = t
            if badMask.sum() < (0.0025 * nPixels):
                # fit not worth
                for i in badParameters:
                    results[i][badMask] = 0.0
                    uncertainties[i][badMask] = 0.0
                    if DEBUG:
                        print("WARNING: %d pixels of parameter %s set to zero" % (badMask.sum(),
                                                                                  freeNames[i]))
            else:
                if DEBUG:
                    print("Number of secondary fits = %d" % (nFits + 1))
                nFits += 1
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
                        results[i][badMask] = 0.0
                        uncertainties[i][badMask] = 0.0
                    else:
                        results[i][badMask] = ddict['parameters'][idx]
                        uncertainties[i][badMask] = ddict['uncertainties'][idx]
                        idx += 1
        outputDict = {'parameters':results, 'uncertainties':uncertainties, 'names':freeNames}
        
        if concentrations:
            # check if an internal reference is used and if it is set to auto
            ####################################################
            # CONCENTRATIONS
            cTool = ConcentrationsTool.ConcentrationsTool()
            cToolConf = cTool.configure()
            cToolConf.update(config['concentrations'])

            fitFirstSpectrum = False
            if config['concentrations']['usematrix']:
                if DEBUG:
                    print("USING MATRIX")
                if config['concentrations']['reference'].upper() == "AUTO":
                    fitFirstSpectrum = True

            fitresult = {}
            if fitFirstSpectrum:
                # we have to fit the "reference" spectrum just to get the reference element
                mcafitresult = self._mcaTheory.startfit(digest=0, linear=True)
                # if one of the elements has zero area this cannot be made directly
                fitresult['result'] = self._mcaTheory.imagingDigestResult()
                fitresult['result']['config'] = config
                concentrationsResult, addInfo = cTool.processFitResult(config=cToolConf,
                                                    fitresult=fitresult,
                                                    elementsfrommatrix=False,
                                                    fluorates=self._mcaTheory._fluoRates,
                                                    addinfo=True)
                # and we have to make sure that all the areas are positive
                for group in fitresult['result']['groups']:
                    if fitresult['result'][group]['fitarea'] <= 0.0:
                        # give a tiny area
                        fitresult['result'][group]['fitarea'] = 1.0e-6
                config['concentrations']['reference'] = addInfo['ReferenceElement']
            else:
                fitresult['result'] = {}
                fitresult['result']['config'] = config
                fitresult['result']['groups'] = []
                idx = 0
                for i, param in enumerate(self._mcaTheory.PARAMETERS):
                    if self._mcaTheory.codes[0][i] == Gefit.CFIXED:
                        continue
                    if i < self._mcaTheory.NGLOBAL:
                        # background
                        pass
                    else:
                        fitresult['result']['groups'].append(param)
                        fitresult['result'][param] = {}
                        # we are just interested on the factor to be applied to the area to get the
                        # concentrations
                        fitresult['result'][param]['fitarea'] = 1.0
                        fitresult['result'][param]['sigmaarea'] = 1.0
                    idx += 1
            concentrationsResult, addInfo = cTool.processFitResult(config=cToolConf,
                                                    fitresult=fitresult,
                                                    elementsfrommatrix=False,
                                                    fluorates=self._mcaTheory._fluoRates,
                                                    addinfo=True)
            nValues = 1
            if len(concentrationsResult['layerlist']) > 1:
                nValues += len(concentrationsResult['layerlist'])
            massFractions = numpy.zeros((nValues * (nFree - nFreeBackgroundParameters), nRows, nColumns),
                                        numpy.float32)


            referenceElement = addInfo['ReferenceElement'] 
            referenceTransitions = addInfo['ReferenceTransitions']
            if DEBUG:
                print("Reference <%s>  transition <%s>" % (referenceElement, referenceTransitions))
            if referenceElement in ["", None, "None"]:
                if DEBUG:
                    print("No reference")
                counter = 0
                for i, group in enumerate(fitresult['result']['groups']):
                    outputDict['names'].append("C(%s)" % group)
                    massFractions[counter] = results[nFreeBackgroundParameters+i] *\
                        (concentrationsResult['mass fraction'][group]/fitresult['result'][param]['fitarea'])
                    if len(concentrationsResult['layerlist']) > 1:
                        for layer in concentrationsResult['layerlist']:
                            outputDict['names'].append("C(%s)-%s" % (group, layer))
                            massFractions[counter] = results[nFreeBackgroundParameters+i] *\
                        (concentrationsResult[layer]['mass fraction'][group]/fitresult['result'][param]['fitarea'])
                            counter += 1
                    counter += 1
            else:
                if DEBUG:
                    print("With reference")
                idx = None
                testGroup = referenceElement+ " " + referenceTransitions.split()[0]
                for i, group in enumerate(fitresult['result']['groups']):
                    if group == testGroup:
                        idx = i
                if idx is None:
                    raise ValueError("Invalid reference:  <%s> <%s>" %\
                                     (referenceElement, referenceTransitions))

                group = fitresult['result']['groups'][idx]
                referenceArea = fitresult['result'][group]['fitarea']
                referenceConcentrations = concentrationsResult['mass fraction'][group]
                goodIdx = results[nFreeBackgroundParameters+idx] > 0
                massFractions[idx] = referenceConcentrations
                counter = 0
                for i, group in enumerate(fitresult['result']['groups']):
                    outputDict['names'].append("C(%s)" % group)
                    if i == idx:
                        continue
                    goodI = results[nFreeBackgroundParameters+i] > 0
                    tmp = results[nFreeBackgroundParameters+idx][goodI]
                    massFractions[i][goodI] = (results[nFreeBackgroundParameters+i][goodI]/(tmp + (tmp == 0))) *\
                                ((referenceArea/fitresult['result'][group]['fitarea']) *\
                                (concentrationsResult['mass fraction'][group]))
                    if len(concentrationsResult['layerlist']) > 1:
                        for layer in concentrationsResult['layerlist']:
                            outputDict['names'].append("C(%s)-%s" % (group, layer))
                            massFractions[i][goodI] = (results[nFreeBackgroundParameters+i][goodI]/(tmp + (tmp == 0))) *\
                                ((referenceArea/fitresult['result'][group]['fitarea']) *\
                                (concentrationsResult[layer]['mass fraction'][group]))
                            counter += 1                    
                    counter += 1
            outputDict['concentrations'] = massFractions
            ####################################################
        return outputDict
        
if __name__ == "__main__":
    import time
    import glob
    from PyMca.PyMcaIO import EDFStack
    if 1:
        configurationFile = "E:\DATA\COTTE\CH1777\G4-4720eV-NOWEIGHT-NO_Constant-batch.cfg"
        fileList = glob.glob("E:\DATA\COTTE\CH1777\G4_mca_0012_0000_0*.edf")
    else:
        configurationFile = "E2_line.cfg"
        fileList = glob.glob("E:\DATA\PyMca-Training\FDM55\AS_EDF\E2_line*.edf")
    dataStack = EDFStack.EDFStack(filelist=fileList)

    t0 = time.time()
    fastFit = FastXRFLinearFit()
    fastFit.setFitConfigurationFile(configurationFile)
    print("Configuring Elapsed = % s " % (time.time() - t0))
    results = fastFit.fitMultipleSpectra(y=dataStack, concentrations=True)
    print("Total Elapsed = % s " % (time.time() - t0))

