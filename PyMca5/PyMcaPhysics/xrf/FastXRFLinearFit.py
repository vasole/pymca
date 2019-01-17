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
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
Module to perform a fast linear fit on a stack of fluorescence spectra.
"""
import os
import numpy
import logging
from PyMca5.PyMcaMath.linalg import lstsq
from . import ClassMcaTheory
from PyMca5.PyMcaMath.fitting import Gefit
from . import ConcentrationsTool
from PyMca5.PyMcaMath.fitting import SpecfitFuns
from PyMca5.PyMcaIO import ConfigDict
from PyMca5.PyMcaIO import NexusUtils
import time

_logger = logging.getLogger(__name__)


class FastXRFLinearFit(object):
    def __init__(self, mcafit=None):
        self._config = None
        if mcafit is None:
            self._mcaTheory = ClassMcaTheory.McaTheory()
        else:
            self._mcaTheory = mcafit

    def setFitConfiguration(self, configuration):
        self._mcaTheory.setConfiguration(configuration)
        self._config = self._mcaTheory.getConfiguration()

    def setFitConfigurationFile(self, ffile):
        if not os.path.exists(ffile):
            raise IOError("File <%s> does not exists" % ffile)
        configuration = ConfigDict.ConfigDict()
        configuration.read(ffile)
        self.setFitConfiguration(configuration)

    def fitMultipleSpectra(self, x=None, y=None, xmin=None, xmax=None,
                           configuration=None, concentrations=False,
                           ysum=None, weight=None, refit=True, livetime=None,
                           saveresiduals=False, savefit=False, savedata=False):
        """
        This method performs the actual fit. The y keyword is the only mandatory input argument.

        :param x: 1D array containing the x axis (usually the channels) of the spectra.
        :param y: 3D array containing the spectra as [nrows, ncolumns, nchannels]
        :param xmin: lower limit of the fitting region
        :param xmax: upper limit of the fitting region
        :param ysum: sum spectrum
        :param weight: 0 Means no weight, 1 Use an average weight, 2 Individual weights (slow)
        :param concentrations: 0 Means no calculation, 1 Calculate elemental concentrations
        :param saveresiduals: 0 Means no calculation, 1 Calculate fit residuals
        :param savefit: 0 Means no calculation, 1 Calculate fitted spectra
        :param savedata: 0 Mean not| 1 Mean yes
        :param refit: if False, no check for negative results. Default is True.
        :livetime: It will be used if not different from None and concentrations
                   are to be calculated by using fundamental parameters with
                   automatic time. The default is None.
        :return: A dictionnary with the parameters, uncertainties, concentrations and names as keys.
        """
        # Parse data
        x, data, mcaIndex, livetime = self._fit_parse_data(x=x, y=y,
                                                           livetime=livetime)

        # Check data dimensions
        nSpectra = data.size // data.shape[mcaIndex]
        if data.ndim != 3:
            txt = "For the time being only three dimensional arrays supported"
            raise IndexError(txt)
        elif mcaIndex not in [-1, 2]:
            txt = "For the time being only mca arrays supported"
            raise IndexError(txt)

        t0 = time.time()

        # Configure fit
        configorg, config, weight, weightPolicy, \
        autotime, liveTimeFactor = self._fit_configure(
                                            configuration=configuration,
                                            concentrations=concentrations,
                                            livetime=livetime,
                                            weight=weight,
                                            nSpectra=nSpectra)

        # Sum spectrum
        if ysum is None:
            if weightPolicy == 1:
                # we need to calculate the sum spectrum
                # to derive the uncertainties
                sumover = 'all'
            elif not concentrations:
                # one spectrum is enough
                sumover = 'first pixel'
            else:
                sumover = 'first row'
            ysum = self._fit_mcasum_3d(data=data, mcaIndex=mcaIndex, sumover=sumover)

        # Get the basis of the linear models (i.e. derivative to peak areas)
        if xmin is None:
            xmin = config['fit']['xmin']
        if xmax is None:
            xmax = config['fit']['xmax']
        self._mcaTheory.setData(x=x, y=ysum, xmin=xmin, xmax=xmax)
        derivatives, freeNames, nFree, nFreeBkg = self._fit_model()

        # Background anchor points (if any)
        anchorslist = self._fit_bkg_anchorlist(config=config)

        # MCA trimming: [iXMin:iXMax]
        iXMin, iXMax = self._fit_mcatrim_info(x=x)
        idxChan = slice(iXMin, iXMax)
        nChan = iXMax-iXMin

        # Least-squares parameters
        if weightPolicy == 2:
            # Individual spectrum weights (assumed Poisson)
            SVD = False
            sigma_b = None
        elif weightPolicy == 1:
            # Average weight from sum spectrum (assume Poisson)
            # the +1 is to prevent misbehavior due to weights less than 1.0
            sigma_b = 1 + numpy.sqrt(ysum[idxChan])/nSpectra
            sigma_b = sigma_b.reshape(-1, 1)
            SVD = True
        else:
            # No weights
            SVD = True
            sigma_b = None
        lstsq_kwargs = {'svd': SVD, 'sigma_b': sigma_b, 'weight': weight}

        # Allocate output buffers
        nRows = data.shape[0]
        nColumns = data.shape[1]
        results = numpy.zeros((nFree, nRows, nColumns), numpy.float32)
        uncertainties = numpy.zeros((nFree, nRows, nColumns), numpy.float32)
        if savefit or saveresiduals:
            fitmodel = numpy.zeros_like(data, dtype=numpy.float32)
            fitmodel[..., 0:iXMin] = numpy.nan
            fitmodel[..., iXMax:] = numpy.nan
        else:
            fitmodel = None

        _logger.debug("Configuration elapsed = %f", time.time() - t0)
        t0 = time.time()

        # Fit all spectra
        self._fit_lstsq_all(data=data, nChan=nChan, idxChan=idxChan,
                        derivatives=derivatives, fitmodel=fitmodel,
                        results=results, uncertainties=uncertainties,
                        config=config, anchorslist=anchorslist,
                        lstsq_kwargs=lstsq_kwargs)

        t = time.time() - t0
        _logger.debug("First fit elapsed = %f", t)
        if t > 0.:
            _logger.debug("Spectra per second = %f",
                          data.shape[0]*data.shape[1]/float(t))
        t0 = time.time()

        # Refit spectra with negative peak areas
        if refit:
            self._fit_lstsq_negpeaks(data=data, nChan=nChan, idxChan=idxChan,
                        freeNames=freeNames, nFreeBkg=nFreeBkg,
                        derivatives=derivatives, fitmodel=fitmodel,
                        results=results, uncertainties=uncertainties,
                        config=config, anchorslist=anchorslist,
                        lstsq_kwargs=lstsq_kwargs)
            t = time.time() - t0
            _logger.debug("Fit of negative peaks elapsed = %f", t)
            t0 = time.time()

        # Return results as a dictionary
        outputDict = {'configuration': configorg,
                      'parameters': results,
                      'uncertainties': uncertainties,
                      'names': freeNames}
        if savefit:
            outputDict['model'] = fitmodel
        if saveresiduals:
            outputDict['residuals'] = data - fitmodel
        if savedata:
            outputDict['data'] = data
        if savefit or saveresiduals or savedata:
            outputDict['axes_used'] = ('row', 'column', 'energy')
            xdata = self._mcaTheory.xdata0.flatten()
            zero, gain = self._mcaTheory.parameters[:2]
            xenergy = zero + gain*xdata
            outputDict['axes'] = \
                    (('row', numpy.arange(data.shape[0]), {}),
                    ('column', numpy.arange(data.shape[1]), {}),
                    ('channels', xdata, {}),
                    ('energy', xenergy, {'units': 'keV'}))
        if concentrations:
            t0 = time.time()
            self._fit_concentration(config=config,
                                    outputDict=outputDict,
                                    results=results,
                                    autotime=autotime,
                                    liveTimeFactor=liveTimeFactor)
            t = time.time() - t0
            _logger.debug("Calculation of concentrations elapsed = %f", t)
        return outputDict

    def _fit_parse_data(self, x=None, y=None, livetime=None):
        """Parse the input data (MCA and livetime)
        """
        if y is None:
            raise RuntimeError("y keyword argument is mandatory!")
        if hasattr(y, "info") and hasattr(y, "data"):
            data = y.data
            mcaIndex = y.info.get("McaIndex", -1)
        else:
            data = y
            mcaIndex = -1
        if x is None:
            if hasattr(y, "info") and hasattr(y, "x"):
                x = y.x[0]
        if livetime is None:
            if hasattr(y, "info"):
                if "McaLiveTime" in y.info:
                    livetime = y.info["McaLiveTime"]
        return x, data, mcaIndex, livetime

    def _fit_configure(self, configuration=None, concentrations=False,
                       livetime=None, weight=None, nSpectra=None):
        """Prepare configuration for fitting
        """
        if configuration is not None:
            self._mcaTheory.setConfiguration(configuration)
        elif self._config is None:
            raise ValueError("Fit configuration missing")
        else:
            _logger.debug("Setting default configuration")
            self._mcaTheory.setConfiguration(self._config)
        # read the current configuration
        # it is a copy, we can modify it at will
        configorg = self._mcaTheory.getConfiguration()
        config = self._mcaTheory.getConfiguration()
        toReconfigure = False

        # if concentrations and use times, it needs to be reconfigured
        # without using times and correct later on. If the concentrations
        # are to be calculated from internal standard there is no need to
        # raise an exception either.
        autotime = 0
        liveTimeFactor = 1.0
        if not concentrations:
            # ignore any time information to prevent unnecessary errors when
            # setting the fitting data whithout the time information
            if config['concentrations'].get("useautotime", 0):
                config['concentrations']["useautotime"] = 0
                toReconfigure = True
        elif config["concentrations"]["usematrix"]:
            if config['concentrations'].get("useautotime", 0):
                config['concentrations']["useautotime"] = 0
                toReconfigure = True
        else:
            # we are calculating concentrations from fundamental parameters
            autotime = config['concentrations'].get("useautotime", 0)
            if autotime:
                if livetime is None:
                    txt = "Automatic time requested but no time information provided"
                    raise RuntimeError(txt)
                elif numpy.isscalar(livetime):
                    liveTimeFactor = \
                        float(config['concentrations']["time"]) / livetime
                elif livetime.size == nSpectra:
                    liveTimeFactor = \
                        float(config['concentrations']["time"]) / livetime
                else:
                    raise RuntimeError(
                        "Number of live times not equal number of spectra")
                config['concentrations']["useautotime"] = 0
                toReconfigure = True

        # use of strategies is not supported for the time being
        strategy = config['fit'].get('strategyflag', 0)
        if strategy:
            raise RuntimeError("Strategies are incompatible with fast fit")

        # background
        if config['fit']['stripflag']:
            if config['fit']['stripalgorithm'] == 1:
                _logger.debug("SNIP")
            else:
                raise RuntimeError("Please use the faster SNIP background")

        if weight is None:
            # dictated by the file
            weight = config['fit']['fitweight']
            if weight:
                # individual pixel weights (slow)
                weightPolicy = 2
            else:
                # No weight
                weightPolicy = 0
        elif weight == 1:
            # use average weight from the sum spectrum
            weightPolicy = 1
            if not config['fit']['fitweight']:
                config['fit']['fitweight'] = 1
                toReconfigure = True
        elif weight == 2:
            # individual pixel weights (slow)
            weightPolicy = 2
            if not config['fit']['fitweight']:
                config['fit']['fitweight'] = 1
                toReconfigure = True
            weight = 1
        else:
            # No weight
            weightPolicy = 0
            if config['fit']['fitweight']:
                config['fit']['fitweight'] = 0
                toReconfigure = True
            weight = 0

        if not config['fit']['linearfitflag']:
            # make sure we force a linear fit
            config['fit']['linearfitflag'] = 1
            toReconfigure = True

        if toReconfigure:
            # we must configure again the fit
            self._mcaTheory.setConfiguration(config)

        # make sure we calculate the matrix of the contributions
        self._mcaTheory.enableOptimizedLinearFit()

        return configorg, config, weight, weightPolicy, autotime, liveTimeFactor

    def _fit_mcasum_3d(self, data=None, mcaIndex=None, sumover='all'):
        """Get sum spectrum
        """
        if sumover == 'all':
            # Sum in chunks of maximal 5000 or nColumns spectra
            jStep = min(5000, data.shape[1])
            ysum = numpy.zeros((data.shape[mcaIndex],), numpy.float)
            for iRow in range(0, data.shape[0]):
                jStart = 0
                while jStart < data.shape[1]:
                    jEnd = min(jStart + jStep, data.shape[1])
                    ysum += data[iRow, jStart:jEnd, :].sum(axis=0, dtype=numpy.float)
                    jStart = jEnd
        elif sumover == 'first row':
            # Sum spectrum of the first row
            ysum = data[0, :, :].sum(axis=0, dtype=numpy.float)
        else:
            # First spectrum
            ysum = data[0, 0, :]
        return ysum

    def _fit_model(self):
        """Get linear model for fitting
        """
        # Initialize the derivatives
        self._mcaTheory.estimate()

        # now we can get the derivatives respect to the free parameters
        # These are the "derivatives" respect to the peaks
        # linearMatrix = self._mcaTheory.linearMatrix

        # but we are still missing the derivatives from the background
        nFree = 0
        freeNames = []
        nFreeBkg = 0
        for iParam, param in enumerate(self._mcaTheory.PARAMETERS):
            if self._mcaTheory.codes[0][iParam] != ClassMcaTheory.Gefit.CFIXED:
                nFree += 1
                freeNames.append(param)
                if iParam < self._mcaTheory.NGLOBAL:
                    nFreeBkg += 1
        if nFree == 0:
            txt = "No free parameters to be fitted!\n"
            txt += "No peaks inside fitting region?"
            raise ValueError(txt)

        # build the matrix of derivatives
        derivatives = None
        idx = 0
        for iParam, param in enumerate(self._mcaTheory.PARAMETERS):
            if self._mcaTheory.codes[0][iParam] == ClassMcaTheory.Gefit.CFIXED:
                continue
            deriv = self._mcaTheory.linearMcaTheoryDerivative(self._mcaTheory.parameters,
                                                              iParam,
                                                              self._mcaTheory.xdata)
            deriv.shape = -1
            if derivatives is None:
                derivatives = numpy.zeros((deriv.shape[0], nFree), numpy.float)
            derivatives[:, idx] = deriv
            idx += 1

        return derivatives, freeNames, nFree, nFreeBkg

    def _fit_bkg_anchorlist(self, config=None):
        """Get anchors for background subtraction
        """
        xdata = self._mcaTheory.xdata
        if config['fit']['stripflag']:
            anchorslist = []
            if config['fit']['stripanchorsflag']:
                if config['fit']['stripanchorslist'] is not None:
                    ravelled = numpy.ravel(xdata)
                    for channel in config['fit']['stripanchorslist']:
                        if channel <= ravelled[0]:
                            continue
                        index = numpy.nonzero(ravelled >= channel)[0]
                        if len(index):
                            index = min(index)
                            if index > 0:
                                anchorslist.append(index)
            if len(anchorslist) == 0:
                anchorslist = [0, self._mcaTheory.ydata.size - 1]
            anchorslist.sort()
        else:
            anchorslist = None
        return anchorslist

    def _fit_mcatrim_info(self, x=None):
        """Start and end channels for MCA trimming
        """
        xdata = self._mcaTheory.xdata

        # find the indices to be used for selecting the appropriate data
        # if the original x data were not ordered we have a problem
        # TODO: check for original ordering.
        if x is None:
            # we have an enumerated channels axis
            iXMin = xdata[0]
            iXMax = xdata[-1]
        else:
            iXMin = numpy.nonzero(x <= xdata[0])[0][-1]
            iXMax = numpy.nonzero(x >= xdata[-1])[0][0]
        # numpy 1.11.0 returns an array on previous expression
        # and then complains about a future deprecation warning
        # because of using an array and not an scalar in the selection
        if hasattr(iXMin, "shape"):
            if len(iXMin.shape):
                iXMin = iXMin[0]
        if hasattr(iXMax, "shape"):
            if len(iXMax.shape):
                iXMax = iXMax[0]
        return iXMin, iXMax+1

    def _fit_lstsq_all(self, data=None, nChan=None, idxChan=None, derivatives=None,
                   fitmodel=None, results=None, uncertainties=None,
                   config=None, anchorslist=None, lstsq_kwargs=None):
        """Fit all spectra
        """
        # Fit in chunks of maximal 100 or nColumns spectra
        jStep = min(100, data.shape[1])
        chunkshape = (nChan, jStep)
        chunkbuffer = numpy.zeros(chunkshape, numpy.float)
        chunkmodel = None
        bkgsub = config['fit']['stripflag']
        for iRow in range(0, data.shape[0]):
            jStart = 0
            while jStart < data.shape[1]:
                jEnd = min(jStart + jStep, data.shape[1])
                chunk = chunkbuffer[:, :(jEnd - jStart)]
                if bkgsub:
                    chunk[()] = data[iRow, jStart:jEnd, idxChan].T.copy()
                else:
                    chunk[()] = data[iRow, jStart:jEnd, idxChan].T
                if fitmodel is not None:
                    chunkmodel = fitmodel[iRow, jStart:jEnd, idxChan].T

                # Subtract background
                if bkgsub:
                    self._fit_bkg_subtract(chunk, config=config,
                                           anchorslist=anchorslist,
                                           fitmodel=chunkmodel)

                # Solve linear system of equations
                ddict = lstsq(derivatives, chunk,
                              digested_output=True, **lstsq_kwargs)

                # Save results
                lstsq_kwargs['last_svd'] = ddict.get('svd', None)
                parameters = ddict['parameters']
                results[:, iRow, jStart:jEnd] = parameters
                uncertainties[:, iRow, jStart:jEnd] = ddict['uncertainties']
                if chunkmodel is not None:
                    chunkmodel += numpy.dot(derivatives, parameters)
                jStart = jEnd

    def _fit_lstsq_reduced(self, data=None, mask=None, skipParams= None,
                    skipNames=None, nmin=None, nChan=None, idxChan=None,
                    derivatives=None, fitmodel=None, results=None, uncertainties=None,
                    config=None, anchorslist=None, lstsq_kwargs=None):
        """
        Fit reduced number of spectra (mask) with a reduced model (skipped parameters are fixed at zero)
        """
        npixels = mask.sum()
        nFree = results.shape[0]
        if npixels < nmin:
            _logger.debug("Not worth refitting #%d pixels", npixels)
            for iFree, name in zip(skipParams, skipNames):
                results[iFree][mask] = 0.0
                uncertainties[iFree][mask] = 0.0
                _logger.debug("%d pixels of parameter %s set to zero",
                                npixels, name)
        else:
            _logger.debug("Refitting #%d pixels", npixels)

            # Fit all selected spectra in one chunk
            bkgsub = config['fit']['stripflag']
            chunk = self._fit_reduce_mca(data=data, mask=mask,
                                         npixels=npixels, nChan=nChan,
                                         idxChan=idxChan, copy=bkgsub)
            chunk = chunk.T
            if fitmodel is None:
                chunkmodel = None
            else:
                chunkmodel = fitmodel[mask, idxChan].T

            # Subtract background
            if bkgsub:
                self._fit_bkg_subtract(chunk, config=config,
                                        anchorslist=anchorslist,
                                        fitmodel=chunkmodel)

            # Solve linear system of equations
            A = derivatives[:, [i for i in range(nFree) if i not in skipParams]]
            lstsq_kwargs['last_svd'] = None
            ddict = lstsq(A, chunk, digested_output=True,
                          **lstsq_kwargs)

            # Save results
            idx = 0
            for iFree in range(nFree):
                if iFree in skipParams:
                    results[iFree][mask] = 0.0
                    uncertainties[iFree][mask] = 0.0
                else:
                    results[iFree][mask] = ddict['parameters'][idx]
                    uncertainties[iFree][mask] = ddict['uncertainties'][idx]
                    idx += 1
            if chunkmodel is not None:
                chunkmodel += numpy.dot(A, ddict['parameters'])

    @staticmethod
    def _fit_reduce_mca(data=None, mask=None, npixels=None, copy=None,
                        nChan=None, idxChan=None):
        if data.dtype not in [numpy.float32, numpy.float64]:
            if data.itemsize < 5:
                data_dtype = numpy.float32
            else:
                data_dtype = numpy.float64
        else:
            data_dtype = data.dtype
        try:
            if data.dtype != data_dtype:
                chunk = numpy.zeros((int(npixels), nChan),
                                    data_dtype)
                chunk[:] = data[mask, idxChan]
            else:
                chunk = data[mask, idxChan]
                if copy:
                    chunk = chunk.copy()
            chunk.shape = npixels, -1
        except TypeError:
            # in case of dynamic arrays, two dimensional indices are not
            # supported by h5py
            chunkshape = (int(npixels), nChan)
            chunk = numpy.zeros(chunkshape, data_dtype)
            selectedIndices = numpy.nonzero(mask > 0)
            tmpData = numpy.zeros((1, nChan), data_dtype)
            oldDataRow = -1
            j = 0
            for i in range(len(selectedIndices[0])):
                j = selectedIndices[0][i]
                if j != oldDataRow:
                    tmpData = data[j]
                    olddataRow = j
                chunk[i] = tmpData[selectedIndices[1][i], idxChan]
        return chunk

    @staticmethod
    def _fit_bkg_subtract(spectra, config=None, anchorslist=None, fitmodel=None):
        """Subtract brackground from data and add it to fit model
        """
        for k in range(spectra.shape[1]):
            # obtain the smoothed spectrum
            background = SpecfitFuns.SavitskyGolay(spectra[:, k],
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
            spectra[:, k] -= background
            if fitmodel is not None:
                fitmodel[:, k] = background

    def _fit_lstsq_negpeaks(self, data=None, freeNames=None, nFreeBkg=None,
                            results=None, **kwargs):
        """Refit pixels with negative peak areas (remove the parameters from the model)
        """
        nFree = len(freeNames)
        iIter = 1
        nIter = 2 * (nFree - nFreeBkg) + iIter
        negativePresent = True
        while negativePresent:
            # Pixels with negative peak areas
            negList = []
            for iFree in range(nFreeBkg, nFree):
                negMask = results[iFree] < 0
                nNeg = negMask.sum()
                if nNeg > 0:
                    negList.append((nNeg, iFree, negMask))

            # No refit needed when no negative peak areas
            if not negList:
                negativePresent = False
                continue

            # Set negative peak areas to zero when
            # the maximal iterations is reached
            if iIter > nIter:
                for nNeg, iFree, negMask in negList:
                    results[iFree][negMask] = 0.0
                    _logger.warning("%d pixels of parameter %s forced to zero",
                                    nNeg, freeNames[iFree])
                continue

            # Bad pixels: use peak area with the most negative values
            negList.sort()
            negList.reverse()
            badParameters = []
            badParameters.append(negList[0][1])
            badMask = negList[0][2]

            # Combine with masks of all other peak areas
            # (unless none of them has negative pixels)
            # This is done to prevent endless loops:
            # if two or more parameters have common negative pixels
            # and one of them remains negative when forcing other one to zero
            for iFree, (nNeg, iFree, negMask) in enumerate(negList):
                if iFree not in badParameters and nNeg:
                    combMask = badMask & negMask
                    if combMask.sum():
                        badParameters.append(iFree)
                        badMask = combMask

            # Fit with a reduced model (skipped parameters are fixed at zero)
            badNames = [freeNames[iFree] for iFree in badParameters]
            nmin = 0.0025 * badMask.size
            _logger.debug("Refit iteration #{}. Fixed to zero: {}"
                          .format(iIter, badNames))
            self._fit_lstsq_reduced(data=data, mask=badMask,
                                    skipParams=badParameters,
                                    skipNames=badNames,
                                    results=results,
                                    nmin=nmin, **kwargs)
            iIter += 1

    def _fit_concentration(self, config=None, outputDict=None, results=None,
                           autotime=None, liveTimeFactor=None):
        """Calculate concentrations from peak areas
        """
        # check if an internal reference is used and if it is set to auto
        cTool = ConcentrationsTool.ConcentrationsTool()
        cToolConf = cTool.configure()
        cToolConf.update(config['concentrations'])

        fitysum = False
        if config['concentrations']['usematrix']:
            _logger.debug("USING MATRIX")
            if config['concentrations']['reference'].upper() == "AUTO":
                fitysum = True
        elif autotime:
            # we have to calculate with the time in the configuration
            # and correct later on
            cToolConf["autotime"] = 0

        fitresult = {}
        if fitysum:
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
            for iParam, param in enumerate(self._mcaTheory.PARAMETERS):
                if self._mcaTheory.codes[0][iParam] == Gefit.CFIXED:
                    continue
                if iParam < self._mcaTheory.NGLOBAL:
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
        nElements = len(list(concentrationsResult['mass fraction'].keys()))
        nFree, nRows, nColumns = results.shape
        massFractions = numpy.zeros((nValues * nElements, nRows, nColumns),
                                    numpy.float32)

        referenceElement = addInfo['ReferenceElement']
        referenceTransitions = addInfo['ReferenceTransitions']
        _logger.debug("Reference <%s>  transition <%s>",
                      referenceElement, referenceTransitions)
        if referenceElement in ["", None, "None"]:
            _logger.debug("No reference")
            counter = 0
            for i, group in enumerate(fitresult['result']['groups']):
                if group.lower().startswith("scatter"):
                    _logger.debug("skept %s", group)
                    continue
                outputDict['names'].append("C(%s)" % group)
                if counter == 0:
                    if hasattr(liveTimeFactor, "shape"):
                        liveTimeFactor.shape = results[nFreeBkg+i].shape
                massFractions[counter] = liveTimeFactor * \
                    results[nFreeBkg+i] * \
                    (concentrationsResult['mass fraction'][group] / \
                        fitresult['result'][group]['fitarea'])
                counter += 1
                if len(concentrationsResult['layerlist']) > 1:
                    for layer in concentrationsResult['layerlist']:
                        outputDict['names'].append("C(%s)-%s" % (group, layer))
                        massFractions[counter] = liveTimeFactor * \
                                results[nFreeBkg+i] * \
                                (concentrationsResult[layer]['mass fraction'][group] / \
                                    fitresult['result'][group]['fitarea'])
                        counter += 1
        else:
            _logger.debug("With reference")
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
            goodIdx = results[nFreeBkg+idx] > 0
            massFractions[idx] = referenceConcentrations
            counter = 0
            for i, group in enumerate(fitresult['result']['groups']):
                if group.lower().startswith("scatter"):
                    _logger.debug("skept %s", group)
                    continue
                outputDict['names'].append("C(%s)" % group)
                goodI = results[nFreeBkg+i] > 0
                tmp = results[nFreeBkg+idx][goodI]
                massFractions[counter][goodI] = (results[nFreeBkg+i][goodI]/(tmp + (tmp == 0))) *\
                            ((referenceArea/fitresult['result'][group]['fitarea']) *\
                            (concentrationsResult['mass fraction'][group]))
                counter += 1
                if len(concentrationsResult['layerlist']) > 1:
                    for layer in concentrationsResult['layerlist']:
                        outputDict['names'].append("C(%s)-%s" % (group, layer))
                        massFractions[counter][goodI] = (results[nFreeBkg+i][goodI]/(tmp + (tmp == 0))) *\
                            ((referenceArea/fitresult['result'][group]['fitarea']) *\
                            (concentrationsResult[layer]['mass fraction'][group]))
                        counter += 1
        outputDict['concentrations'] = massFractions


def getFileListFromPattern(pattern, begin, end, increment=None):
    if type(begin) == type(1):
        begin = [begin]
    if type(end) == type(1):
        end = [end]
    if len(begin) != len(end):
        raise ValueError(\
            "Begin list and end list do not have same length")
    if increment is None:
        increment = [1] * len(begin)
    elif type(increment) == type(1):
        increment = [increment]
    if len(increment) != len(begin):
        raise ValueError(
            "Increment list and begin list do not have same length")
    fileList = []
    if len(begin) == 1:
        for j in range(begin[0], end[0] + increment[0], increment[0]):
            fileList.append(pattern % (j))
    elif len(begin) == 2:
        for j in range(begin[0], end[0] + increment[0], increment[0]):
            for k in range(begin[1], end[1] + increment[1], increment[1]):
                fileList.append(pattern % (j, k))
    elif len(begin) == 3:
        raise ValueError("Cannot handle three indices yet.")
        for j in range(begin[0], end[0] + increment[0], increment[0]):
            for k in range(begin[1], end[1] + increment[1], increment[1]):
                for l in range(begin[2], end[2] + increment[2], increment[2]):
                    fileList.append(pattern % (j, k, l))
    else:
        raise ValueError("Cannot handle more than three indices.")
    return fileList


def save(result, outputDir, outputRoot=None, fileEntry=None,
         fileProcess=None, tif=False, edf=False, csv=False, h5=True):
    """
    Save result of Fast XRF fitting

    :param dict result:
    :param str outputDir: directory on local file system
    :param str outputRoot: output dir or HDF5 filename without extension (default: 'IMAGES')
    :param str fileEntry: output radix or NXentry (default: 'images')
    :param str fileProcess: output NXprocess (default: fileEntry)
    :param tif: save as tif
    :param edf: save as edf
    :param csv: save as csv
    :param h5: save as h5
    """
    if not (tif or edf or csv or h5):
        _logger.warning('fit result not saved (no output format specified)')
        return
    t0 = time.time()
    _logger.debug('Saving results ...')

    # For .edf: outputDir/outputRoot/fileEntry.edf
    # For .h5 : outputDir/outputRoot.h5::/fileEntry/fileProcess
    if not outputRoot:
        outputRoot = "IMAGES"
    if not fileEntry:
        fileEntry = "images"
    if not fileProcess:
        fileProcess = fileEntry

    # List of images and labels
    imageList = []
    imageLabels = []
    for i, name in enumerate(result['names']):
        newname = name.replace(" ","-")
        if newname.startswith("C("):
            # Elemental concentration
            img = result['concentrations'][i]
            imageList.append(img)
            imageLabels.append(newname)
        else:
            # Fitted parameters
            img = result['parameters'][i]
            imageList.append(img)
            imageLabels.append(newname)
            newname = "s(%s)" % newname
            img = result['uncertainties'][i]
            imageList.append(img)
            imageLabels.append(newname)

    if tif or edf or csv:
        from PyMca5.PyMca import ArraySave

        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        imagesDir = os.path.join(outputDir, outputRoot)
        if not os.path.exists(imagesDir):
            os.mkdir(imagesDir)

        if edf:
            fileName = os.path.join(imagesDir, fileEntry+".edf")
            ArraySave.save2DArrayListAsEDF(imageList, fileName,
                                            labels=imageLabels)
        if csv:
            fileName = os.path.join(imagesDir, fileEntry+".csv")
            ArraySave.save2DArrayListAsASCII(imageList, fileName, csv=True,
                                                labels=imageLabels)
        if tif:
            for label,image in zip(imageLabels,imageList):
                if label.startswith("s("):
                    suffix = "_s" + label[2:-1]
                elif label.startswith("C("):
                    suffix = "_w" + label[2:-1]
                else:
                    suffix  = "_" + label
                fileName = os.path.join(imagesDir,
                                        fileEntry + suffix + ".tif")
                ArraySave.save2DArrayListAsMonochromaticTiff([image],
                                        fileName,
                                        labels=[label],
                                        dtype=numpy.float32)
        result['configuration'].write(os.path.join(imagesDir, fileEntry+".cfg"))

    if h5:
        filename = os.path.join(outputDir, outputRoot+'.h5')
        with NexusUtils.nxroot(filename, mode='a') as root:
            # Create fileEntry/fileProcess
            entry = NexusUtils.nxentry(root, fileEntry)
            if fileProcess in entry:
                # Overwrite like ArraySave does for the other formats
                _logger.warning('overwriting {}'.format(entry[fileProcess].name))
                del entry[fileProcess]
            process = NexusUtils.nxprocess(entry, fileProcess,
                                           configdict=result['configuration'])

            # Fitted parameters
            imageList = [{'data':img, 'chunks':True} for img in imageList]
            imageAttrs = [{'interpretation':'image'}]*len(imageLabels)
            data = NexusUtils.nxdata(process['results'], 'parameters')
            signals = [signal for signal in zip(imageLabels,imageList,imageAttrs) if not signal[0].startswith("s(")]
            NexusUtils.nxdata_add_signals(data, signals)
            NexusUtils.mark_default(data)

            # Estimated parameter errors
            data = NexusUtils.nxdata(process['results'], 'uncertainties')
            signals = [signal for signal in zip(imageLabels,imageList,imageAttrs) if signal[0].startswith("s(")]
            NexusUtils.nxdata_add_signals(data, signals)

            # Fitted model and residuals
            signals = []
            attr = {'interpretation':'spectrum'}
            for name in ['data', 'model', 'residuals']:
                if name in result:
                    value = {'data':result[name], 'chunks':True}
                    signals.append((name,value,attr))
            if signals:
                data = NexusUtils.nxdata(process['results'], 'fit')
                NexusUtils.nxdata_add_signals(data, signals)
                if 'axes' in result:
                    NexusUtils.nxdata_add_axes(data, result['axes'])
                if 'axes_used' in result:
                    axes = [(ax, None, None) for ax in result['axes_used']]
                    NexusUtils.nxdata_add_axes(data, axes, append=False)

    t = time.time() - t0
    _logger.debug("Saving results elapsed = %f", t)

def prepareDataStack(fileList):
    if (not os.path.exists(fileList[0])) and \
        os.path.exists(fileList[0].split("::")[0]):
        # odo convention to get a dataset form an HDF5
        fname, dataPath = fileList[0].split("::")
        # compared to the ROI imaging tool, this way of reading puts data
        # into memory while with the ROI imaging tool, there is a check.
        if 0:
            import h5py
            h5 = h5py.File(fname, "r")
            dataStack = h5[dataPath][:]
            h5.close()
        else:
            from PyMca5.PyMcaIO import HDF5Stack1D
            # this way reads information associated to the dataset (if present)
            if dataPath.startswith("/"):
                pathItems = dataPath[1:].split("/")
            else:
                pathItems = dataPath.split("/")
            if len(pathItems) > 1:
                scanlist = ["/" + pathItems[0]]
                selection = {"y":"/" + "/".join(pathItems[1:])}
            else:
                selection = {"y":dataPath}
                scanlist = None
            print(selection)
            print("scanlist = ", scanlist)
            dataStack = HDF5Stack1D.HDF5Stack1D([fname],
                                                selection,
                                                scanlist=scanlist)
    else:
        from PyMca5.PyMca import EDFStack
        dataStack = EDFStack.EDFStack(fileList, dtype=numpy.float32)
    return dataStack


def main():
    import sys
    import getopt
    options     = ''
    longoptions = ['cfg=', 'outdir=', 'concentrations=', 'weight=', 'refit=',
                   'tif=', 'edf=', 'csv=', 'h5=',
                   'filepattern=', 'begin=', 'end=', 'increment=',
                   'outroot=', 'outentry=', 'outprocess=',
                   'savefit=', 'saveresiduals=', 'savedata=']
    try:
        opts, args = getopt.getopt(
                     sys.argv[1:],
                     options,
                     longoptions)
    except:
        print(sys.exc_info()[1])
        sys.exit(1)
    outputDir = None
    outputRoot = ""
    fileEntry = ""
    fileProcess = ""
    refit = None
    filepattern = None
    begin = None
    end = None
    increment = None
    backend = None
    weight = 0
    tif = 0
    edf = 0
    csv = 0
    h5 = 1
    concentrations = 0
    savefit = 0
    saveresiduals = 0
    savedata = 0
    for opt, arg in opts:
        if opt == '--cfg':
            configurationFile = arg
        elif opt == '--begin':
            if "," in arg:
                begin = [int(x) for x in arg.split(",")]
            else:
                begin = [int(arg)]
        elif opt == '--end':
            if "," in arg:
                end = [int(x) for x in arg.split(",")]
            else:
                end = int(arg)
        elif opt == '--increment':
            if "," in arg:
                increment = [int(x) for x in arg.split(",")]
            else:
                increment = int(arg)
        elif opt == '--filepattern':
            filepattern = arg.replace('"', '')
            filepattern = filepattern.replace("'", "")
        elif opt == '--outdir':
            outputDir = arg
        elif opt == '--weight':
            weight = int(arg)
        elif opt == '--refit':
            refit = int(arg)
        elif opt == '--concentrations':
            concentrations = int(arg)
        elif opt == '--savefit':
            savefit = int(arg)
        elif opt == '--saveresiduals':
            saveresiduals = int(arg)
        elif opt == '--savedata':
            savedata = int(arg)
        elif opt == '--outroot':
            outputRoot = arg
        elif opt == '--outentry':
            fileEntry = arg
        elif opt == '--outprocess':
            fileProcess = arg
        elif opt in ('--tif', '--tiff'):
            tif = int(arg)
        elif opt == '--edf':
            edf = int(arg)
        elif opt == '--csv':
            csv = int(arg)
        elif opt == '--h5':
            h5 = int(arg)

    if filepattern is not None:
        if (begin is None) or (end is None):
            raise ValueError(\
                "A file pattern needs at least a set of begin and end indices")
    if filepattern is not None:
        fileList = getFileListFromPattern(filepattern, begin, end, increment=increment)
    else:
        fileList = args
    if refit is None:
        refit = 0
        _logger.warning("--refit=%d taken as default" % refit)
    if len(fileList):
        dataStack = prepareDataStack(fileList)
    else:
        print("OPTIONS:", longoptions)
        sys.exit(0)
    if outputDir is None:
        print("RESULTS WILL NOT BE SAVED: No output directory specified")

    t0 = time.time()
    fastFit = FastXRFLinearFit()
    fastFit.setFitConfigurationFile(configurationFile)
    print("Main configuring Elapsed = % s " % (time.time() - t0))
    if not h5:
        savefit = False
        saveresiduals = False
    result = fastFit.fitMultipleSpectra(y=dataStack,
                                        weight=weight,
                                        refit=refit,
                                        concentrations=concentrations,
                                        savedata=savedata,
                                        savefit=savefit,
                                        saveresiduals=saveresiduals)
    print("Total Elapsed = % s " % (time.time() - t0))

    if outputDir is not None:
        save(result, outputDir, outputRoot=outputRoot,
            fileEntry=fileEntry, fileProcess=fileProcess,
             tif=tif, edf=edf, csv=csv, h5=h5)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _logger.setLevel(logging.DEBUG)
    main()