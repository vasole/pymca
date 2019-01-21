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
import time
import h5py
import collections
from contextlib import contextmanager
from . import ClassMcaTheory
from . import ConcentrationsTool
from PyMca5.PyMcaMath.linalg import lstsq
from PyMca5.PyMcaMath.fitting import Gefit
from PyMca5.PyMcaMath.fitting import SpecfitFuns
from PyMca5.PyMcaIO import ConfigDict
from PyMca5.PyMcaIO import NexusUtils
from PyMca5.PyMcaMisc import PhysicalMemory


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
                           outbuffer=None):
        """
        This method performs the actual fit. The y keyword is the only mandatory input argument.

        :param x: 1D array containing the x axis (usually the channels) of the spectra.
        :param y: 3D array containing the spectra as [nrows, ncolumns, nchannels]
        :param xmin: lower limit of the fitting region
        :param xmax: upper limit of the fitting region
        :param ysum: sum spectrum
        :param weight: 0 Means no weight, 1 Use an average weight, 2 Individual weights (slow)
        :param concentrations: 0 Means no calculation, 1 Calculate elemental concentrations
        :param refit: if False, no check for negative results. Default is True.
        :livetime: It will be used if not different from None and concentrations
                   are to be calculated by using fundamental parameters with
                   automatic time. The default is None.
        :outbuffer dict: 
        :return dict: outbuffer
        """
        # Parse data
        x, data, mcaIndex, livetime = self._fit_parse_data(x=x, y=y,
                                                           livetime=livetime)

        # Check data dimensions
        if data.ndim != 3:
            txt = "For the time being only three dimensional arrays supported"
            raise IndexError(txt)
        elif mcaIndex not in [-1, 2]:
            txt = "For the time being only mca arrays supported"
            raise IndexError(txt)

        if outbuffer is None:
            outbuffer = FastFitOutputBuffer()

        with outbuffer._buffer_context(update=False):
            t0 = time.time()

            # Configure fit
            nSpectra = data.size // data.shape[mcaIndex]
            configorg, config, weight, weightPolicy, \
            autotime, liveTimeFactor = self._fit_configure(
                                                configuration=configuration,
                                                concentrations=concentrations,
                                                livetime=livetime,
                                                weight=weight,
                                                nSpectra=nSpectra)
            outbuffer['configuration'] = configorg

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
                yref = self._fit_reference_spectrum(data=data, sumover=sumover)
            else:
                yref = ysum

            # Get the basis of the linear models (i.e. derivative to peak areas)
            if xmin is None:
                xmin = config['fit']['xmin']
            if xmax is None:
                xmax = config['fit']['xmax']
            dtype = self._fit_dtype(data)
            self._mcaTheory.setData(x=x, y=yref, xmin=xmin, xmax=xmax)
            derivatives, freeNames, nFree, nFreeBkg = self._fit_model(dtype=dtype)
            outbuffer['parameter_names'] = freeNames

            # Background anchor points (if any)
            anchorslist = self._fit_bkg_anchorlist(config=config)

            # MCA trimming: [iXMin:iXMax]
            iXMin, iXMax = self._fit_mcatrim_info(x=x)
            sliceChan = slice(iXMin, iXMax)
            nObs = iXMax-iXMin

            # Least-squares parameters
            if weightPolicy == 2:
                # Individual spectrum weights (assumed Poisson)
                SVD = False
                sigma_b = None
            elif weightPolicy == 1:
                # Average weight from sum spectrum (assume Poisson)
                # the +1 is to prevent misbehavior due to weights less than 1.0
                sigma_b = 1 + numpy.sqrt(yref[sliceChan])/nSpectra
                sigma_b = sigma_b.reshape(-1, 1)
                SVD = True
            else:
                # No weights
                SVD = True
                sigma_b = None
            lstsq_kwargs = {'svd': SVD, 'sigma_b': sigma_b, 'weight': weight}

            # Allocate output buffers
            nRows, nColumns, nChan = data.shape
            param_shape = nFree, nRows, nColumns
            image_shape = nRows, nColumns
            results = outbuffer.allocate_memory('parameters',
                                                shape=param_shape,
                                                dtype=dtype)
            uncertainties = outbuffer.allocate_memory('uncertainties',
                                                      shape=param_shape,
                                                      dtype=dtype)
            if outbuffer.save_diagnostics:
                nFreeParameters = outbuffer.allocate_memory('nFreeParameters',
                                                 shape=image_shape,
                                                 fill_value=nFree)
                _ = outbuffer.allocate_memory('nObservations',
                                                 shape=image_shape,
                                                 fill_value=nObs)
                fitmodel = outbuffer.allocate_h5('model',
                                                 nxdata='fit',
                                                 shape=data.shape,
                                                 dtype=dtype,
                                                 chunks=True,
                                                 fill_value=0)
                fitmodel[..., 0:iXMin] = numpy.nan
                fitmodel[..., iXMax:] = numpy.nan
            else:
                fitmodel = None
                nFreeParameters = None

            _logger.debug("Configuration elapsed = %f", time.time() - t0)
            t0 = time.time()

            # Fit all spectra
            self._fit_lstsq_all(data=data, sliceChan=sliceChan,
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
                self._fit_lstsq_negpeaks(data=data, sliceChan=sliceChan,
                            freeNames=freeNames, nFreeBkg=nFreeBkg,
                            derivatives=derivatives, fitmodel=fitmodel,
                            results=results, uncertainties=uncertainties,
                            nFreeParameters=nFreeParameters,
                            config=config, anchorslist=anchorslist,
                            lstsq_kwargs=lstsq_kwargs)
                t = time.time() - t0
                _logger.debug("Fit of negative peaks elapsed = %f", t)
                t0 = time.time()

            # Return results as a dictionary
            outaxes = False
            if outbuffer.save_data:
                outaxes = True
                outbuffer.allocate_h5('data', nxdata='fit', data=data, chunks=True)
            if outbuffer.save_residuals:
                outaxes = True
                residuals = outbuffer.allocate_h5('residuals', nxdata='fit', data=data, chunks=True)
                residuals[()] -= fitmodel
            if outaxes:
                outbuffer['data_axes_used'] = ('row', 'column', 'energy')
                xdata = self._mcaTheory.xdata0.flatten()
                zero, gain = self._mcaTheory.parameters[:2]
                xenergy = zero + gain*xdata
                outbuffer['data_axes'] = \
                        (('row', numpy.arange(data.shape[0]), {}),
                        ('column', numpy.arange(data.shape[1]), {}),
                        ('channels', xdata, {}),
                        ('energy', xenergy, {'units': 'keV'}))
            if concentrations:
                t0 = time.time()
                self._fit_concentration(config=config,
                                        outputDict=outbuffer,
                                        nFreeBkg=nFreeBkg,
                                        results=results,
                                        autotime=autotime,
                                        liveTimeFactor=liveTimeFactor)
                t = time.time() - t0
                _logger.debug("Calculation of concentrations elapsed = %f", t)
            return outbuffer

    @staticmethod
    def _fit_parse_data(x=None, y=None, livetime=None):
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

        return configorg, config, weight, weightPolicy, \
               autotime, liveTimeFactor

    def _fit_reference_spectrum(self, data=None, sumover='all'):
        """Get sum spectrum
        """
        dtype = self._fit_dtype(data)
        if sumover == 'all':
            #nspectra = PhysicalMemory.chunks_in_memory(data.shape, data.dtype,
            #                                   axis=-1, maximal=5000)
            datastack = ChunkedMcaStack(data, nmca=5000)
            yref = numpy.zeros((data.shape[-1],), dtype)
            for _, chunk in datastack.items():
                yref += chunk.sum(axis=0, dtype=dtype)
        elif sumover == 'first row':
            # Sum spectrum of the first row
            yref = data[0, :, :].sum(axis=0, dtype=dtype)
        else:
            # First spectrum
            yref = data[0, 0, :].astype(dtype)
        return yref

    def _fit_model(self, dtype=None):
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
                derivatives = numpy.zeros((deriv.shape[0], nFree), dtype=dtype)
            derivatives[:, idx] = deriv
            idx += 1

        return derivatives, freeNames, nFree, nFreeBkg

    def _fit_bkg_anchorlist(self, config=None):
        """Get anchors for background subtraction
        """
        xdata = self._mcaTheory.xdata  # trimmed
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

    def _data_iter(self, slicecls, data=None, fitmodel=None, copy=True, **kwargs):
        dtype = self._fit_dtype(data)
        datastack = slicecls(data, dtype=dtype, modify=False, **kwargs)
        chunkiter = datastack.items(copy=copy)
        if fitmodel is not None:
            modelstack = slicecls(fitmodel, dtype=dtype, modify=True, **kwargs)
            modeliter = modelstack.items(copy=False)
            chunkiter = izip_clean(chunkiter, modeliter)
        return chunkiter

    def _fit_lstsq_all(self, data=None, sliceChan=None, derivatives=None,
                   fitmodel=None, results=None, uncertainties=None,
                   config=None, anchorslist=None, lstsq_kwargs=None):
        """
        Fit all spectra
        """
        bkgsub = bool(config['fit']['stripflag'])
        #nspectra = PhysicalMemory.chunks_in_memory(data.shape, data.dtype,
        #                                   axis=-1, maximal=100)
        chunkiter = self._data_iter(ChunkedMcaStack, data=data, fitmodel=fitmodel,
                                    mcaslice=sliceChan, copy=bkgsub, nmca=100)
        for chunk in chunkiter:
            if fitmodel is None:
                idx, chunk = chunk
                chunkmodel = None
            else:
                (idx, chunk), (idxmodel, chunkmodel) = chunk
                chunkmodel = chunkmodel.T
            chunk = chunk.T

            # Subtract background
            if bkgsub:
                self._fit_bkg_subtract(chunk, config=config,
                                       anchorslist=anchorslist,
                                       fitmodel=chunkmodel)

            # Solve linear system of equations
            ddict = lstsq(derivatives, chunk, digested_output=True,
                          **lstsq_kwargs)
            lstsq_kwargs['last_svd'] = ddict.get('svd', None)

            # Save results
            idx = slice(None), idx[0], idx[1]
            results[idx] = ddict['parameters']
            uncertainties[idx] = ddict['uncertainties']
            if chunkmodel is not None:
                if bkgsub:
                    chunkmodel += numpy.dot(derivatives, ddict['parameters'])
                else:
                    chunkmodel[()] = numpy.dot(derivatives, ddict['parameters'])

    def _fit_lstsq_reduced(self, data=None, mask=None, skipParams= None,
                    skipNames=None, nmin=None, sliceChan=None,
                    derivatives=None, fitmodel=None, results=None,
                    uncertainties=None, nFreeParameters=None,
                    config=None, anchorslist=None, lstsq_kwargs=None):
        """
        Fit reduced number of spectra (mask) with a reduced model (skipped parameters will be set to zero)
        """
        npixels = int(mask.sum())
        if npixels < nmin:
            _logger.debug("Not worth refitting #%d pixels", npixels)
            for iFree, name in zip(skipParams, skipNames):
                results[iFree][mask] = 0.0
                uncertainties[iFree][mask] = 0.0
                _logger.debug("%d pixels of parameter %s set to zero",
                              npixels, name)
            if nFreeParameters is not None:
                nFreeParameters[mask] = 0
        else:
            _logger.debug("Refitting #%d pixels", npixels)
            nFreeOrg = results.shape[0]
            idxSel = [i for i in range(nFreeOrg) if i not in skipParams]
            nFree = len(idxSel)
            A = derivatives[:, idxSel]
            lstsq_kwargs['last_svd'] = None

            # Fit all selected spectra in one chunk
            bkgsub = bool(config['fit']['stripflag'])
            chunkiter = self._data_iter(MaskedMcaStack, data=data, fitmodel=fitmodel,
                                        mcaslice=sliceChan, copy=bkgsub, mask=mask)
            for chunk in chunkiter:
                if fitmodel is None:
                    idx, chunk = chunk
                    chunkmodel = None
                else:
                    (idx, chunk), (idxmodel, chunkmodel) = chunk
                    chunkmodel = chunkmodel.T
                chunk = chunk.T

                # Subtract background
                if bkgsub:
                    self._fit_bkg_subtract(chunk, config=config,
                                           anchorslist=anchorslist,
                                           fitmodel=chunkmodel)

                # Solve linear system of equations
                ddict = lstsq(A, chunk, digested_output=True,
                              **lstsq_kwargs)
                lstsq_kwargs['last_svd'] = ddict.get('svd', None)

                # Save results
                iParam = 0
                idx = mask  # for now MaskedMcaStack only has one chunk
                for iFree in range(nFreeOrg):
                    if iFree in skipParams:
                        results[iFree][idx] = 0.0
                        uncertainties[iFree][idx] = 0.0
                    else:
                        results[iFree][idx] = ddict['parameters'][iParam]
                        uncertainties[iFree][idx] = ddict['uncertainties'][iParam]
                        iParam += 1
                if chunkmodel is not None:
                    if bkgsub:
                        chunkmodel += numpy.dot(A, ddict['parameters'])
                    else:
                        chunkmodel[()] = numpy.dot(A, ddict['parameters'])

            if nFreeParameters is not None:
                nFreeParameters[mask] = nFree

    @staticmethod
    def _fit_dtype(data):
        if data.dtype not in [numpy.float32, numpy.float64]:
            if data.itemsize < 5:
                return numpy.float32
            else:
                return numpy.float64
        else:
            return data.dtype

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
                           nFreeBkg=None, autotime=None, liveTimeFactor=None):
        """Calculate concentrations from peak areas
        """
        # check if an internal reference is used and if it is set to auto
        cTool = ConcentrationsTool.ConcentrationsTool()
        cToolConf = cTool.configure()
        cToolConf.update(config['concentrations'])

        fitreference = False
        if config['concentrations']['usematrix']:
            _logger.debug("USING MATRIX")
            if config['concentrations']['reference'].upper() == "AUTO":
                fitreference = True
        elif autotime:
            # we have to calculate with the time in the configuration
            # and correct later on
            cToolConf["autotime"] = 0

        fitresult = {}
        if fitreference:
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
                                    dtype=results.dtype)

        referenceElement = addInfo['ReferenceElement']
        referenceTransitions = addInfo['ReferenceTransitions']
        _logger.debug("Reference <%s>  transition <%s>",
                      referenceElement, referenceTransitions)
        outputDict['concentration_names'] = []
        if referenceElement in ["", None, "None"]:
            _logger.debug("No reference")
            counter = 0
            for i, group in enumerate(fitresult['result']['groups']):
                if group.lower().startswith("scatter"):
                    _logger.debug("skept %s", group)
                    continue
                outputDict['concentration_names'].append(group)
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
                        outputDict['concentration_names'].append("%s-%s" % (group, layer))
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
                outputDict['concentration_names'].append(group)
                goodI = results[nFreeBkg+i] > 0
                tmp = results[nFreeBkg+idx][goodI]
                massFractions[counter][goodI] = (results[nFreeBkg+i][goodI]/(tmp + (tmp == 0))) *\
                            ((referenceArea/fitresult['result'][group]['fitarea']) *\
                            (concentrationsResult['mass fraction'][group]))
                counter += 1
                if len(concentrationsResult['layerlist']) > 1:
                    for layer in concentrationsResult['layerlist']:
                        outputDict['concentration_names'].append("%s-%s" % (group, layer))
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


def slice_len(slc, n):
    start, stop, step = slc.indices(n)
    if step < 0:
        one = -1
    else:
        one = 1
    return max(0, (stop - start + step - one) // step)


def chunk_indices(start, stop=None, step=None):
    """
    :param start:
    :param stop:
    :param step:
    :returns list: list of (slice,n)
    """
    if stop is None:
        start, stop, step = 0, start, 1
    elif step is None:
        step = 1
    if step < 0:
        func = max
        one = -1
    else:
        func = min
        one = 1
    for a in range(start, stop, step):
        b = func(a+step, stop)
        yield slice(a, b), abs(b-a)


# TODO: nonlocal in python 2 and 3
#def izip_clean(*iters):
#    """
#    Like Python 3's zip but making sure next is called
#    on all items when StopIteration occurs
#    """
#    bloop = True
#    def _next(it):
#        nonlocal bloop
#        try:
#            return next(it)
#        except StopIteration:
#            bloop = False
#            return None
#    while bloop:
#        ret = tuple(_next(it) for it in iters)
#        if bloop:
#            yield ret
def izip_clean(*iters):
    """
    Like Python 3's zip but making sure next is called
    on all items when StopIteration occurs
    """
    bloop = {'b': True}  # because of python 2
    def _next(it):
        try:
            return next(it)
        except StopIteration:
            bloop['b'] = False
            return None
    while bloop['b']:
        ret = tuple(_next(it) for it in iters)
        if bloop['b']:
            yield ret


class SlicedMcaStack(object):

    def __init__(self, data, nbuffer=None, mcaslice=None, dtype=None, modify=False):
        """
        :param array data: 3D array (n0, n1, n2)
        :param num nbuffer: number of elements from (n0, n1) in buffer
        :param slice mcaslice: slice MCA axis=2
        :param dtype:
        :param bool modify: modify original on access
        """
        # Buffer shape
        n2 = data.shape[-1]
        if mcaslice:
            nChan = slice_len(mcaslice, n2)
        else:
            nChan = n2
            mcaslice = slice(None)
        self._mcaslice = mcaslice
        self._buffershape = nbuffer, nChan
        self._buffer = None

        # Buffer dtype
        if dtype is None:
            dtype = data.dtype
        self._dtype = dtype
        self._change_type = data.dtype != dtype

        # Data
        self._data = data
        self._modify = modify
        self._isndarray = isinstance(data, numpy.ndarray)

    def _prepare_access(self, copy=True):
        _logger.debug('Iterate MCA stack in chunks of {} spectra'
                      .format(self._buffershape[0]))
        needs_copy = copy or self._change_type
        yields_copy = needs_copy or not self._isndarray
        post_copy = yields_copy and self._modify
        if yields_copy and self._buffer is None:
            self._buffer = numpy.empty(self._buffershape, self._dtype)
        return needs_copy, yields_copy, post_copy

    def items(self, copy=True):
        raise NotImplementedError


class MaskedMcaStack(SlicedMcaStack):

    def __init__(self, data, mask=None, **kwargs):
        """
        3D stack (n0, n1, n2) with mask (n0, n1) and slice n2

        :param array data: numpy array or h5py dataset
        :param array mask: shape = (n0, n1)
        :param \**kwargs: see SlicedMcaStack
        """
        if mask is None:
            self._indices = None
            self._mask = Ellipsis
            nbuffer = data.shape[0]*data.shape[1]
        else:
            if isinstance(data, numpy.ndarray):
                # Support multiple advanced indexes
                self._indices = None
                self._mask = mask
                nbuffer = int(mask.sum())
            else:
                # Does not support multiple advanced indexes
                self._indices = numpy.where(mask)
                self._mask = None
                nbuffer = len(self._indices[0])
        super(MaskedMcaStack, self).__init__(data, nbuffer=nbuffer, **kwargs)

    def items(self, copy=True):
        needs_copy, _, post_copy = self._prepare_access(copy=copy)
        if self._indices is None:
            ret = self._get_with_mask(copy=needs_copy)
        else:
            ret = self._get_with_indices()
        yield ret
        if post_copy:
            if self._indices is None:
                self._set_with_mask(*ret)
            else:
                self._set_with_indices(*ret)

    def _get_with_mask(self, copy=True):
        idx = self._mask, self._mcaslice
        if copy:
            chunk = self._buffer
            chunk[()] = self._data[idx]
        else:
            chunk = self._data[idx]
        return idx, chunk

    def _set_with_mask(self, idx, chunk):
        self._data[idx] = chunk

    def _get_with_indices(self):
        chunk = self._buffer
        mcaslice = self._mcaslice
        idx = self._indices + (mcaslice,)
        j0keep = -1
        for i, (j0, j1) in enumerate(zip(*self._indices)):
            if j0 != j0keep:
                tmpData = self._data[j0]
                j0keep = j0
            chunk[i] = tmpData[j1, mcaslice]
        return idx, chunk

    def _set_with_indices(self, idx, chunk):
        lst0, lst1, mcaslice = idx
        for v, j0, j1 in zip(chunk, lst0, lst1):
            self._data[j0, j1, mcaslice] = v


class ChunkedMcaStack(SlicedMcaStack):

    def __init__(self, data, nmca=None, **kwargs):
        """
        3D stack (n0, n1, n2) to be iterated over in nmca spectra

        :param array data: numpy array or h5py dataset
        :param num nmca: number of spectra in one chunk
        :param \**kwargs: see SlicedMcaStack
        """
        # Outer loop (non-chunked dimension)
        n0, n1, n2 = data.shape
        if nmca is None:
            nmca = min(n0, n1)
        if abs(n0-nmca) < abs(n1-nmca):
            self._loopout = list(range(n1))
            self._axes = 1,0
            n = n0
        else:
            self._loopout = list(range(n0))
            self._axes = 0,1
            n = n1
        
        # Inner loop (chunked dimension)
        nbuffer = min(nmca, n)
        nchunks = n//nbuffer + int(bool(n % nbuffer))
        nbuffer = n//nchunks + int(bool(n % nchunks))
        self._loopin = list(chunk_indices(0, n, nbuffer))

        super(ChunkedMcaStack, self).__init__(data, nbuffer=nbuffer, **kwargs)

    def items(self, copy=True):
        _, yields_copy, post_copy = self._prepare_access(copy=copy)
        idx_data = [slice(None), slice(None), self._mcaslice]
        i,j = self._axes
        buffer = self._buffer
        for idxout in self._loopout:
            idx_data[i] = idxout
            for idxin, n in self._loopin:
                idx_data[j] = idxin
                idx = tuple(idx_data)
                if yields_copy:
                    chunk = buffer[:n, :]
                    chunk[()] = self._data[idx]
                else:
                    chunk = self._data[idx]
                yield idx, chunk
                if post_copy:
                    self._data[idx] = chunk


class FastFitOutputBuffer(object):

    def __init__(self, outputDir=None, outputRoot=None, fileEntry=None,
                 fileProcess=None, tif=False, edf=False, csv=False, h5=True,
                 overwrite=False, save_residuals=False, save_fit=False, save_data=False):
        """
        Fast fitting output buffer, to be saved as:
         .h5 : outputDir/outputRoot.h5::/fileEntry/fileProcess
         .edf/.csv/.tif: outputDir/outputRoot/fileEntry.ext

        Usage with context:
            outbuffer = FastFitOutputBuffer(...)
            with outbuffer.save_context():
                ...

        Usage without context:
            outbuffer = FastFitOutputBuffer(...)
            ...
            outbuffer.save()

        :param str outputDir:
        :param str outputRoot:
        :param str fileEntry:
        :param str fileProcess:
        :param save_residuals:
-       :param save_fit:
-       :param save_data:
        :param bool tif:
        :param bool edf:
        :param bool csv:
        :param bool h5:
        :param bool overwrite:
        """
        self._init_buffer = False
        self._output = {}
        self._nxprocess = None

        self.outputDir = outputDir
        self.outputRoot = outputRoot
        self.fileEntry = fileEntry
        self.fileProcess = fileProcess
        self.tif = tif
        self.edf = edf
        self.csv = csv
        self.h5 = h5
        self.save_residuals = save_residuals
        self.save_fit = save_fit
        self.save_data = save_data
        self.overwrite = overwrite

    @property
    def outputRoot(self):
        return self._outputRoot
    
    @outputRoot.setter
    def outputRoot(self, value):
        self._check_buffer_context()
        if value:
            self._outputRoot = value
        else:
            self._outputRoot = 'IMAGES'

    @property
    def fileEntry(self):
        return self._fileEntry
    
    @fileEntry.setter
    def fileEntry(self, value):
        self._check_buffer_context()
        if value:
            self._fileEntry = value
        else:
            self._fileEntry = 'images'

    @property
    def fileProcess(self):
        return self._fileProcess
    
    @fileProcess.setter
    def fileProcess(self, value):
        self._check_buffer_context()
        if value:
            self._fileProcess = value
        else:
            self._fileProcess = self.fileEntry

    @property
    def edf(self):
        return self._edf
    
    @edf.setter
    def edf(self, value):
        self._check_buffer_context()
        self._edf = value

    @property
    def tif(self):
        return self._tif
    
    @tif.setter
    def tif(self, value):
        self._check_buffer_context()
        self._tif = value

    @property
    def csv(self):
        return self._csv
    
    @csv.setter
    def csv(self, value):
        self._check_buffer_context()
        self._csv = value

    @property
    def cfg(self):
        return self.csv or self.edf or self.tif

    @property
    def save_data(self):
        return self._save_data and self.h5
    
    @save_data.setter
    def save_data(self, value):
        self._check_buffer_context()
        self._save_data = value

    @property
    def save_fit(self):
        return self._save_fit and self.h5
    
    @save_fit.setter
    def save_fit(self, value):
        self._check_buffer_context()
        self._save_fit= value

    @property
    def save_residuals(self):
        return self._save_residuals and self.h5
    
    @save_residuals.setter
    def save_residuals(self, value):
        self._check_buffer_context()
        self._save_residuals = value

    @property
    def save_diagnostics(self):
        return self.save_residuals or self.save_fit

    @property
    def overwrite(self):
        return self._overwrite
    
    @overwrite.setter
    def overwrite(self, value):
        self._check_buffer_context()
        self._overwrite = value

    def _check_buffer_context(self):
        if self._init_buffer:
            raise RuntimeError('Buffer is locked')

    @property
    def outroot_localfs(self):
        return os.path.join(self.outputDir, self.outputRoot)

    def filename(self, ext, suffix=None):
        if not suffix:
            suffix = ""
        if ext == '.h5':
            return os.path.join(self.outputDir, self.outputRoot+suffix+ext)
        else:
            return os.path.join(self.outroot_localfs, self.fileEntry+suffix+ext)

    def __getitem__(self, key):
        return self._output[key]

    def __setitem__(self, key, value):
        self._output[key] = value

    def __contains__(self, key):
        return key in self._output

    def get(self, key, default=None):
        return self._output.get(key, default)

    def allocate_memory(self, name, fill_value=None, shape=None, dtype=None):
        """
        :param str name:
        :param num fill_value:
        :param tuple shape:
        :param dtype:
        """
        if fill_value is None:
            buffer = numpy.empty(shape, dtype=dtype)
        elif fill_value == 0:
            buffer = numpy.zeros(shape, dtype=dtype)
        else:
            buffer = numpy.full(shape, fill_value, dtype=dtype)
        self._output[name] = buffer
        return buffer

    def allocate_h5(self, name, nxdata=None, fill_value=None, **kwargs):
        """
        :param str name:
        :param str nxdata:
        :param num fill_value:
        :param \**kwargs: see h5py.Group.create_dataset
        """
        parent = self._nxprocess['results']
        if nxdata:
            parent = NexusUtils.nxdata(parent, nxdata)
        buffer = parent.create_dataset(name, **kwargs)
        if fill_value is not None:
            buffer[()] = fill_value
        self.flush()
        self._output[name] = buffer
        return buffer

    @contextmanager
    def _buffer_context(self, update=True):
        """
        Prepare output buffers (HDF5: create file, NXentry and NXprocess)

        :param bool update: True: update existing NXprocess,  False: overwrite or raise an exception
        :raises RuntimeError: NXprocess exists and overwrite==False
        """
        if self._init_buffer:
            yield
        else:
            self._init_buffer = True
            _logger.debug('Output buffer hold ...')
            try:
                if self.h5:
                    if self._nxprocess is None and self.outputDir:
                        cleanup_funcs = []
                        try:
                            with self._h5_context(cleanup_funcs, update=update):
                                yield
                        except:
                            # clean-up and re-raise
                            for func in cleanup_funcs:
                                func()
                            raise
                    else:
                        yield
                else:
                    yield
            finally:
                self._init_buffer = False
                _logger.debug('Output buffer released')

    @contextmanager
    def _h5_context(self, cleanup_funcs, update=True):
        fileName = self.filename('.h5')
        existed = [False]*3
        existed[0] = os.path.exists(fileName)
        with NexusUtils.nxroot(fileName, mode='a') as f:
            entryname = self.fileEntry
            existed[1] = entryname in f
            entry = NexusUtils.nxentry(f, entryname)
            procname = self.fileProcess
            if procname in entry:
                existed[2] = True
                path = entry[procname].name
                if update:
                    pass
                elif self.overwrite:
                    _logger.warning('overwriting {}'.format(path))
                    del entry[procname]
                    existed[2] = False
                else:
                    raise RuntimeError('{} already exists'.format(path))
            self._nxprocess = NexusUtils.nxprocess(entry, procname)
            try:
                yield
            except:
                # clean-up and re-raise
                if not existed[0]:
                    cleanup_funcs.append(lambda: os.remove(fileName))
                elif not existed[1]:
                    del f[entryname]
                elif not existed[2]:
                    del entry[procname]
                raise
            finally:
                self._nxprocess = None

    @contextmanager
    def save_context(self):
        with self._buffer_context(update=False):
            try:
                yield
            except:
                raise
            else:
                self.save()

    def flush(self):
        if self._nxprocess is not None :
            self._nxprocess.file.flush()

    def save(self):
        """
        Save result of Fast XRF fitting. Preferrable use save_context instead.
        HDF5 NXprocess will be updated, not overwritten.
        """
        if not (self.tif or self.edf or self.csv or self.h5):
            _logger.warning('fit result not saved (no output format specified)')
            return
        t0 = time.time()
        _logger.debug('Saving results ...')

        with self._buffer_context(update=True):
            if self.tif or self.edf or self.csv:
                self._save_single()
            if self.h5:
                self._save_h5()

        t = time.time() - t0
        _logger.debug("Saving results elapsed = %f", t)

    @property
    def parameter_names(self):
        return self._get_names('parameter_names', '{}')

    @property
    def uncertainties_names(self):
        return self._get_names('parameter_names', 's({})')

    @property
    def concentration_names(self):
        return self._get_names('concentration_names', 'C({}){}')

    def _get_names(self, names, fmt):
        labels = self.get(names, None)
        if not labels:
            return []
        out = []
        for label in labels:
            label = label.split('-')
            name = label[0].replace(" ", "-")
            if len(label)>1:
                layer = '-'.join(label[1:])
            else:
                layer = ''
            label = fmt.format(name, layer)
            out.append(label)
        return out

    def _save_single(self):
        from PyMca5.PyMca import ArraySave

        imageNames = []
        imageList = []
        lst = [('parameter_names', 'parameters', '{}'),
               ('parameter_names', 'uncertainties', 's({})'),
               ('concentration_names', 'concentrations', 'C({}){}')]
        for names, key, fmt in lst:
            images = self.get(key, None)
            if images is not None:
                for img in images:
                    imageList.append(img)
                imageNames += self._get_names(names, fmt)
        NexusUtils.mkdir(self.outroot_localfs)
        if self.edf:
            fileName = self.filename('.edf')
            self._check_overwrite(fileName)
            ArraySave.save2DArrayListAsEDF(imageList, fileName,
                                           labels=imageNames)
        if self.csv:
            fileName = self.filename('.csv')
            self._check_overwrite(fileName)
            ArraySave.save2DArrayListAsASCII(imageList, fileName, csv=True,
                                             labels=imageNames)
        if self.tif:
            for label,image in zip(imageNames, imageList):
                if label.startswith("s("):
                    suffix = "_s" + label[2:-1]
                elif label.startswith("C("):
                    suffix = "_w" + label[2:-1]
                else:
                    suffix  = "_" + label
                fileName = self.filename('.tif', suffix=suffix)
                self._check_overwrite(fileName)
                ArraySave.save2DArrayListAsMonochromaticTiff([image],
                                                             fileName,
                                                             labels=[label],
                                                             dtype=numpy.float32)
        if self.cfg and 'configuration' in self:
            fileName = self.filename('.cfg')
            self._check_overwrite(fileName)
            self['configuration'].write(fileName)

    def _check_overwrite(self, fileName):
        if os.path.exists(fileName):
            if self.overwrite:
                _logger.warning('overwriting {}'.format(fileName))
            else:
                raise RuntimeError('{} already exists'.format(fileName))

    def _save_h5(self):
        # Save fit configuration
        nxprocess = self._nxprocess
        nxresults = nxprocess['results']
        configdict = self.get('configuration', None)
        NexusUtils.nxprocess_configuration_init(nxprocess, configdict=configdict)

        # Save fitted parameters, uncertainties and elemental concentrations
        mill = numpy.float32(1e6)
        lst = [('parameter_names', 'uncertainties', lambda x:x, None),
               ('parameter_names', 'parameters', lambda x:x, None),
               ('concentration_names', 'concentrations', lambda x:x*mill, 'ug/g')]
        for names, key, proc, units in lst:
            images = self.get(key, None)
            if images is not None:
                attrs = {'interpretation':'image'}
                if units:
                    attrs = {'units': units}
                signals = [(label, {'data':proc(img), 'chunks':True}, attrs)
                           for label,img in zip(self[names], images)]
                data = NexusUtils.nxdata(nxresults, key)
                NexusUtils.nxdata_add_signals(data, signals)
                NexusUtils.mark_default(data)

        # Save fitted model and residuals
        signals = []
        attrs = {'interpretation':'spectrum'}
        for name in ['data', 'model', 'residuals']:
            dset = self.get(name, None)
            if dset is not None:
                signals.append((name,None,attrs))
        if signals:
            nxdata = NexusUtils.nxdata(nxresults, 'fit')
            NexusUtils.nxdata_add_signals(nxdata, signals)
            axes = self.get('data_axes', None)
            axes_used = self.get('data_axes_used', None)
            if axes:
                NexusUtils.nxdata_add_axes(nxdata, axes)
            if axes_used:
                axes = [(ax, None, None) for ax in axes_used]
                NexusUtils.nxdata_add_axes(nxdata, axes, append=False)

        # Save diagnostics
        signals = []
        attrs = {'interpretation':'image'}
        for name in ['nObservations', 'nFreeParameters']:
            img = self.get(name, None)
            if img is not None:
                signals.append((name,img,attrs))
        if signals:
            nxdata = NexusUtils.nxdata(nxresults, 'diagnostics')
            NexusUtils.nxdata_add_signals(nxdata, signals)


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
                   'diagnostics=', 'debug=']
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
    save_fit = 0
    save_residuals = 0
    save_data = 0
    debug = 0
    overwrite = 1
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
        elif opt == '--diagnostics':
            save_fit = int(arg)
            save_residuals = save_fit
            save_data = save_fit
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
        elif opt == '--debug':
            debug = int(arg)
        elif opt == '--overwrite':
            overwrite = int(arg)

    if debug:
        _logger.setLevel(logging.DEBUG)
    else:
        _logger.setLevel(logging.INFO)
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

    outbuffer = FastFitOutputBuffer(outputDir=outputDir,
                        outputRoot=outputRoot, fileEntry=fileEntry,
                        fileProcess=fileProcess, save_data=save_data,
                        save_fit=save_fit, save_residuals=save_residuals,
                        tif=tif, edf=edf, csv=csv, h5=h5, overwrite=overwrite)

    from PyMca5.PyMcaMisc import ProfilingUtils
    with ProfilingUtils.profile(memory=debug, time=debug):
        with outbuffer.save_context():
            outbuffer = fastFit.fitMultipleSpectra(y=dataStack,
                                                weight=weight,
                                                refit=refit,
                                                concentrations=concentrations,
                                                outbuffer=outbuffer)
            print("Total Elapsed = % s " % (time.time() - t0))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()