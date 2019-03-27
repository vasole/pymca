#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2019 European Synchrotron Radiation Facility
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
__author__ = "Wout De Nolf"
__contact__ = "wout.de_nolf@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import os
import numpy
import logging
import time
import re
from six import string_types
from contextlib import contextmanager
from PyMca5.PyMcaIO import NexusUtils

_logger = logging.getLogger(__name__)


class OutputBuffer(object):

    def __init__(self, outputDir=None, 
                 outputRoot=None, fileEntry=None, fileProcess=None, 
                 saveResiduals=False, saveFit=False, saveData=False,
                 tif=False, edf=False, csv=False, dat=False, h5=True,
                 overwrite=False, suffix=None):
        """
        XRf batch fitting output buffer, to be saved as:
         .h5 : outputDir/outputRoot+suffix.h5::/fileEntry/fileProcess
         .edf/.csv/.tif: outputDir/outputRoot/fileEntry+suffix.ext

        Usage with context:
            outbuffer = OutputBuffer(...)
            with outbuffer.saveContext():
                ...

        Usage without context:
            outbuffer = OutputBuffer(...)
            ...
            outbuffer.save()

        :param str outputDir: default: current working directory
        :param str outputRoot: default: "IMAGES"
        :param str fileEntry: default: "images"
        :param str fileProcess: default: "xrf_fit"
        :param saveResiduals:
-       :param saveFit:
-       :param saveData:
        :param bool tif:
        :param bool edf:
        :param bool csv:
        :param bool dat:
        :param bool h5:
        :param bool overwrite:
        """
        self._init_buffer = False
        self._output = {}
        self._attrs = {}
        self._nxprocess = None

        self.outputDir = outputDir
        self.outputRoot = outputRoot
        self.fileEntry = fileEntry
        self.fileProcess = fileProcess
        self.tif = tif
        self.edf = edf
        self.csv = csv
        self.dat = dat
        self.h5 = h5
        self.saveResiduals = saveResiduals
        self.saveFit = saveFit
        self.saveData = saveData
        self.overwrite = overwrite
        self.suffix = suffix

    @property
    def outputRoot(self):
        return self._outputRoot
    
    @outputRoot.setter
    def outputRoot(self, value):
        self._check_bufferContext()
        if value:
            self._outputRoot = value
        else:
            self._outputRoot = 'IMAGES'

    @property
    def fileEntry(self):
        return self._fileEntry
    
    @fileEntry.setter
    def fileEntry(self, value):
        self._check_bufferContext()
        if value:
            self._fileEntry = value
        else:
            self._fileEntry = 'images'

    @property
    def fileProcess(self):
        return self._fileProcess
    
    @fileProcess.setter
    def fileProcess(self, value):
        self._check_bufferContext()
        if value:
            self._fileProcess = value
        else:
            self._fileProcess = 'xrf_fit'

    @property
    def extensions(self):
        lst = []
        if self.h5:
            lst.append('.h5')
        if self.dat:
            lst.append('.dat')
        if self.csv:
            lst.append('.csv')
        if self.tif:
            lst.append('.tif')
        if self.edf:
            lst.append('.edf')
        return lst

    @extensions.setter
    def extensions(self, lst):
        for ext in lst:
            if ext.startswith('.'):
                attr = ext[1:]
            else:
                attr = ext
            if hasattr(self, attr):
                setattr(self, attr, True)

    @property
    def edf(self):
        return self._edf
    
    @edf.setter
    def edf(self, value):
        self._check_bufferContext()
        self._edf = value

    @property
    def tif(self):
        return self._tif
    
    @tif.setter
    def tif(self, value):
        self._check_bufferContext()
        self._tif = value

    @property
    def csv(self):
        return self._csv
    
    @csv.setter
    def csv(self, value):
        self._check_bufferContext()
        self._csv = value

    @property
    def dat(self):
        return self._dat
    
    @dat.setter
    def dat(self, value):
        self._check_bufferContext()
        self._dat = value

    @property
    def cfg(self):
        return self.csv or self.dat or self.edf or self.tif

    @property
    def saveData(self):
        return self._saveData and self.h5
    
    @saveData.setter
    def saveData(self, value):
        self._check_bufferContext()
        self._saveData = value

    @property
    def saveFit(self):
        return self._saveFit and self.h5
    
    @saveFit.setter
    def saveFit(self, value):
        self._check_bufferContext()
        self._saveFit = value

    @property
    def saveResiduals(self):
        return self._saveResiduals and self.h5
    
    @saveResiduals.setter
    def saveResiduals(self, value):
        self._check_bufferContext()
        self._saveResiduals = value

    @property
    def saveDiagnostics(self):
        return self.saveResiduals or self.saveFit

    @property
    def overwrite(self):
        return self._overwrite
    
    @overwrite.setter
    def overwrite(self, value):
        self._check_bufferContext()
        self._overwrite = value

    def _check_bufferContext(self):
        if self._init_buffer:
            raise RuntimeError('Buffer is locked')

    @property
    def outroot_localfs(self):
        return os.path.join(self.outputDir, self.outputRoot)

    def filename(self, ext, suffix=None):
        if not suffix:
            suffix = ""
        if self.suffix:
            suffix += self.suffix
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

    def allocateMemory(self, name, fill_value=None, shape=None, dtype=None, attrs=None):
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
        self._attrs[name] = attrs
        return buffer

    def allocateH5(self, name, nxdata=None, fill_value=None, attrs=None, **kwargs):
        """
        :param str name:
        :param str nxdata:
        :param num fill_value:
        :param \**kwargs: see h5py.Group.create_dataset
        """
        parent = self._nxprocess['results']
        if nxdata:
            parent = NexusUtils.nxData(parent, nxdata)
        buffer = parent.create_dataset(name, **kwargs)
        if attrs:
            buffer.attrs.update(attrs)
        if fill_value is not None:
            buffer[()] = fill_value
        self.flush()
        self._output[name] = buffer
        return buffer

    @contextmanager
    def _bufferContext(self, update=True):
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
                            with self._h5Context(cleanup_funcs, update=update):
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
    def _h5Context(self, cleanup_funcs, update=True):
        """
        Initialize NXprocess on enter and close/cleanup on exit
        """
        fileName = self.filename('.h5')
        existed = [False]*3  # h5file, nxentry, nxprocess
        existed[0] = os.path.exists(fileName)
        with NexusUtils.nxRoot(fileName, mode='a') as f:
            # Open/overwrite NXprocess: h5file::/entry/process
            entryname = self.fileEntry
            existed[1] = entryname in f
            entry = NexusUtils.nxEntry(f, entryname)
            procname = self.fileProcess
            if procname in entry:
                existed[2] = True
                path = entry[procname].name
                if update:
                    _logger.debug('edit {}'.format(path))
                elif self.overwrite:
                    _logger.warning('overwriting {}::{}'.format(fileName, path))
                    del entry[procname]
                    existed[2] = False
                else:
                    raise RuntimeError('{}::{} already exists'.format(fileName, path))
            self._nxprocess = NexusUtils.nxProcess(entry, procname)
            try:
                with self._h5DatasetContext(f):
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
    def _h5DatasetContext(self, f):
        """
        Swap strings for dataset objects on enter and back on exit
        """
        update = {}
        for k, v in self._output.items():
            if isinstance(v, string_types):
                update[k] = f[v]
        self._output.update(update)
        try:
            yield
        finally:
            update = {}
            for k, v in self._output.items():
                if isinstance(v, NexusUtils.h5py.Dataset):
                    update[k] = v.name
            self._output.update(update)

    @contextmanager
    def saveContext(self):
        with self._bufferContext(update=False):
            try:
                yield
            except:
                raise
            else:
                self.save()

    def flush(self):
        if self._nxprocess is not None:
            self._nxprocess.file.flush()

    def save(self):
        """
        Save result of XRF batch fitting. Preferrable use saveContext instead.
        HDF5 NXprocess will be updated, not overwritten.
        """
        if not (self.tif or self.edf or self.csv or self.dat or self.h5):
            _logger.warning('fit result not saved (no output format specified)')
            return
        if not self.outputDir:
            _logger.warning('fit result not saved (no output directory specified)')
            return
        t0 = time.time()
        _logger.debug('Saving results ...')

        with self._bufferContext(update=True):
            if self.tif or self.edf or self.csv or self.dat:
                self._saveSingle()
            if self.h5:
                self._saveH5()

        t = time.time() - t0
        _logger.debug("Saving results elapsed = %f", t)

    @property
    def parameter_names(self):
        return self._getNames('parameter_names', '{}')

    @property
    def uncertainty_names(self):
        return self._getNames('parameter_names', 's({})')

    @property
    def massfraction_names(self):
        return self._getNames('massfraction_names', 'w({}){}', n=2)

    @property
    def molarconcentration_names(self):
        return self._getNames('molarconcentration_names', 'mM({}){}', n=2)

    def _getNames(self, names, fmt, n=1):
        labels = self.get(names, None)
        if not labels:
            return []
        out = []
        for args in labels:
            if not isinstance(args, tuple):
                args = (args,)
            nempty = n-len(args)
            if nempty > 0:
                args = args + ('',)*nempty
            elif nempty < 0:
                raise RuntimeError('{} cannot be represented as {}'
                                   .format(args, repr(fmt)))
            out.append(fmt.format(*args))
        return out

    def _saveSingle(self):
        from PyMca5.PyMca import ArraySave

        imageNames = []
        imageList = []
        lst = [('parameter_names', 'parameters'),
               ('uncertainty_names', 'uncertainties'),
               ('massfraction_names', 'massfractions'),
               ('molarconcentration_names', 'molarconcentrations')]
        for names, key in lst:
            images = self.get(key, None)
            if images is not None:
                for img in images:
                    imageList.append(img)
                imageNames += getattr(self, names)
        NexusUtils.mkdir(self.outroot_localfs)
        if self.edf:
            fileName = self.filename('.edf')
            self._checkOverwrite(fileName)
            ArraySave.save2DArrayListAsEDF(imageList, fileName,
                                           labels=imageNames)
        if self.csv:
            fileName = self.filename('.csv')
            self._checkOverwrite(fileName)
            ArraySave.save2DArrayListAsASCII(imageList, fileName, csv=True,
                                             labels=imageNames)
        if self.dat:
            fileName = self.filename('.dat')
            self._checkOverwrite(fileName)
            ArraySave.save2DArrayListAsASCII(imageList, fileName, csv=False,
                                             labels=imageNames)
        if self.tif:
            for label, image in zip(imageNames, imageList):
                label = label.replace('(', '')
                label = label.replace(')', '')
                label = label.replace(' ', '_')
                fileName = self.filename('.tif', suffix="_" + label)
                self._checkOverwrite(fileName)
                ArraySave.save2DArrayListAsMonochromaticTiff([image],
                                                             fileName,
                                                             labels=[label],
                                                             dtype=numpy.float32)
        if self.cfg and 'configuration' in self:
            fileName = self.filename('.cfg')
            self._checkOverwrite(fileName)
            self['configuration'].write(fileName)

    def _checkOverwrite(self, fileName):
        if os.path.exists(fileName):
            if self.overwrite:
                _logger.warning('overwriting {}'.format(fileName))
            else:
                raise RuntimeError('{} already exists'.format(fileName))

    def _saveH5(self):
        # Save fit configuration
        nxprocess = self._nxprocess
        if nxprocess is None:
            return
        nxresults = nxprocess['results']
        configdict = self.get('configuration', None)
        NexusUtils.nxProcessConfigurationInit(nxprocess, configdict=configdict)

        # Save fitted parameters, uncertainties and elemental concentrations
        lst = [('parameter_names', 'uncertainties'),
               ('parameter_names', 'parameters'),
               ('massfraction_names', 'massfractions'),
               ('molarconcentration_names', 'molarconcentrations')]
        for names, key in lst:
            images = self.get(key, None)
            if images is not None:
                attrs = self._attrs.get(key, {})
                attrs['interpretation'] = 'image'
                signals = []
                for label, img in zip(self[names], images):
                    if isinstance(label, tuple):
                        label = ' '.join(label)
                    signals.append((label, {'data': img, 'chunks': True}, attrs))
                data = NexusUtils.nxData(nxresults, key)
                NexusUtils.nxDataAddSignals(data, signals)
                NexusUtils.markDefault(data)
        if 'parameters' in nxresults and 'uncertainties' in nxresults:
            NexusUtils.nxDataAddErrors(nxresults['parameters'], nxresults['uncertainties'])

        # Save fitted model and residuals
        signals = []
        attrs = {'interpretation': 'spectrum'}
        for name in ['data', 'model', 'residuals']:
            if name in self:
                signals.append((name, self[name], attrs))
        if signals:
            nxdata = NexusUtils.nxData(nxresults, 'fit')
            NexusUtils.nxDataAddSignals(nxdata, signals)
            axes = self.get('dataAxes', None)
            axes_used = self.get('dataAxesUsed', None)
            if axes:
                NexusUtils.nxDataAddAxes(nxdata, axes)
            if axes_used:
                axes = [(ax, None, None) for ax in axes_used]
                NexusUtils.nxDataAddAxes(nxdata, axes, append=False)

        # Save diagnostics
        signals = []
        attrs = {'interpretation': 'image'}
        for name in ['nObservations', 'nFreeParameters', 'Chisq']:
            img = self.get(name, None)
            if img is not None:
                signals.append((name, img, attrs))
        if signals:
            nxdata = NexusUtils.nxData(nxresults, 'diagnostics')
            NexusUtils.nxDataAddSignals(nxdata, signals)
