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
from contextlib import contextmanager
from PyMca5.PyMcaIO import NexusUtils

_logger = logging.getLogger(__name__)


class OutputBuffer(object):

    def __init__(self, outputDir=None, outputRoot=None, fileEntry=None,
                 fileProcess=None, tif=False, edf=False, csv=False, h5=True,
                 overwrite=False, save_residuals=False, save_fit=False, save_data=False):
        """
        Fast fitting output buffer, to be saved as:
         .h5 : outputDir/outputRoot.h5::/fileEntry/fileProcess
         .edf/.csv/.tif: outputDir/outputRoot/fileEntry.ext

        Usage with context:
            outbuffer = OutputBuffer(...)
            with outbuffer.save_context():
                ...

        Usage without context:
            outbuffer = OutputBuffer(...)
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
        if self._nxprocess is not None:
            self._nxprocess.file.flush()

    def save(self):
        """
        Save result of Fast XRF fitting. Preferrable use save_context instead.
        HDF5 NXprocess will be updated, not overwritten.
        """
        if not (self.tif or self.edf or self.csv or self.h5):
            _logger.warning('fit result not saved (no output format specified)')
            return
        if not self.outputDir:
            _logger.warning('fit result not saved (no output directory specified)')
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
    def massfraction_names(self):
        return self._get_names('massfraction_names', 'w({}){}')

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
               ('massfraction_names', 'massfractions', 'w({}){}')]
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
                elif label.startswith("w("):
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
        if nxprocess is None:
            return
        nxresults = nxprocess['results']
        configdict = self.get('configuration', None)
        NexusUtils.nxprocess_configuration_init(nxprocess, configdict=configdict)

        # Save fitted parameters, uncertainties and elemental massfractions
        mill = numpy.float32(1e6)
        lst = [('parameter_names', 'uncertainties'),
               ('parameter_names', 'parameters'),
               ('massfraction_names', 'massfractions')]
        for names, key in lst:
            images = self.get(key, None)
            if images is not None:
                attrs = {'interpretation': 'image'}
                signals = [(label, {'data': img, 'chunks': True}, attrs)
                           for label, img in zip(self[names], images)]
                data = NexusUtils.nxdata(nxresults, key)
                NexusUtils.nxdata_add_signals(data, signals)
                NexusUtils.mark_default(data)
        if 'parameters' in nxresults and 'uncertainties' in nxresults:
            NexusUtils.nxdata_add_errors(nxresults['parameters'], nxresults['uncertainties'])

        # Save fitted model and residuals
        signals = []
        attrs = {'interpretation':'spectrum'}
        for name in ['data', 'model', 'residuals']:
            dset = self.get(name, None)
            if dset is not None:
                signals.append((name, None, attrs))
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