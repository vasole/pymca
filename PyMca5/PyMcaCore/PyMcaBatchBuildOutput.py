#!/usr/bin/env python
#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
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
import os
import sys
import numpy
import logging
import shutil
import re
import itertools
from PyMca5.PyMcaIO import EdfFile
from PyMca5.PyMcaIO import TiffIO
from PyMca5.PyMcaCore import NexusTools
try:
    import h5py
except ImportError:
    h5py = None

_logger = logging.getLogger(__name__)


class PyMcaBatchBuildOutput(object):
    def __init__(self, inputdir=None, outputdir=None):
        self.inputDir = inputdir
        self.outputDir = outputdir

    def buildOutput(self, inputdir=None, basename=None, outputdir=None, delete=None):
        """
        :returns: 3 lists of merged filenames: .edf filenames, .dat filenames and .h5 filenames
        """
        if inputdir is None:
            inputdir = self.inputDir
        if inputdir is None:
            inputdir = os.getcwd()
        if not os.path.isdir(inputdir):
            return [], [], []
        if outputdir is None:
            outputdir = self.outputDir
        if outputdir is None:
            outputdir = inputdir
        if delete is None:
            if outputdir == inputdir:
                delete = True
        _logger.debug("delete option = %s", delete)
        allfiles = os.listdir(inputdir)
        partialList = {'edf': {'ext': '.edf', 'list': []},
                       'tif': {'ext': '.tif', 'list': []},
                       'dat': {'ext': '.dat', 'list': []},
                       'csv': {'ext': '.csv', 'list': []},
                       'h5': {'ext': '.h5', 'list': []},
                       'cfg': {'ext': '.cfg', 'list': []},
                       'conc': {'ext': '_concentrations.txt', 'list': []}
                       }
        for filepath in allfiles:
            filename = os.path.basename(filepath)
            for typ, value in partialList.items():
                if basename:
                    if not filename.startswith(basename):
                        continue
                if filename.endswith('000000_partial' + value['ext']):
                    value['list'].append(filename)
        outListH5 = self._merge(inputdir, outputdir, delete,
                                partialList['h5']['list'], self._mergeH5)
        outListEdf = self._merge(inputdir, outputdir, delete,
                                 partialList['edf']['list'], self._mergeEdf)
        outListDat = self._merge(inputdir, outputdir, delete,
                                 partialList['dat']['list'], self._mergeDat)
        self._merge(inputdir, outputdir, delete,
                    partialList['tif']['list'], self._mergeTif)
        self._merge(inputdir, outputdir, delete,
                    partialList['csv']['list'], self._mergeCsv)
        self._merge(inputdir, outputdir, delete,
                    partialList['conc']['list'], self._mergeConcTxt)
        self._merge(inputdir, outputdir, delete,
                    partialList['cfg']['list'], self._mergeCfg)
        return outListEdf, outListDat, outListH5

    def _merge(self, inputdir, outputdir, delete, partialList, func):
        """
        The images to be merged already have the final size but are filled with NaN's
        """
        outList = []
        for filename in partialList:
            parts = self.getPartialFileList(os.path.join(inputdir, filename))
            outfilename = parts[0].replace("_000000_partial", "")
            _logger.debug("Merging %s (%d parts)", outfilename, len(parts))
            outfilename = os.path.join(outputdir, outfilename)
            try:
                func(parts, outfilename)
            except:
                _logger.error("Error merging %s\n: %s", outfilename, sys.exc_info()[1])
                continue
            outList.append(outfilename)
            if delete:
                for filename in parts:
                    try:
                        os.remove(filename)
                    except:
                        _logger.warning("Cannot delete file %s" % filename)
        return outList

    def _mergeH5(self, parts, outfilename):
        shutil.copy(parts[0], outfilename)
        with h5py.File(outfilename, mode='a') as fout:
            for entry in NexusTools.getNXClassGroups(fout, '/', [u'NXentry']):
                for process in NexusTools.getNXClassGroups(fout, entry.name, [u'NXprocess']):
                    for results in NexusTools.getNXClassGroups(fout, process.name, [u'NXcollection']):
                        for dataout in NexusTools.getNXClassGroups(fout, results.name, [u'NXdata']):
                            for part in parts[1:]:
                                with h5py.File(part, mode='r') as fin:
                                    try:
                                        datain = fin[dataout.name]
                                    except KeyError:
                                        _logger.error('%s does not have %s', part, repr(dataout.name))
                                        continue
                                    for datasetname in dataout:
                                            self._fillPartial(dataout[datasetname],
                                                              datain[datasetname],
                                                              maxdims=2)

    def _mergeCfg(self, parts, outfilename):
        # They should be all the same so pick the first one
        shutil.copy(parts[0], outfilename)
        return outfilename

    def _mergeEdf(self, parts, outfilename):
        for i, edfname in enumerate(parts):
            edf = EdfFile.EdfFile(edfname, access='rb', fastedf=0)
            nImages = edf.GetNumImages()
            if i == 0:
                images = [edf.GetData(j).copy() for j in range(nImages)]
                headers = [{'Title': edf.GetHeader(j)['Title']}
                           for j in range(nImages)]
            else:
                headersi = [{'Title': edf.GetHeader(j)['Title']}
                            for j in range(nImages)]
                for header, img in zip(headers, images):
                    k = headersi.index(header)
                    self._fillPartial(img, edf.GetData(k))
            del edf
        if os.path.exists(outfilename):
            _logger.debug("Output file already exists, trying to delete it")
            os.remove(outfilename)
        edfout = EdfFile.EdfFile(outfilename, access="ab")
        for i, (img, header) in enumerate(zip(images, headers)):
            edfout.WriteImage(header, img, Append=i > 0)
        del edfout

    def _mergeTif(self, parts, outfilename):
        for i, tifname in enumerate(parts):
            tif = TiffIO.TiffIO(tifname, mode='rb')
            nImages = tif.getNumberOfImages()
            if i == 0:
                images = [tif.getData(j).copy() for j in range(nImages)]
                headers = [{'Title': tif.getInfo(j)['info']['Title']}
                           for j in range(nImages)]
            else:
                headersi = [{'Title': tif.getInfo(j)['info']['Title']}
                            for j in range(nImages)]
                for header, img in zip(headers, images):
                    k = headersi.index(header)
                    self._fillPartial(img, tif.getData(k))
            del tif
        if os.path.exists(outfilename):
            _logger.debug("Output file already exists, trying to delete it")
            os.remove(outfilename)
        for i, (img, header) in enumerate(zip(images, headers)):
            # TODO: there must be a better way
            if i == 0:
                tifout = TiffIO.TiffIO(outfilename, mode="wb+")
            elif i == 1:
                del tifout
                tifout = TiffIO.TiffIO(outfilename, mode="rb+")
            tifout.writeImage(img, info=header)
        del tifout

    def _fillPartial(self, output, input, maxdims=None):
        if output.shape != input.shape:
            _logger.error("Cannot merge array's with different shapes")
            return
        if maxdims is None:
            maxdims = output.ndim
        if output.ndim > maxdims:
            # This is meant to preserve memory when copying
            # h5py datasets
            idx = [slice(None)]*output.ndim
            shape = output.shape
            iterdims = sorted(numpy.argsort(shape)[:-maxdims])
            iterlst = [list(range(shape[i])) for i in iterdims]
            for iteridx in itertools.product(*iterlst):
                for axis, i in zip(iterdims, iteridx):
                    idx[axis] = i
                idxtpl = tuple(idx)
                bufferin = input[idxtpl]
                mask = ~numpy.isnan(bufferin)
                if mask.any():
                    bufferout = output[idxtpl]
                    bufferout[mask] = bufferin[mask]
                    output[idxtpl] = bufferout
        else:
            mask = ~numpy.isnan(input)
            if mask.any():
                output[mask] = input[mask]

    def _mergeDat(self, parts, outfilename):
        self._mergeAscii(parts, outfilename, '  ')
    
    def _mergeCsv(self, parts, outfilename):
        self._mergeAscii(parts, outfilename, ';')

    def _mergeAscii(self, parts, outfilename, separator):
        first = True
        for specname in parts:
            f = open(specname)
            lines = f.readlines()
            f.close()
            j = 1
            while not len(lines[-j].replace("\n", "")):
                j += 1
            if first:
                first = False
                labels = lines[0].replace("\n", "").split(separator)
                nlabels = len(labels)
                nrows = len(lines) - j
                data = numpy.zeros((nrows, nlabels), numpy.double)
                inputdata = numpy.zeros((nrows, nlabels), numpy.double)
                colSelect = list(range(nlabels))
            else:
                labelsi = lines[0].replace("\n", "").split(separator)
                colSelect = [labels.index(label) for label in labelsi]
            for i in range(nrows):
                inputdata[i, colSelect] = [float(x) for x in lines[i+1].split(separator)]
            self._fillPartial(data, inputdata)
        if os.path.exists(outfilename):
            os.remove(outfilename)
        outfile = open(outfilename, 'w+')
        outfile.write("%s\n" % separator.join(labels))
        for row in range(nrows):
            line = ""
            for col in range(nlabels):
                if col == 0:
                    line += "%d" % inputdata[row, col]
                elif col == 1:
                    line += separator + "%d" % inputdata[row, col]
                else:
                    line += separator + "%g" % data[row, col]
            outfile.write("%s\n" % line)
        outfile.write("\n")
        outfile.close()

    def _mergeConcTxt(self, parts, outfilename):
        for i, infilename in enumerate(parts):
            ffile = open(infilename, 'rb')
            if i == 0:
                if os.path.exists(outfilename):
                    os.remove(outfilename)
                outfile = open(outfilename, 'wb')
            lines = ffile.readlines()
            for line in lines:
                outfile.write(line)
            ffile.close()
        outfile.close()

    @staticmethod
    def getPartialFileList(filename, begin=None, end=None, skip=None):
        # Decempose filename, for example "/tmp/base_000000_partial.ext"
        name, ext = os.path.splitext(os.path.basename(filename))
        m = re.search(r"^(.+?)(\d+)([^\d]+)$", name)
        if not m:
            return [filename]
        prefix, number, suffix = m.groups()
        prefix = os.path.join(os.path.dirname(filename), prefix)
        suffix += ext
        # Prepare iteration over "/tmp/base_{:d}_partial.ext"
        fformat = prefix + "{{:0{}d}}".format(len(number)) + suffix
        if begin is None:
            i = 0
            while not os.path.exists(fformat.format(i)):
                i += 1
        else:
            i = begin
        if not skip:
            skip = []
        # Find all "/tmp/base_{:d}_partial.ext"
        filelist = []
        while os.path.exists(fformat.format(i)) or i in skip:
            if i not in skip:
                filelist.append(fformat.format(i))
            i += 1
            if end is not None:
                if i > end:
                    break
        return filelist


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("python PyMcaBatchBuildOutput.py directory")
        sys.exit(0)
    directory = sys.argv[1]
    w = PyMcaBatchBuildOutput(directory)
    w.buildOutput()
    """
    allfiles = os.listdir(directory)
    edflist = []
    datlist = []
    for filename in allfiles:
        if filename.endswith('000000_partial.edf'):edflist.append(filename)
        elif filename.endswith('000000_partial.dat'):datlist.append(filename)
    for filename in edflist:
        print w.getPartialFileList(os.path.join(directory, filename))
    """
