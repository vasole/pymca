#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2022 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF.
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
__author__ = "V.A. Sole - ESRF"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "26/10/2022"

import sys
import os
import struct
import numpy
import logging

_logger = logging.getLogger(__name__)

try:
    from bcflight import bruker
    _logger.info("Bruker BCF file supported")
    HAS_BCF_SUPPORT = True
except:
    _logger.info("Bruker BCF file support not available")
    HAS_BCF_SUPPORT = False

try:
    from PyMca5.PyMcaCore.Dataobject import DataObject
except ImportError:
    _logger.info("Using built-in container")
    class DataObject:
        def __init__(self):
            self.info = {}
            self.data = numpy.array([])

SOURCE_TYPE = "EdfFileStack"

class BrukerBCF(DataObject):
    def __init__(self, filename):
        if not isBrukerBCFFile(filename):
            raise IOError("Filename %s does not seem to be a Bruker bcf file")
        DataObject.__init__(self)
        reader = bruker.BCF_reader(filename)
        self.data = None
        self._sampling = 1
        try:
            self.data = reader.parse_hypermap(downsample=self._sampling,
                                              lazy=False)
        except:
            if "MemoryError" in "%s" % (sys.exc_info()[0],):
                self._sampling += 1
                self.data = reader.parse_hypermap(downsample=self._sampling,
                                                  lazy=False)
                _logger.warning("Data downsampled to fit into memory")
            else:
                raise
        self.sourceName = filename
        self.info = {}
        self.info["SourceType"] = SOURCE_TYPE
        self.info["SourceName"] = self.sourceName
        shape = self.data.shape
        for i in range(len(shape)):
            key = 'Dim_%d' % (i + 1,)
            self.info[key] = shape[i]
        self.info["NumberOfFiles"] = 1
        self.info["McaIndex"] = -1
        self.info["McaCalib"] = [0.0, 1.0, 0.0]
        self.info["Channel0"] = 0.0

        # header information
        header_file = reader.get_file('EDSDatabase/HeaderData')
        header_byte_str = header_file.get_as_BytesIO_string().getvalue()
        hd_bt_str = bruker.fix_dec_patterns.sub(b'\\1.\\2', header_byte_str)
        xml_str = hd_bt_str

        root = bruker.ET.fromstring(xml_str)
        root = root.find("./ClassInstance[@Type='TRTSpectrumDatabase']")
        xScale, yScale = get_scales(root)
        self.info["xScale"] = xScale
        self.info["xScale"][1] = xScale[1] * self._sampling
        self.info["yScale"] = yScale
        self.info["yScale"][1] = yScale[1] * self._sampling
        self.info["McaCalib"] = get_calibration(root)

def get_scales(root):
    semData = root.find("./ClassInstance[@Type='TRTSEMData']")
    semData_dict = bruker.dictionarize(semData)
    if "DX" in semData_dict and "DY" in semData_dict:
        xScale = [0.0, semData_dict["DX"]]
        yScale = [0.0, semData_dict["DY"]]
    else:
        xScale = None
        yScale = None
    return xScale, yScale

def get_calibration(root):
    spectrum_header = root.find(".//ClassInstance[@Type='TRTSpectrumHeader']")
    if spectrum_header:
        spectrum_header_data = bruker.dictionarize(spectrum_header)
        calculate = True
        for key in ["ChannelCount", "CalibAbs", "CalibLin"]:
            if key not in spectrum_header_data:
                calculate = False
                break
        if calculate:
            return [spectrum_header_data["CalibAbs"],
                    spectrum_header_data["CalibLin"],
                    0.0]
    return [0.0, 1.0, 0.0]

def isBrukerBCFFile(filename):
    _logger.info("BrukerBCF.isBrukerBCFFile called %s" % filename)
    result = False
    try:
        owner = False
        if not hasattr(filename, "seek"):
            fid = open(filename, mode='rb')
            owner = True
        else:
            fid =filename
        fid.seek(0)
        eight_chars = fid.read(8)
        if hasattr(eight_chars, "decode"):
            if eight_chars == b"AAMVHFSS":
                if owner:
                    fid.close()
                _logger.info("It is a Bruker bcf file")
                result = True
        else:
            if eight_chars == "AAMVHFSS":
                if owner:
                    fid.close()
                _logger.info("It is a Bruker bcf file")
                result = True
    except:
        if owner:
            fid.close()
    if result:
        _logger.info("It is a Bruker bcf file")
    else:
        _logger.info("Not a Bruker bcf file")
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        print("Usage: ")
        print("python BrukerBCF.py filename")
        sys.exit(0)
    print("is Bruker BCF File?", isBrukerBCFFile(filename))
    stack = BrukerBCF(filename)
    print(stack.data)
    print(stack.info)

