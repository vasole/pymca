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
import numpy
import os
import struct
from PyMca5.PyMcaCore import DataObject
import sys
SOURCE_TYPE = "EdfFileStack"

class OmdaqLmf(list):
    """
    Table showing the RUNDATA and  ADCINFO
    structure versions associated with each header version

    HV    RUNDATA    ADCINFO
    1       1           1
    2       2           1
    3       3           1
    4       4           2
    5       5           3
    6       6           3
    7       6           4
    8       7           4
    9       8           5
    10      8           6
    11      8           7
    """
    GENERAL_SIZE = 6
    RUNDATA_SIZE = {1:1043,
                    2:1047,
                    3:1055,
                    4:3604} # discrepancy with documentation
    def __init__(self, filelist):
        """
        Parse a list of files into a list of stacks. One for each stack
        The maximum number of stacks is 8.
        An ADC with no hits will give a stack equal to None 
        """
        super(OmdaqLmf, self).__init__()
        for i in range(8):
            self.append(None)
        if type(filelist) not in [type([]), type((1,))]:
            filelist = [filelist]
        for fname in filelist:
            self.parseFile(fname)

    def parseFile(self, fname):
        f = open(fname, "rb")
        d = f.read()
        f.close()
        informationHeader = parseInformationHeader(d)
        if informationHeader["Identifier"] != 66:
            raise IOError("Not an OMDAQ File")
        if informationHeader["ListMode"] != 2:
            raise IOError("Not an list mode file")

        hv = informationHeader["HeaderVersion"]
        adc_offset = self.GENERAL_SIZE + self.RUNDATA_SIZE[hv]
        adc_list = parseAdcInfo(d, hv, offset=adc_offset)

        # the offset to the events is unclear, but we know they
        # are at the end of the file, how they end and the block size
        block_size = informationHeader["ListModeBlockSize"]
        n_blocks = len(d) // block_size
        for i in range(n_blocks):
            block_end = len(d) - i * block_size
            block_start = block_end - block_size
            events = parseLmfBlock(d[block_start:block_end],
                          lmf_version=informationHeader["ListModeVersion"],
                          offset=0)
            n_events = events.shape[0]
            for idx in range(n_events):
                adc, row, col, energy = events[idx]
                #print(adc, energy, row, col)
                nChannels = int(adc_list[adc]["Calibration"][-1])
                if nChannels < 1:
                    continue
                if self[adc] is None:
                    self[adc] = DataObject.DataObject()
                    self[adc].data = numpy.zeros((256, 256, nChannels),
                                                      dtype=numpy.uint32)
                    self[adc].info = {}
                    self[adc].info["SourceType"] = SOURCE_TYPE
                    try:
                        name = adc_list[adc]["Name"]
                        if hasattr(name, "decode"):
                            name = name.decode("utf-8").strip(chr(0))
                        self[adc].info["SourceName"] = name
                    except:
                        self[adc].info["SourceName"] = adc_list[adc]["Name"]
                    self[adc].info["McaCalib"] = [\
                                adc_list[adc]["Calibration"][0],
                                adc_list[adc]["Calibration"][1],
                                0.0]
                    self[adc].info["Channel0"] = 0.0
                    nSpectra = 256 * 256
                    nRows = 256
                    nFiles = nSpectra // nRows
                    self[adc].info["Size"] = nFiles
                    self[adc].info["NumberOfFiles"] = nFiles
                    self[adc].info["FileIndex"] = 0
                if energy >= nChannels:
                    continue
                self[adc].data[row, col, energy] += 1

def parseAdcInfo(block, header_version, offset=0):
    HV_ADC_OFFSETS = {1: 122,
                      2: 122,
                      3: 122,
                      4: 128}
    NMAX_ADC = 8
    adc = [None] * NMAX_ADC
    fmt = "H3f9s"
    size = struct.calcsize(fmt)
    for i in range(NMAX_ADC):
        values = struct.unpack(fmt, block[offset:offset+size])
        live = values[0]
        calibration = values[1:4]
        name = values[4]
        info = {}
        info["Live"] = live
        info["Name"] = name
        info["Calibration"] = calibration
        #print("ADC %d" % i)
        #print("Live = ", live)
        #print("Calibration = ", calibration)
        #print("Name = ", name)
        offset += HV_ADC_OFFSETS[header_version]
        adc[i] = info
        #sys.exit(0)
    return adc

def parseLmfBlock(block, lmf_version=0, offset=0):
    EnergyMask = 0x0fff
    ChannelMask = 0x7000
    if lmf_version < 2:
        fmt = "<BBH"
    else:
        fmt = "<III"
    # size of block header
    block_header_size = 20
    size = struct.calcsize(fmt)
    block_start = offset + block_header_size
    block_end = len(block)
    while block[block_end - 2: block_end] == b'\xff\xff':
        block_end -= size
    offset = block_start
    n_events = (block_end - offset) // size
    events = numpy.zeros((n_events, 4), dtype=numpy.uint16)
    for event in range(n_events):
        row, col, adc_energy = struct.unpack(fmt,
                                             block[offset:offset+size])
        adc = (adc_energy & ChannelMask) >> 12
        energy = (adc_energy & EnergyMask)
        events[event] = adc, row, col, energy
        offset += size
    return events
  
def parseInformationHeader(d):
    """
    Parse the first 6 bytes of the buffer
    """
    offset = 0
    fmt = "B"
    size = struct.calcsize(fmt)
    HeaderVersion = struct.unpack(fmt, d[offset:offset+size])[0]
    offset += size
    #print("Header Version = ", HeaderVersion)
    fmt = "B"
    size = struct.calcsize(fmt)
    value = struct.unpack(fmt, d[offset:offset+size])[0]
    offset += size
    Identifier = value
    #print("Identifier (66) = ", value)
    fmt = "B"
    size = struct.calcsize(fmt)
    value = struct.unpack(fmt, d[offset:offset+size])[0]
    offset += size
    ListMode = value
    #print("List mode file (it has to be 2)  = ", value)
    fmt = "B"
    size = struct.calcsize(fmt)
    value = struct.unpack(fmt, d[offset:offset+size])[0]
    offset += size
    ListModeVersion = value
    #print("LMF version number = ", value)
    fmt = "H"
    size = struct.calcsize(fmt)
    value = struct.unpack(fmt, d[offset:offset+size])[0]
    offset += size
    ListModeBlockSize = value
    #print("List mode block size in bytes  = ", value)
    informationHeader = {}
    informationHeader["HeaderSize"] = offset
    informationHeader["HeaderVersion"] = HeaderVersion
    informationHeader["Identifier"] = Identifier
    informationHeader["ListMode"] = ListMode
    informationHeader["ListModeVersion"] = ListModeVersion
    informationHeader["ListModeBlockSize"] = ListModeBlockSize
    return informationHeader

def isOmdaqLmf(fname):
    try:
        f = open(fname, "rb")
        d = f.read(6)
        f.close()
    except:
        return False
    informationHeader = parseInformationHeader(d)
    #print(informationHeader)
    if informationHeader["Identifier"] != 66:
        # Not an OMDAQ file
        return False
    if informationHeader["ListMode"] != 2:
        # Not a list mode file
        return False
    else:
        return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = "-42181.LMF"
    print("Is OMDAQ LMF File = ", isOmdaqLmf(fname))
    omdaq = OmdaqLmf([fname])
    for i in range(len(omdaq)):
        adc = omdaq[i]
        if adc is None:
            continue
        print("ADC = ", i + 1)
        print(adc.info)
