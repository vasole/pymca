#!/usr/bin/python
#    Copyright (c) 2010 Matthew Newville, The University of Chicago
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
__license__ = "MIT"
__author__ = "M. Newville - The University of Chicago"
"""
Simple interface to M. River's Multi-Element MCA Data Format

M. Newville
"""
import numpy as np
import re

MIN_SLOPE   = 1.e-7

def str_converter(strin, delim=None, converter=None):
    """convert a string of a delimited array to a list"""
    if delim is None:
        arr = strin.split()
    else:
        arr = re.split(delim, strin)
    conv = converter
    if hasattr(conv, '__call__'):
        return [conv(elem) for elem in arr]
    else:
        return arr

def str2float(strin, delim=None):
    "string of floats to array of floats"
    return str_converter(strin, delim=delim, converter=float)

def str2int(strin, delim=None):
    "string of integers to array of ints"
    return str_converter(strin, delim=delim, converter=int)

def str2str(strin, delim=None):
    "string to array of strings"
    return str_converter(strin, delim=delim)

class ROI(object):
    "simple Region of Interest"
    def __init__(self, index=0, left=-1, right=-1, name=None, spectra=None):
        self.index = index
        self.left = left
        self.right = right
        self.name = name
        self.spectra = np.array(spectra)
        self.__counts = -1

    def __repr__(self):
        return "<ROI('%s' chan:[%i, %i])>" % (self.name, self.left,
                                              self.right)
    def counts(self):
        "total counts in roi"
        if self.spectra is None:
            return None
        return self.spectra[self.left:self.right+1].sum()

class MCA(object):
    """ basic MCA spectra"""
    def __init__(self, data=None):
        self.npts = 1
        self.data = data
        self.energy = None
        self.realtime = 0
        self.livetime = 0
        self.deadtime_correction = 1.0
        self.cal_offset = 0
        self.cal_slope  = 0.010
        self.cal_quad   = 0
        self.cal_tth    = 0
        self.rois = []

    def __repr__(self):
        return "<MCA(%i points) %s>" % (self.npts, hex(id(self)))

    def chan2energy(self, i):
        "get energy from a channel number"
        if self.energy is not None:
            self.get_energy()
        return self.energy[i]

    def get_calibration(self):
        "return calibration constants"
        self.cal_slope = max(MIN_SLOPE, self.cal_slope)
        return  (self.cal_offset, self.cal_slope, self.cal_quad)

    def get_energy(self):
        "return full energy array"
        if self.energy is not None:
            return self.energy
        idx = np.arange(self.npts)
        self.cal_slope = max(MIN_SLOPE, self.cal_slope)
        self.energy = self.cal_offset + idx * (self.cal_slope +
                                               idx * self.cal_quad)
        return self.energy

class MEDFile(object):
    """MultiElement XRF Data File Format
    """
    def __init__(self, filename=None):
        self.default_detector = 0 # "good" detector for energy calibration
        self.env = []
        self.mcas = []
        self.filename = filename
        if filename is not None:
            self.mca_read_file(filename)

    def get_calibration(self, detector=None):
        "get calibration constants"
        if detector is None:
            detector = self.default_detector
        return self.mcas[detector].get_calibration()

    def chan2energy(self, i, detector=None):
        "get energy from a channel number"
        if detector is None:
            detector = self.default_detector
        return self.mcas[detector].chan2energy(i)

    def get_energy(self, detector=None):
        "get energy array"
        if detector is None:
            detector = self.default_detector
        return self.mcas[detector].get_energy()

    def get_data(self, detector=None, sum_all=True):
        """ get detector data,
        if sum_all == False, just the 1 array is returned
        if sum_all == True, the sum of all detectors is returned, aligned
                        to the energy of the specified detector"""
        if detector is None:
            detector = self.default_detector

        dat = self.mcas[detector].data
        if sum_all:
            enref = self.mcas[detector].get_energy()
            dat = np.zeros(len(enref))
            for mca in self.mcas:
                et = mca.get_energy()
                dt = mca.data[:]
                dat = dat + np.interp(enref, et, dt)
        return dat

    def mca_read_file(self, fname):
        "read MCA data file"
        self.filename = fname
        f     = open(fname)
        lines = f.readlines()
        f.close()

        mode   = 'HEADER'
        nelem = 1
        # tmp data for data and headers, and rois
        tmpdat = []
        header = {}
        _roi_0, _roi_1, _roi_n = {}, {}, {}
        for line in lines:
            line  = line.strip()
            if len(line) < 1:
                continue
            if mode == 'DATA': # data mode
                tmpdat.append(str2int(line))
            else:
                words = [x.strip() for x in line.split(' ', 1)]
                if len(words) < 2:  # note that 'Data:' line as 1 word.
                    words.append('')
                tag, val = words[0], words[1]

                tag = tag.replace(':', '').lower()
                if tag == 'data':
                    mode = 'DATA'
                elif tag == 'elements':
                    nelem = int(val)
                elif tag in ('rois', 'real_time', 'live_time', 'cal_offset',
                             'cal_slope', 'cal_quad', 'two_theta'):
                    header[tag] = str2float(val)
                elif tag == 'environment':
                    self.env.append(val)
                elif tag.startswith('roi_'):
                    x, sroi, item = tag.split('_')
                    iroi = int(sroi)
                    if item == "label":
                        labels = str2str(val, delim='\&')
                        if labels[-1] == '':
                            labels = labels[:-1]
                        _roi_n[iroi] = labels
                    elif item == "left":
                        _roi_0[iroi] = str2int(val)
                    elif item == "right":
                        _roi_1[iroi] = str2int(val)
                else:
                    header[tag] = val

        #  find first valid detector, identify bad detectors
        self.mcas = [MCA() for i in range(nelem)]
        tmpdat  = np.transpose(np.array(tmpdat))

        for imca, mca in enumerate(self.mcas):
            mca.npts       = int(header['channels'])
            mca.nrois      = int(header['rois'][imca])
            mca.start_time = header['date']
            mca.realtime   = header['real_time'][imca]
            mca.livetime   = header['live_time'][imca]
            mca.cal_offset = header['cal_offset'][imca]
            mca.cal_slope  = header['cal_slope'][imca]
            mca.cal_quad   = header['cal_quad'][imca]
            mca.cal_tth    = header['two_theta'][imca]

            mca.data = 1 * tmpdat[imca, :]
            for iroi in _roi_n:
                name   = _roi_n[iroi][imca].strip()
                ileft  = _roi_0[iroi][imca]
                iright = _roi_1[iroi][imca]
                mca.rois.append(ROI(index=iroi, left=ileft,
                                    right=iright, name=name,
                                    spectra=mca.data))


    def write_ascii(self, fname, elem=None, sum_all=True):
        """write data to ASCII column file"""
        out = []

        out.append("# XRF data from %s\n" % (self.filename))
        if len(self.env)>0:
            out.append("# Extra PVs:\n")
            for i in self.env:
                out.append("#     %s\n" % i)
        out.append("#-------------------------\n")
        out.append("# energy  counts\n")

        en = self.get_energy()
        if elem is not None:
            dat = self.get_data(detector=elem)
        elif sum_all:
            dat = self.get_data()
        for i in ("%8.4f  %i\n" % (ei, di) for ei, di in zip(en, dat)):
            out.append("%s"%i)

        f = open(fname, "w+")
        f.writelines(out)
        f.close()

if __name__ == '__main__':
    try:
        import pylab
        HAS_PYLAB = True
    except ImportError:
        HAS_PYLAB = False
    xrf = MEDFile('test.xrf')

    energy = xrf.get_energy()

    d0 = xrf.get_data(detector=0, sum_all=False)
    d1 = xrf.get_data(detector=1, sum_all=False)
    d2 = xrf.get_data(detector=2, sum_all=False)
    d3 = xrf.get_data(detector=3, sum_all=False)
    dsum = xrf.get_data(detector=0, sum_all=True)

    xrf.write_ascii('test.dat')

    print(' ROIs from Element 2:')
    print(' ------------------')
    print(' Name |  Sum ')
    for roi in xrf.mcas[1].rois:
        print(' %s = %d ' % (roi.name, roi.counts()))

    if HAS_PYLAB:
        pylab.plot(energy, dsum)
        pylab.show()

