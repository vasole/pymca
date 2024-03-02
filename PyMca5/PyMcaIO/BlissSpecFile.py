#!/usr/bin/env python
#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2020-2023 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
"""
This class exposes scan information stored by Bliss into redis as SPEC file.
"""
import os
import sys
import numpy
import time
import logging
_logger = logging.getLogger(__name__)

from PyMca5.PyMcaCore import TiledTools
HAS_REDIS = True


# try:
#     import RedisTools as redis
#     HAS_REDIS = True
# except Exception:
#     try:
#         import PyMca5.PyMcaCore.RedisTools as redis
#         HAS_REDIS = True
#     except Exception:
#         _logger.info("Cannot import PyMca5.PyMcaCore.RedisTools")
#         HAS_REDIS = False

if HAS_REDIS:
    from collections import OrderedDict

class BlissSpecFile(object):
    def __init__(self, filename, nscans=10):
        """
        filename is the name of the bliss session

        nscans to be ignored for tiled
        """
        self._scan_nodes = []
        self._session = filename
        # self._filename = redis.get_session_filename(self._session)
        self._tiledAdaptor=TiledTools.get_node(filename)

        #prefer to refer directly to tiled root, therefor no scan_nodes
        #self._scan_nodes = TiledTools.get_node(filename)

        self.list()
        self.__lastTime = 0
        self.__lastKey = "0.0"
        self._updatedOnce=False

    def list(self):
        """
        Return a string with all the scan keys separated by ,
        """
        _logger.info("list method called")
        scanlist = [f"{x.metadata['summary']['scan_id']}" for x in self._tiledAdaptor.client.values()]
        self._list = ["%s.1" % idx for idx in scanlist]
        return ",".join(scanlist)

    def __getitem__(self, item):
        """
        Returns the scan data
        """
        _logger.info("__getitem__ called %s" % item)
        t0 = time.time()
        _logger.info("trying to access %s",item)
        key = self._list[item]
        _logger.info("got item %s", key)
        if key == self.__lastKey and (t0 - self.__lastTime) < 1:
            # less than one second since last call, return cached value
            _logger.info("Returning cached value for key %s" % key)
        else:
            if key == self.__lastKey:
                _logger.info("Re-reading value for key %s" % key)
            self.__lastKey = key
            self.__lastItem = TiledScan(key,self._tiledAdaptor)
            self.__lastTime = time.time()
        return self.__lastItem

    def select(self, key):
        """
        key is of the from s.o
        scan number, scan order
        """
        _logger.info("select called %s" % key)
        n = self._list.index(key)
        return self.__getitem__(n)

    def scanno(self):
        """
        Gives back the number of scans in the file
        """
        _logger.info("scanno called")
        return len(self._tiledAdaptor._client)

    def allmotors(self):
        _logger.info("allmotors called")
        return []

    def isUpdated(self):
        _logger.info("BlissSpecFile is updated called")
        
        if not self._updatedOnce:
            self._updatedOnce=True
            return True

        #FIXME: UPDATE not implemented for tiled

        # # get last scan
        # scan_nodes = redis.get_session_scan_list(self._session,
        #                                           self._filename)
        # if not len(scan_nodes):
        #     # if we get no scans, information was emptied/lost and we'll get errors in any case
        #     # just say the file was updated. Perhaps the application asks for an update
        #     return True
        # scanlist = ["%s" % scan.name.split("_")[0] for scan in scan_nodes]
        # keylist = ["%s.1" % idx for idx in scanlist]
        # scankey = keylist[-1]

        # # if the last node is different, there are new data
        # if scankey != self._list[-1]:
        #     return True

        # # if the number of points or of mcas in the last node are different there are new data
        # # the problem is how to obtain the previous number of points and mcas but in any case
        # # we are going to read again the last scan
        # if self.__lastKey == scankey:
        #     # we have old data available
        #     previous_npoints = self.__lastItem.lines()
        #     previous_nmca = self.__lastItem.nbmca()

        # # read back (I do not force to read for the time being)
        # scan = self.select(scankey)
        # npoints = scan.lines()
        # nmca = scan.nbmca()
        # if self.__lastKey == scankey:
        #     if npoints > previous_npoints or nmca > previous_nmca:
        #         _logger.info("BlissSpecFile <%s> updated. New last scan data" % self._session)
        #         return True
        # # there might be new points or mcas in the last scan, but that is easy and fast
        # # to check by the main application because data are in cache
        # _logger.info("BlissSpecFile <%s> NOT updated." % self._session)
        return False

class TiledScan(object):
    def __init__(self, scan_ref, tiled_adaptor):
        self._scan_id=int(scan_ref.split(".")[0])
        self._scoped_tiled=tiled_adaptor.get_node(scan_id=self._scan_id)
        
        #following the nsls2 specific structure in tiled
        self._counters=None
        
        try:
            self._primary_data=self._scoped_tiled["primary"]["data"]
            self._tiledAdaptor=tiled_adaptor
            self._read_counters()
        except Exception:
            print("Scan ", scan_ref, "not valid")
            self._counters={}




    def _read_counters(self, force=False):
        _logger.info("_red_counters")
        self._counters={x:(y.shape) for x,y in self._primary_data.items() if len(y.shape)==1}

    def _sort_counters(self, counters):
        _logger.info("_sort_counters")
       
    def alllabels(self):
        """
        These are the labels associated to the counters
        """
        _logger.info("alllabels called")
        if self._counters is None:
            self.read_counters()
        return [key for key in self._counters]

    def allmotors(self):
        _logger.info("allmotors called")

    def allmotorpos(self):
        _logger.info("allmotorpos called")

    def cols(self):
        _logger.info("cols called")

    def command(self):
        _logger.info("command called")
        return "still to be added"
        #return self._scan_info.get("title", "No COMMAND")

    def data(self):
        # somehow I have to manage to get the same number of points in all counters
        _logger.info("TiledScan data called")
        keys = list(self._counters.keys())
        n_expected = self.lines()
        _logger.info("TiledScan data before data")
        data = numpy.empty((len(keys), n_expected), dtype=numpy.float32)
        _logger.info("TiledScan data after data")

        _logger.info("data shape %s ",data.shape) 
        i = 0
        for key in keys:
            cdata = numpy.array(self._primary_data[key])
            n = cdata.size
            if n >= n_expected:
                data[i] = cdata[:n_expected]
            else:
                data[i, :n] = cdata
                data[i, n:n_expected] = numpy.nan
            i += 1
        _logger.info("data shape %s, data %s",data.shape,data)
        return data

    def datacol(self, col):
        _logger.info("datacol called")
        return self.data()[col, :]

    def dataline(self,line):
        _logger.info("dataline called")
        return self.data()[:, line]

    def date(self):
        _logger.info("date called")
        return "2024-04-23"  #FIXME

    def fileheader(self, key=''):
        _logger.info("fileheader called")
        # this implementations returns the scan header instead of the correct
        # keys #E (file), #D (date) #O0 (motor names)
        

    def header(self,key):
        _logger.info("header called")

    def order(self):
        _logger.info("order called")
        return 1

    def number(self):
       _logger.info("number called")
        
    def lines(self):
        _logger.info("lines called")
        if self._counters is None:
            self._read_counters()
        ##return number of counters
        
        return max(self._counters.values())[0]
    
    def nbmca(self):
        _logger.info("nbmca called")
        return 0

    def mca(self,number):
        _logger.info("mca called")
        return 0
        

class BlissSpecScan(object):
    def __init__(self, scanNode):
        _logger.info("__init__ called %s" % scanNode.name)
        self._node = scanNode
        self._identification = scanNode.name.split("_")[0] + ".1"
        self._scan_info = redis.scan_info(self._node)
        # check if there are 1D detectors
        top_master, channels = next(iter(scanNode.info["acquisition_chain"].items()))
        if len(channels["spectra"]):
            # for the time being only one MCA read
            self._spectra = redis.get_spectra(scanNode, unique=True)
        else:
            self._spectra = []
        self._counters = None
        self._motors = self._scan_info.get("positioners", {})

    def _read_counters(self, force=False):
        if force or not self._counters:
            _counters = redis.get_scan_data(self._node)
            try:
                _counters = self._sort_counters(_counters)
            except Exception:
                _logger.error("Error sorting counters %s" % sys.exc_info()[1])
            self._counters = _counters

    def _sort_counters(self, counters):
        positioners = self.allmotors()
        title = self.command()
        tokens = title.split()
        scanned = [item for item in positioners if item in counters]
        if not len(scanned):
            scanned = [item for item in tokens if item in counters]

        # do nothing if there are no scanned motors assuming that the default
        # order will have the relevant items first
        if not len(scanned):
            return counters

        pure_counters = [item for item in counters if not (item in scanned)]
        # sort the pure counters
        if len(pure_counters) > 1:
            if sys.version_info > (3, 3):
                # sort irrespective of capital or lower case
                pure_counters.sort(key=str.casefold)
            else:
                # sort (capital letters first)
                pure_counters.sort()

        # sort the scanned motors
        if len(scanned) > 1:
            if sys.version_info > (3, 3):
                # sort irrespective of capital or lower case
                scanned.sort(key=str.casefold)
            else:
                # sort (capital letters first)
                scanned.sort()
            indices = []
            offset = len(tokens) + len(scanned)
            for item in scanned:
                if item in tokens:
                    indices.append((tokens.index(item), item))
                else:
                    indices.append((offset + scanned.index(item), item))
            indices.sort()
            scanned = [item for idx, item in indices]

        ordered = OrderedDict()
        for key in scanned:
            ordered[key] = counters[key]
        for key in pure_counters:
            ordered[key] = counters[key]
        return ordered

    def alllabels(self):
        """
        These are the labels associated to the counters
        """
        _logger.info("alllabels called")
        self._read_counters()
        return [key for key in self._counters]

    def allmotors(self):
        _logger.info("allmotors called")
        positioners = self._motors.get("positioners_start", {})
        return [key for key in positioners if not hasattr(positioners[key], "endswith")]

    def allmotorpos(self):
        _logger.info("allmotorpos called")
        positioners = self._motors.get("positioners_start", {})
        return [positioners[key] for key in positioners if not hasattr(positioners[key], "endswith")]

    def cols(self):
        _logger.debug("cols called")
        self._read_counters()
        return len(self._counters)

    def command(self):
        _logger.debug("command called")
        return self._scan_info.get("title", "No COMMAND")

    def data(self):
        # somehow I have to manage to get the same number of points in all counters
        _logger.info("BlissSpecScan data called")
        self._read_counters(force=True)
        counters = self._counters
        keys = list(counters.keys())
        n_actual = len(counters[keys[0]])
        n_expected = self.lines()
        data = numpy.empty((len(keys), n_expected), dtype=numpy.float32)
        i = 0
        for key in keys:
            cdata = counters[key]
            n = cdata.size
            if n >= n_expected:
                data[i] = cdata[:n_expected]
            else:
                data[i, :n] = cdata
                data[i, n:n_expected] = numpy.nan
            i += 1
        return data

    def datacol(self, col):
        _logger.debug("datacol called")
        return self.data()[col, :]

    def dataline(self,line):
        _logger.debug("dataline called")
        return self.data()[:, line]

    def date(self):
        _logger.debug("date called")
        text = 'sometime'
        text = self._scan_info.get("start_time", text)
        return self._scan_info.get("start_time_str", text)

    def fileheader(self, key=''):
        _logger.debug("fileheader called")
        # this implementations returns the scan header instead of the correct
        # keys #E (file), #D (date) #O0 (motor names)
        #
        self._read_counters()
        labels = '#L '
        for label in self._counters:
            labels += '  '+label
        return ['#S %s  %s' %(self._node.name.split("_")[0], self.command()),
                '#D %s' % self.date(),
                '#N %d' % self.cols(),
                labels]

    def header(self,key):
        _logger.debug("header called")
        if key == 'S':
            return self.fileheader()[0]
        elif key == 'D':
            return self.fileheader()[1]
        elif key == 'N':
            return self.fileheader()[-2]
        elif key == 'L':
            return self.fileheader()[-1]
        elif key == '@CALIB':
            output = []
            return output
        elif key == '@CTIME':
            # expected to send Preset Time, Live Time, Real (Elapsed) Time
            output = []
            return output
        elif key == "" or key == " ":
            return self.fileheader()
        else:
            output = []
            return output

    def order(self):
        _logger.debug("order called")
        return 1

    def number(self):
        _logger.debug("number called")
        return int(self._node.name.split("_")[0])

    def lines(self):
        _logger.debug("lines called")
        self._read_counters()
        counters = self._counters
        if len(counters):
            nlines = 0
            keyList = list(counters.keys())
            for key in keyList:
                n = len(counters[key])
                if n > nlines:
                    nlines = n 
            return nlines
        else:
            return 0

    def nbmca(self):
        _logger.debug("nbmca called")
        if len(self._spectra):
            return len(self._spectra[0])
        else:
            return 0

    def mca(self,number):
        _logger.debug("mca called")
        if number <= 0:
            raise IndexError("Mca numbering starts at 1")
        elif number > self.nbmca():
            raise IndexError("Only %d MCAs in file" % self.nbmca())
        return self._spectra[0][number - 1]

def isBlissSpecFile(filename):
    if os.path.exists(filename):
        return False
    try:
        if filename in redis.get_sessions_list():
            return True
    except Exception:
        pass
    return False

def test(filename):
    sf = BlissSpecFile(filename)
    print(sf[0].header('D'))
    print(sf[0].alllabels())
    print(sf[0].nbmca())
    if sf[0].nbmca():
        print(sf[0].mca(1))
    print(sf[0].header('S'))
    print(sf[0].allmotors())
    print(sf[0].allmotorpos())
    print(sf[0].header('@CTIME'))
    print(sf[0].header('@CALIB'))
    print(sf[0].header(''))
    print("Number of lines = ", sf[0].lines())
    if sf[0].lines():
        print("1st column = ", sf[0].datacol(0))
        print("1st line = ", sf[0].dataline(0))
    if sf.scanno() > 1:
        t0 = time.time()
        for i in range(sf.scanno()):
            #print(i)
            print(sf[i].header('S'))
            print(sf[i].header('D'))
            print(sf[i].alllabels())
            print(sf[i].nbmca())
            if sf[i].nbmca():
                print(sf[i].mca(1))
            print(sf[i].allmotors())
            print(sf[i].allmotorpos())
        print("elapsed = ", time.time() - t0)

if __name__ == "__main__":
    test(sys.argv[1])
