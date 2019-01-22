#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2019 European Synchrotron Radiation Facility
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
import copy
import logging
from . import DataObject
from PyMca5.PyMcaIO import spswrap as sps

_logger = logging.getLogger(__name__)


SOURCE_TYPE = 'SPS'


class SpsDataSource(object):
    def __init__(self, name):
        if not isinstance(name, str):
            raise TypeError("Constructor needs string as first argument")
        self.name = name
        self.sourceName = name
        self.sourceType = SOURCE_TYPE

    def refresh(self):
        pass

    def getSourceInfo(self):
        """
        Returns information about the Spec version in self.name
        to give application possibility to know about it before loading.
        Returns a dictionary with the key "KeyList" (list of all available keys
        in this source). Each element in "KeyList" is an shared memory
        array name.
        """
        return self.__getSourceInfo()

    def getKeyInfo(self, key):
        if key in self.getSourceInfo()['KeyList']:
            return self.__getArrayInfo(key)
        else:
            return {}

    def getDataObject(self, key_list, selection=None):
        if type(key_list) not in [type([])]:
            nolist = True
            key_list = [key_list]
        else:
            output = []
            nolist = False
        if self.name in sps.getspeclist():
            sourcekeys = self.getSourceInfo()['KeyList']
            for key in key_list:
                #a key corresponds to an array name
                if key not in sourcekeys:
                    raise KeyError("Key %s not in source keys" % key)
                #array = key
                #create data object
                data = DataObject.DataObject()
                data.info = self.__getArrayInfo(key)
                data.info['selection'] = selection
                data.data = sps.getdata(self.name, key)
                if nolist:
                    if selection is not None:
                        scantest = (data.info['flag'] &
                                    sps.TAG_SCAN) == sps.TAG_SCAN
                        if ((key in ["SCAN_D"]) or scantest) \
                            and 'cntlist' in selection:
                            data.x = None
                            data.y = None
                            data.m = None
                            if 'nopts' in data.info:
                                nopts = data.info['nopts']
                            elif 'nopts' in data.info['envdict']:
                                nopts = int(data.info['envdict']['nopts']) + 1
                            else:
                                nopts = data.info['rows']
                            if not 'LabelNames' in data.info:
                                data.info['LabelNames'] =\
                                    selection['cntlist'] * 1
                            newMemoryProblem = len(data.info['LabelNames']) != len(selection['cntlist'])
                            # check the current information is up-to-date
                            # (new HKL handling business)
                            actualLabelSelection = {'x':[], 'y':[], 'm':[]}
                            for tmpKey in ['x', 'y', 'm']:
                                if tmpKey in selection:
                                    for labelIndex in selection[tmpKey]:
                                        actualLabelSelection[tmpKey].append( \
                                                    selection['cntlist'][labelIndex])
                            if 'x' in selection:
                                for labelindex in selection['x']:
                                    #label = selection['cntlist'][labelindex]
                                    label = data.info['LabelNames'][labelindex]
                                    if label not in data.info['LabelNames']:
                                        raise ValueError("Label %s not in scan labels" % label)
                                    index = data.info['LabelNames'].index(label)
                                    if data.x is None: data.x = []
                                    data.x.append(data.data[:nopts, index])
                            if 'y' in selection:
                                #for labelindex in selection['y']:
                                for label in actualLabelSelection['y']:                                
                                    #label = data.info['LabelNames'][labelindex]
                                    if label not in data.info['LabelNames']:
                                        raise ValueError("Label %s not in scan labels" % label)
                                    index = data.info['LabelNames'].index(label)
                                    if data.y is None: data.y = []
                                    data.y.append(data.data[:nopts, index])
                            if 'm' in selection:
                                #for labelindex in selection['m']:
                                for label in actualLabelSelection['m']:                                
                                    #label = data.info['LabelNames'][labelindex]
                                    if label not in data.info['LabelNames']:
                                        raise ValueError("Label %s not in scan labels" % label)
                                    index = data.info['LabelNames'].index(label)
                                    if data.m is None: data.m = []
                                    data.m.append(data.data[:nopts, index])
                            data.info['selectiontype'] = "1D"
                            data.info['scanselection'] = True
                            if newMemoryProblem:
                                newSelection = copy.deepcopy(selection)
                                for tmpKey in ['x', 'y', 'm']:
                                    if tmpKey in selection:
                                        for i in range(len(selection[tmpKey])):
                                            if tmpKey == "x":
                                                label = data.info['LabelNames'][selection[tmpKey][i]]
                                            else:
                                                label = selection['cntlist'][selection[tmpKey][i]]
                                            newSelection[tmpKey][i] = data.info['LabelNames'].index(label)
                                data.info['selection'] = newSelection
                                data.info['selection']['cntlist'] = data.info['LabelNames']
                                selection = newSelection
                            data.data = None
                            return data
                        if (key in ["XIA_DATA"]) and 'XIA' in selection:
                            if selection["XIA"]:
                                if 'Detectors' in data.info:
                                    for i in range(len(selection['rows']['y'])):
                                        selection['rows']['y'][i] = \
                                            data.info['Detectors'].index(selection['rows']['y'][i]) + 1
                                    del selection['XIA']
                        return data.select(selection)
                    else:
                        if data.data is not None:
                            data.info['selectiontype'] = "%dD" % len(data.data.shape)
                            if data.info['selectiontype'] == "2D":
                                data.info["imageselection"] = True
                        return data
                else:
                    output.append(data.select(selection))
            return output
        else:
            return None

    def __getSourceInfo(self):
        arraylist = []
        sourcename = self.name
        for array in sps.getarraylist(sourcename):
            arrayinfo = sps.getarrayinfo(sourcename, array)
            arraytype = arrayinfo[2]
            arrayflag = arrayinfo[3]
            if arraytype != sps.STRING:
                if (arrayflag & sps.TAG_ARRAY) == sps.TAG_ARRAY:
                    arraylist.append(array)
                    continue
            _logger.debug("array not added %s", array)
        source_info = {}
        source_info["Size"] = len(arraylist)
        source_info["KeyList"] = arraylist
        return source_info

    def __getArrayInfo(self, array):
        info = {}
        info["SourceType"] = SOURCE_TYPE
        info["SourceName"] = self.name
        info["Key"] = array

        arrayinfo = sps.getarrayinfo(self.name, array)
        info["rows"] = arrayinfo[0]
        info["cols"] = arrayinfo[1]
        info["type"] = arrayinfo[2]
        info["flag"] = arrayinfo[3]
        counter = sps.updatecounter(self.name, array)
        info["updatecounter"] = counter


        envdict = {}
        keylist = sps.getkeylist(self.name, array + "_ENV")
        for i in keylist:
            val = sps.getenv(self.name, array + "_ENV", i)
            envdict[i] = val
        info["envdict"] = envdict
        scantest = (info['flag'] & sps.TAG_SCAN) == sps.TAG_SCAN
        metadata = None
        if (array in ["SCAN_D"]) or scantest:
            # try to get new style SCAN_D metadata
            metadata = sps.getmetadata(self.name, array)
            if metadata is not None:
                motors, metadata = metadata
                #info["LabelNames"] = metadata["allcounters"].split(";")
                labels = list(motors.keys())
                try:
                    labels = [(int(x),x) for x in labels]
                except:
                    _logger.warning("SpsDataSource error reverting to old behavior")
                    labels = [(x, x) for x in labels]
                labels.sort()
                if len(labels):
                    info["LabelNames"] = [motors[x[1]] for x in labels]
                if len(metadata["allmotorm"]):
                    info["MotorNames"] = metadata["allmotorm"].split(";")
                    info["MotorValues"] = [float(x) \
                                for x in metadata["allpositions"].split(";")]
                info["nopts"] = int(metadata["npts"])
                supplied_info = sps.getinfo(self.name, array)
                if len(supplied_info):
                    info["nopts"] = int(supplied_info[0]) 
                if 'hkl' in metadata:
                    if len(metadata["hkl"]):
                        info['hkl'] = [float(x) \
                                for x in metadata["hkl"].split(";")]
                # current SCAN
                if 'scanno' in metadata:
                    envdict["scanno"] = int(metadata["scanno"])
                # current SPEC file and title
                for key in ["datafile", "title"]:
                    if key in metadata:
                        envdict[key] = metadata[key]
                # put any missing information
                if "selectedcounters" in metadata:
                    info["selectedcounters"] = [x \
                                for x in metadata["selectedcounters"].split()]
                # do not confuse with unhandled keys ...
                #for key in metadata:
                #    if key not in info:
                #        info[key] = metadata[key]
        if (metadata is None) and ((array in ["SCAN_D"]) or scantest):
            # old style SCAN_D metadata
            if 'axistitles' in info["envdict"]:
                info["LabelNames"] = self._buildLabelsList(info['envdict']['axistitles'])
            if 'H' in info["envdict"]:
                if 'K' in info["envdict"]:
                    if 'L' in info["envdict"]:
                        info['hkl'] = [envdict['H'],
                                       envdict['K'],
                                       envdict['L']]
        calibarray = array + "_PARAM"
        if calibarray in sps.getarraylist(self.name):
            try:
                data = sps.getdata(self.name, calibarray)
                updc = sps.updatecounter(self.name, calibarray)
                info["EnvKey"] = calibarray
                # data is an array
                info["McaCalib"] = data.tolist()[0]
                info["env_updatecounter"] = updc
            except:
                # Some of our C modules return NULL without setting
                # an exception ...
                pass

        if array in ["XIA_DATA", "XIA_BASELINE"]:
            envarray = "XIA_DET"
            if envarray in sps.getarraylist(self.name):
                try:
                    data = sps.getdata(self.name, envarray)
                    updc = sps.updatecounter(self.name, envarray)
                    info["EnvKey"] = envarray
                    info["Detectors"] = data.tolist()[0]
                    info["env_updatecounter"] = updc
                except:
                    pass
        return info

    def _buildLabelsList(self, instr):
        _logger.debug('SpsDataSource : building counter list')
        state = 0
        llist  = ['']
        for letter in instr:
            if state == 0:
                if letter == ' ':
                    state = 1
                elif letter == '{':
                    state = 2
                else:
                    llist[-1] = llist[-1] + letter
            elif state == 1:
                if letter == ' ':
                    pass
                elif letter == '{':
                    state = 2
                    llist.append('')
                else:
                    llist.append(letter)
                    state = 0
            elif state == 2:
                if letter == '}':
                    state = 0
                else:
                    llist[-1] = llist[-1] + letter
        try:
            llist.remove('')
        except ValueError:
            pass

        return llist

    def isUpdated(self, sourceName, key):
        if sps.specrunning(sourceName):
            if sps.isupdated(sourceName, key):
                return True

            #return True if its environment is updated
            envkey = key + "_ENV"
            if envkey in sps.getarraylist(sourceName):
                if sps.isupdated(sourceName, envkey):
                    return True
        return False

source_types = {SOURCE_TYPE: SpsDataSource}


# TODO object is a builtins
def DataSource(name="", object=None, copy=True, source_type=SOURCE_TYPE):
    try:
        sourceClass = source_types[source_type]
    except KeyError:
        # ERROR invalid source type
        raise TypeError("Invalid Source Type, source type should be one of %s" % source_types.keys())

    return sourceClass(name, object, copy)


def main():
    import sys

    try:
        specname = sys.argv[1]
        arrayname = sys.argv[2]
        obj = DataSource(specname)
        data = obj.getData(arrayname)
        print("info = ", data.info)
    except:
        # give usage instructions
        print("Usage: SpsDataSource <specversion> <arrayname>")
        sys.exit()

if __name__ == "__main__":
    main()
