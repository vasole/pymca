#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
import types
from . import DataObject
from PyMca5.PyMcaIO import spswrap as sps

DEBUG = 0
SOURCE_TYPE = 'SPS'


class SpsDataSource(object):
    def __init__(self, name):
        if type(name) not in types.StringTypes:
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
        if type(key_list) != types.ListType:
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
                            if 'nopts' in data.info['envdict']:
                                nopts = int(data.info['envdict']['nopts']) + 1
                            else:
                                nopts = data.info['rows']
                            if not 'LabelNames' in data.info:
                                data.info['LabelNames'] =\
                                    selection['cntlist'] * 1
                            if 'x' in selection:
                                for labelindex in selection['x']:
                                    label = data.info['LabelNames'][labelindex]
                                    if label not in data.info['LabelNames']:
                                        raise ValueError("Label %s not in scan labels" % label)
                                    index = data.info['LabelNames'].index(label)
                                    if data.x is None: data.x = []
                                    data.x.append(data.data[:nopts, index])
                            if 'y' in selection:
                                for labelindex in selection['y']:
                                    label = data.info['LabelNames'][labelindex]
                                    if label not in data.info['LabelNames']:
                                        raise ValueError("Label %s not in scan labels" % label)
                                    index = data.info['LabelNames'].index(label)
                                    if data.y is None: data.y = []
                                    data.y.append(data.data[:nopts, index])
                            if 'm' in selection:
                                for labelindex in selection['m']:
                                    label = data.info['LabelNames'][labelindex]
                                    if label not in data.info['LabelNames']:
                                        raise ValueError("Label %s not in scan labels" % label)
                                    index = data.info['LabelNames'].index(label)
                                    if data.m is None: data.m = []
                                    data.m.append(data.data[:nopts, index])
                            data.info['selectiontype'] = "1D"
                            data.info['scanselection'] = True
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
            if DEBUG:
                print("array not added %s" % array)
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
        if (array in ["SCAN_D"]) or scantest:
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
        if DEBUG:
            print('SpsDataSource : building counter list')
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
