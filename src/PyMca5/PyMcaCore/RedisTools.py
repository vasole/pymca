#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2019-2023 European Synchrotron Radiation Facility
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
from collections import OrderedDict
import logging
_logger = logging.getLogger(__name__)

from bliss.config import get_sessions_list
my_get_sessions_list = None
import os
from blissdata.redis_engine.store import DataStore
# initialize some defaults
DATA_STORE = None
BEACON_HOST = os.getenv("BEACON_HOST")
SESSION_LIST = []
SESSION = None
if BEACON_HOST is not None:
    try:
        DATA_STORE = DataStore("redis://%s2" % BEACON_HOST[:-1])
    except Exception:
        _logger.info("DATA_STORE not initialized")
        DATA_STORE = None
print("DATA_STORE = %s" % DATA_STORE)

def my_get_sessions_list(data_store=DATA_STORE):
    timestamp, scan_name =data_store.get_last_scan()
    scan = data_store.load_scan(scan_name)
    last_session = scan.session
    timestamp, scan_list = data_store.search_existing_scans()
    if len(scan_list) > 20:
        scan_list = scan_list[-20:]
    session_list = [last_session]
    for scan_name in scan_list:
        scan = data_store.load_scan(scan_name)
        if scan.session not in session_list:
            session_list.append(scan.session)
    return session_list
get_sessions_list = my_get_sessions_list
def get_node(session_name):
    # dummy usage
    return session_name

if DATA_STORE:
    try:
        SESSION_LIST = get_sessions_list(DATA_STORE)
        SESSION = SESSION_LIST[0]
    except Exception:
        SESSION = None
        SESSION_LIST = []

print("SESSION = %s" % SESSION)
print("SESSION_LIST = %s" % SESSION_LIST)

_NODE_TYPES = [ "channel",
                "lima",
                "node_ref_channel",
                "scan",
                "scan_group"]

def get_node_list(node, node_type=None, name=None, db_name=None, dimension=None,
                  filter=None, unique=False, reverse=False, ignore_underscore=True):
    """
    Return list of nodes matching the given filter
    """
    if not hasattr(node, "name"):
        input_node = get_node(node)
    else:
        input_node = node
    if hasattr(input_node, "walk"):
        if reverse:
            iterator = input_node.walk_from_last
        else:
            iterator = input_node.walk
    else:
        if reverse:
            iterator = input_node.iterator.walk_from_last
        else:
            iterator = input_node.iterator.walk
    if node_type:
        if hasattr(node_type, "lower"):
            node_type = node_type.lower()
            if node_type not in _NODE_TYPES:
                _logger.warning("Node type %s ignored" % node_type)
                node_type = None

    output_list = []
    # walk not waiting
    if node_type or name or db_name or dimension:
        for node in iterator(wait=False, include_filter=filter):
            if ignore_underscore and hasattr(node.name, "startswith") and node.name.startswith("_"):
                continue
            if not _check_dimension(node, dimension):
                continue
            if node_type and (node.type == node_type):
                output_list.append(node)
            elif name and (node.name == name):
                output_list.append(node)
            elif db_name and node.db_name == db_name:
                output_list.append(node)
            else:
                output_list.append(node)
            if unique and len(output_list):
                break
    else:
        for node in iterator(wait=False, include_filter=filter):
            #print(node.name, node.db_name, node)
            if ignore_underscore and hasattr(node.name, "startswith") and node.name.startswith("_"):
                continue
            output_list.append(node)
            if unique:
                break
    return output_list

def _check_dimension(node, dimension=None):
    if dimension is None:
        return True
    elif not hasattr(node, "shape"):
        return False
    elif len(node.shape) == dimension:
        return True
    else:
        return False

def get_session_scan_list(session, filename=None, data_store=DATA_STORE):
    """
    Returns a sorted list of actual scans. Last scan is last.
    """
    timestamp, keys = data_store.search_existing_scans(session=session)
    if len(keys) > 10:
        scans = [data_store.load_scan(scan_name) for scan_name in keys[-10:]]
    else:
        scans = [data_store.load_scan(scan_name) for scan_name in keys[-10:]]
    if filename:
        scans = [scan for scan in scans if scan.info["filename"] == filename]
    return scans

def _get_session_scans(session):
    if hasattr(session, "name"):
        session_node = session
        session_name = session.name
    else:
        session_node =  get_node(session)
        session_name = session
    db_names = rdsscan( f"{session_name}:*_children_list",
                        count=1000000,
                        connection=session_node.db_connection,
                      )
    # we are interested on actual scans, therefore we do not take scans
    # whose name starts by underscore
    return ( node
             for node in get_nodes(
                *(db_name.replace("_children_list", "") for db_name in db_names)
             )
             if node is not None and node.type == "scan" and \
                 hasattr(node.name, "startswith") and not node.name.startswith("_")
            )

def get_session_last_scan(session, data_store=DATA_STORE):
    return get_session_scan_list(session)[-1]

def get_session_filename(session, data_store=DATA_STORE):
    """
    Return filename associated to last session scan or an empty string
    """
    return get_filename(session, data_store=data_store)

def get_scan_list(session_node, data_store=DATA_STORE):
    if my_get_sessions_list is None:
        return get_node_list(session_node, node_type="scan", filter="scan")
    else:
        print("CALLED")

def get_data_channels(node, unique=False):
    return get_node_list(node, node_type="channel", filter="channel", dimension=0, unique=unique)

def get_spectrum_nodes(node, dimension=1, unique=False):
    return get_node_list(node, node_type="channel", filter="channel", dimension=1, unique=unique)

def get_spectra(node, unique=False):
    spectra_nodes = get_spectrum_nodes(node, unique=unique)
    if len(spectra_nodes):
        return [spectra_node.get_as_array(0, -1) for spectra_node in spectra_nodes]
    else:
        return []

def get_filename(session_node, data_store=DATA_STORE):
    filename = ""
    scan_list = get_session_scan_list(session_node, data_store=data_store)
    info = scan_list[-1].info
    return info.get("filename", "")

def get_filenames(node):
    filenames = []
    #print(dir(node))
    #print(node.info.keys())
    if hasattr(node, "info"):
        info = node.info
        if "filename" in info:
            filenames.append(info["filename"])
    elif node.type == "scan":
        info = scan_info(node)
        if "filename" in info:
            filenames.append(info["filename"])
    else:
        scan_list = get_scan_list(node)
        for scan in scan_list:
            info = scan_info(scan)
            if "filename" in info:
                filename = info["filename"]
                if filename not in filenames:
                    filenames.append(filename)
    #print("FILENAMES from node", filenames)
    return filenames

def get_last_spectrum_instances(session_node, offset=None):
    sc = get_scan_list(session_node)
    sc.reverse()
    spectra = OrderedDict()
    if offset is None:
        if len(sc) > 10:
            start = 10
        else:
            start = 0
    else:
        start = offset
    for scan in sc:
        sp = get_spectra(scan)
        names = [(x, x.name) for x in sp]
        for obj, name in names:
            if name not in spectra:
                print("adding name ", obj.db_name, " scan = ", scan.name)
                spectra[name] = (obj, scan)
    return spectra

def shortnamemap(names, separator=":"):
    """
    Map full Redis names to short (but still unique) names

    :param list(str) names:
    :param str separator:
    :returns dict:
    """
    if not names:
        return {}
    names = set(names)
    parts = [name.split(separator) for name in names]
    nparts = max(map(len, parts))
    parts = [([""] * (nparts - len(lst))) + lst for lst in parts]
    ret = {}
    for i in reversed(range(-nparts, 0)):
        joinednames = [separator.join(s for s in lst[i:] if s) for lst in parts]
        newnames = joinednames + list(ret.values())
        selection = [
            (idx, (separator.join(s for s in lst if s), name))
            for idx, (name, lst) in enumerate(zip(joinednames, parts))
            if newnames.count(name) == 1
        ]
        if selection:
            idx, tuples = list(zip(*selection))
            ret.update(tuples)
            parts = [lst for j, lst in enumerate(parts) if j not in idx]
    return ret

def get_scan_data(scan_node, unique=False, top_master=False):
    data_channels = get_data_channels(scan_node, unique=unique)
    if top_master:
        try:
            top_master, channels = \
                    next(iter(scan_node.info["acquisition_chain"].items()))
            if "scalars" in channels:
                _logger.info("Taking only scalar data from top master")
                data_channels = [x for x in data_channels \
                                     if x.name in channels["scalars"]]
        except Exception:
            _logger.warning("Cannot perform top_master filtering")
    names = shortnamemap(x.name for x in data_channels)
    result = {}
    i = 0
    for channel in data_channels:
        # names :mon and :det from ID10 are badly mapped
        if channel.name not in names and channel.name.startswith(":"):
            short_name = names[channel.name[1:]]
        else:
            short_name = names[channel.name]
        result[short_name] = channel.get_as_array(0, -1)
        i += 1
    return result

def scan_info(scan_node):
    """
    See https://gitlab.esrf.fr/bliss/bliss/-/blob/master/bliss/data/display.py

    def collect_channels_info(self, scan_info):

                #------------- scan_info example -------------------------------------------------------

                # session_name = scan_info.get('session_name')             # ex: 'test_session'
                # user_name = scan_info.get('user_name')                   # ex: 'pguillou'
                # filename = scan_info.get('filename')                     # ex: '/mnt/c/tmp/test_session/data.h5'
                # node_name = scan_info.get('node_name')                   # ex: 'test_session:mnt:c:tmp:183_ascan'

                # start_time = scan_info.get('start_time')                 # ex: datetime.datetime(2019, 3, 18, 15, 28, 17, 83204)
                # start_time_str = scan_info.get('start_time_str')         # ex: 'Mon Mar 18 15:28:17 2019'
                # start_timestamp = scan_info.get('start_timestamp')       # ex: 1552919297.0832036

                # save = scan_info.get('save')                             # ex: True
                # sleep_time = scan_info.get('sleep_time')                 # ex: None

                # title = scan_info.get('title')                           # ex: 'ascan roby 0 10 10 0.01'
                # scan_type = scan_info.get('type')                        # ex:    ^
                # start = scan_info.get('start')                           # ex:             ^              = [0]
                # stop = scan_info.get('stop')                             # ex:                ^           = [10]
                # npoints = scan_info.get('npoints')                       # ex:                   ^        = 10
                # count_time = scan_info.get('count_time')                 # ex:                       ^    = 0.01

                # total_acq_time = scan_info.get('total_acq_time')         # ex: 0.1  ( = npoints * count_time )
                # scan_nb = scan_info.get('scan_nb')                       # ex: 183

                # positioners_dial = scan_info.get('positioners_dial')     # ex: {'bad': 0.0, 'calc_mot1': 20.0, 'roby': 20.0, ... }
                # positioners = scan_info.get('positioners')               # ex: {'bad': 0.0, 'calc_mot1': 20.0, 'roby': 10.0, ...}

                # acquisition_chain = scan_info.get('acquisition_chain')
                # ex: {'axis':
                #       {
                #         'master' : {'scalars': ['axis:roby'], 'spectra': [], 'images': [] },
                #         'scalars': ['timer:elapsed_time', 'diode:diode'],
                #         'spectra': [],
                #         'images' : []
                #       }
                #     }
                # master, channels = next(iter(scan_info["acquisition_chain"].items()))
                # master = axis
                # channels = {'master': {'scalars': ['axis:roby'],
                #                        'scalars_units': {'axis:roby': None},
                #                        'spectra': [],
                #                        'images': [],
                #                        'display_names': {'axis:roby': 'roby'}
                #                       },

                #             'scalars': ['timer:elapsed_time',
                #                         'timer:epoch',
                #                         'lima_simulator2:bpm:x',
                #                         'simulation_diode_sampling_controller:diode'],
                #
                #             'scalars_units': {'timer:elapsed_time': 's',
                #                               'timer:epoch': 's',
                #                               'lima_simulator2:bpm:x': 'px',
                #                               'simulation_diode_sampling_controller:diode': None},
                #             'spectra': [],
                #             'images': [],
                #             'display_names': {'timer:elapsed_time': 'elapsed_time',
                #                               'timer:epoch': 'epoch',
                #                               'lima_simulator2:bpm:x': 'x',
                #                               'simulation_diode_sampling_controller:diode': 'diode'}}

        # ONLY MANAGE THE FIRST ACQUISITION BRANCH (multi-top-masters scan are ignored)
        top_master, channels = next(iter(scan_info["acquisition_chain"].items()))
        """
    return scan_node.info

if __name__ == "__main__":
    import sys
    # get the available sessions
    scan_number = None
    reference = None
    _logger.setLevel(logging.DEBUG)
    if len(sys.argv) > 1:
        sessions = [sys.argv[1]]
        if len(sys.argv) > 2:
            scan_number = sys.argv[2]
    else:
        sessions = get_sessions_list()
    for session_name in sessions:
        print("SESSION <%s>"  % session_name)
        #connection = client.get_redis_connection(db=1)
        #while not DataNode.exists(session_name, None, connection):
        #    gevent.sleep(1)
        # get node
        session_node = get_node(session_name)
        if not session_node:
            print("\tNot Available")
            continue
        filename = get_filename(session_node)
        print(filename)
        scans = get_session_scan_list(session_node)
        for scan in scans:
            filenames = get_filenames(scan)
            nFiles = len(filenames)
            if not nFiles:
                filename = "No FILE"
            else:
                if nFiles > 1:
                    print("WARNING, more than one file associated to scan")
                filename = filenames[-1]
            print('*************** ', filename)
            sInfo = scan_info(scan)
            print(list(sInfo.keys()))
            title = sInfo.get("title", "No COMMAND")
            if scan_number:
                if scan.name.startswith("161"):
                    reference = scan
            else:
                print("\t%s %s %s"  % (scan.name, filename, title))
    if len(sessions) == 1:
        if reference:
            scan = reference
        else:
            scan = scans[-1]
        print("SCAN = %s" % scan)
        print("NAME = %s" % scan.name)
        print("NUMBER = %d" % scan.number)
        print("SESSION = %s" % scan.session)
        info = scan_info(scan) # scan.info
        print("TITLE = %s" % info.get("title", "No COMMAND"))
        print("ACQUISITION_CHAIN = ", info.get("acquisition_chain", None))
        #for master in scan_info(scan)["acquisition_chain"]:
        #    print(" master = ", master)
        #    print(" channels = ", list(scan_info(scan)[master].keys()))
        top_master, channels = next(iter(scan.info["acquisition_chain"].items()))
        print("N counters devices = ", len(channels["spectra"]))
        print("N Mca devices = ", len(channels["spectra"]))
        print(top_master)
        print(channels)
        sys.exit(0)
        counters = get_data_channels(scan)
        #for counter in counters:
        #    print(counter.name, counter.short_name, counter.dtype, counter.type, counter.info, counter.get_as_array(0, -1))
        print(get_scan_data(scan))
