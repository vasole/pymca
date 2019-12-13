#/*##########################################################################
# Copyright (C) 2019 European Synchrotron Radiation Facility
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
try:
    from bliss.config import get_sessions_list
except:
    try:
        from bliss.shell.cli import get_sessions_list
    except ImportError:
        # This point should not be reached
        from bliss.config import static
    def get_sessions_list():
        """Return a list of available sessions found in config"""
        all_sessions = list()
        config = static.get_config()
        for name in config.names_list:
            c = config.get_config(name)
            if c.get("class") != "Session":
                continue
            if c.get_inherited("plugin") != "session":
                continue
            all_sessions.append(name)
        return all_sessions
        
from bliss.data.node import get_node, DataNode, DataNodeContainer

try:
    from bliss.data.scan import Scan
except:
    from bliss.data.nodes.scan import Scan

try:
    from bliss.data.channel import ChannelDataNode
except:
    from bliss.data.nodes.channel import ChannelDataNode

NODE_TYPE = {}
NODE_TYPE["Scan"] = Scan
NODE_TYPE["DataNode"] = DataNode
NODE_TYPE["DataNodeContainer"] = DataNodeContainer
NODE_TYPE["ChannelDataNode"] = ChannelDataNode

def get_node_list(input_node, node_type=None, name=None, db_name=None, dimension=None,
                  filter=None, unique=False):
    if node_type in NODE_TYPE.keys():
        node_type = NODE_TYPE[node_type]
    output_list = []
    # walk not waiting
    if node_type or name or db_name or dimension:
        for node in input_node.iterator.walk(wait=False, filter=filter):
            if node_type and isinstance(node, node_type):
                output_list.append(node)
            elif name and (node.name == name):
                output_list.append(node)
            elif db_name and node.db_name == db_name:
                output_list.append(node)
            elif dimension:
                if hasattr(node, "shape"):
                    shape = node.shape
                    if len(shape) == dimension:
                        output_list.append(node)
                        print(node.name, node.db_name)
            if unique and len(output_list):
                break 
    else:
        for node in input_node.iterator.walk(wait=False, filter=filter):
            #print(node.name, node.db_name, node)
            output_list.append(node)
            if unique:
                break
    return output_list

def get_scan_list(session_node):
    return get_node_list(session_node, node_type="Scan", filter="scan")

def get_data_channels(node):
    return get_node_list(node, node_type="ChannelDataNode", filter="channel")

def get_spectra(node, dimension=1):
    return get_node_list(node, filter="channel", dimension=1)

def get_session_filename(session_node):
    scan_list = get_scan_list(session_node)
    filename = ""
    if len(scan_list):
        info = scan_info(scan_list[-1])
        if "filename" in info:
            filename = info["filename"]
    return filename

def get_filenames(node):
    filenames = []
    if isinstance(node, NODE_TYPE["Scan"]):
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

def scan_info(scan_node):
    return scan_node.info.get_all()
        
if __name__ == "__main__":
    # get the available sessions
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
        scans = get_scan_list(session_node)
        for scan in scans:
            filenames = get_filenames(scan)
            nFiles = len(filenames)
            if not nFiles:
                filename = "No FILE"
            else:
                if nFiles > 1:
                    print("WARNING, more than one file associated to scan")
                filename = filenames[-1]
            title = scan_info(scan).get("title", "No COMMAND")
            print("\t%s %s %s"  % (scan.name, filename, title))
