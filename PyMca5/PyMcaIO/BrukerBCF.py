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
import base64
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
        # it seems the library performs a binning when downsampling.
        self._binning = 1
        try:
            self.data = reader.parse_hypermap(downsample=self._binning,
                                              lazy=False)
        except:
            if "MemoryError" in "%s" % (sys.exc_info()[0],):
                self._binning += 1
                self.data = reader.parse_hypermap(downsample=self._binning,
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
        xml_str = bruker.fix_dec_patterns.sub(b'\\1.\\2', header_byte_str)

        root = bruker.ET.fromstring(xml_str)
        root = root.find("./ClassInstance[@Type='TRTSpectrumDatabase']")
        # root.atrib["Name"] contains information
        xScale, yScale = get_scales(root)
        self.info["xScale"] = xScale
        if xScale:
            self.info["xScale"][1] = xScale[1] * self._binning
        self.info["yScale"] = yScale
        if yScale:
            self.info["yScale"][1] = yScale[1] * self._binning
        self.info["McaCalib"] = get_calibration(root)
        if self._binning == 1:
            if 1:#try:
                live_times = get_live_times(root)
            #except:
            #    _logger.warning("Error retrieving spectra live time")
            #    live_times = None
            self.info["live_time"] = live_times

        if __name__ == "__main__":
            # images
            images = get_overview_images(root, name="Default")
            if not len(images):
                images = None
            else:
                im = images[0]
                print(self.info["xScale"])
                print(self.info["yScale"])
                print(im["xScale"])
                print(im["yScale"])
                # I should recover images, image names, scales, ...
                print("MAP START = (%f, %f) END = (%f, %f)"  % (self.info["xScale"][0],
                                                                self.info["yScale"][0],
                                                                self.info["xScale"][1] * shape[1],
                                                                self.info["yScale"][1] * shape[1]))

                print("IMAGE START = (%f, %f) END = (%f, %f)"  % (im["xScale"][0],
                                                                  im["yScale"][0],
                                                                  im["xScale"][1] * im["image"].shape[1],
                                                                  im["yScale"][1] * im["image"].shape[1]))
                # coordinates of rectangle in image pixels
                def convertToRowAndColumn(x, y, shape, xScale=None, yScale=None, safe=True):
                    if xScale is None:
                        c = x
                    else:
                        c = (x - xScale[0]) / xScale[1]
                    if yScale is None:
                        r = y
                    else:
                        r = ( y - yScale[0]) / yScale[1]

                    if safe:
                        c = min(int(c), shape[1] - 1)
                        c = max(c, 0)
                        r = min(int(r), shape[0] - 1)
                        r = max(r, 0)
                    else:
                        c = int(c)
                        r = int(r)
                    return r, c

                r0, c0 = convertToRowAndColumn(self.info["xScale"][0],
                                               self.info["yScale"][0],
                                               im["image"].shape,
                                               xScale=im["xScale"],
                                               yScale=im["yScale"],
                                               safe=True)
                r1, c1 = convertToRowAndColumn(self.info["xScale"][0],
                                               self.info["yScale"][0],
                                               im["image"].shape,
                                               xScale=im["xScale"],
                                               yScale=im["yScale"],
                                               safe=True)


def dump_dict(root, offset=0):
    class_instances = root.findall("./ClassInstance")
    for class_instance in class_instances:
        print(offset * " " + class_instance.attrib["Type"] + " " +class_instance.attrib.get("Name", "-") )
        for child in class_instance:
            if child.tag == "ChildClassInstances":
                dump_dict(child, offset=offset + 5)
        dump_dict(class_instance, offset=offset+10)

def get_scales(root):
    semData = root.find("./ClassInstance[@Type='TRTSEMData']")
    semData_dict = bruker.dictionarize(semData)
    semStageData = root.find("./ClassInstance[@Type='TRTSEMStageData']")
    semStageData_dict = bruker.dictionarize(semStageData)
    if "DX" in semData_dict and "DY" in semData_dict:
        if "X" in semStageData_dict and "Y" in semStageData_dict:
            xScale = [semStageData_dict["X"], semData_dict["DX"]]
            yScale = [semStageData_dict["Y"], semData_dict["DY"]]
        else:
            xScale = [0.0, semData_dict["DX"]]
            yScale = [0.0, semData_dict["DY"]]
    else:
        xScale = None
        yScale = None
    return xScale, yScale

def get_calibration(root):
    spectrum_header = root.find(".//ClassInstance[@Type='TRTSpectrumHeader']")
    calibration = [0.0, 1.0, 0.0]
    if spectrum_header:
        spectrum_header_data = bruker.dictionarize(spectrum_header)
        print(spectrum_header_data.keys())
        calibrated = True
        for key in ["ChannelCount", "CalibAbs", "CalibLin"]:
            if key not in spectrum_header_data:
                calibrated = False
                break
        if calibrated:
            calibration = [spectrum_header_data["CalibAbs"],
                            spectrum_header_data["CalibLin"],
                            0.0]
    return calibration

def get_live_times(root):
    result = None
    image_nodes = root.findall("./ClassInstance[@Type='TRTImageData']")
    # for the time being we retrieve only one live_time image
    for node in image_nodes:
        name = node.get('Name')
        print("name = ", name)
        if not node.get('Name'):
            width = int(node.find('./Width').text)
            height = int(node.find('./Height').text)
            dtype = 'u' + node.find('./ItemSize').text
            plane_count = int(node.find('./PlaneCount').text)
            if plane_count == 1:
                image_data_node = node.find("./Plane0")
                description = image_data_node.find("./Description")
                if hasattr(description, "text"):
                    if description.text.lower() == "video":
                        # can it be different?
                        image_data_node = node.find("./Plane0")
                        decoded = base64.b64decode(image_data_node.find('./Data').text)
                        image_data = numpy.frombuffer(decoded, dtype=dtype)
                        image_data.shape = height, width
                        if __name__ == "__main__":
                            from PyMca5.PyMcaIO import TiffIO
                            tiff = TiffIO.TiffIO(description.text + ".tif", "wb")
                            tiff.writeImage(image_data,
                                            info={"Title":description.text})
                            tiff = None
        if name == "PixelTimes":
            width = int(node.find('./Width').text)
            height = int(node.find('./Height').text)
            dtype = 'u' + node.find('./ItemSize').text
            plane_count = int(node.find('./PlaneCount').text)
            if plane_count == 1:
                # can it be different for PixelTimes?
                image_data_node = node.find("./Plane0")
                decoded = base64.b64decode(image_data_node.find('./Data').text)
                result = numpy.frombuffer(decoded, dtype=dtype)
                result.shape = height, width
                # express in seconds
                result = (result * 1.0e-6).astype(numpy.float32)
                break
    return result

def get_overview_images(root, name=None):
    result = None
    container = root.find("./ClassInstance[@Type='TRTContainerClass']")
    images = []
    image_nodes = []
    for item in container:
        if item.tag == "ChildClassInstances":
            for child in item:
                child_name = child.get("Name")
                if not child_name:
                    continue
                if child_name == "OverviewImages":
                    for element in child:
                        image_nodes = element.findall("./ClassInstance[@Type='TRTImageData']")
                    break

    for node in image_nodes:
        node_name = node.get('Name')
        if name:
            if not node_name:
                # it cannot be equal to requested name
                continue
            elif name != node_name:
                continue
        width = int(node.find('./Width').text)
        height = int(node.find('./Height').text)
        dtype = 'u' + node.find('./ItemSize').text
        plane_count = int(node.find('./PlaneCount').text)
        x_calibration = float(node.find('./XCalibration').text)
        y_calibration = float(node.find('./YCalibration').text)
        plane_count = int(node.find('./PlaneCount').text)
        if plane_count == 1:
            # monochrome or data image
            image_data_node = node.find("./Plane0")
            decoded = base64.b64decode(image_data_node.find('./Data').text)
            result = numpy.frombuffer(decoded, dtype=dtype)
            result.shape = height, width
            images.append(result)
        elif plane_count == 3:
            if __name__ == "__main__":
                from PyMca5.PyMcaIO import TiffIO
                tiff = None
            # color picture
            result = numpy.zeros((height * width, 3), dtype=dtype)
            planes = [] * plane_count
            for plane_index in range(plane_count):
                image_data_node = node.find("./Plane%d" % plane_index)
                decoded = base64.b64decode(image_data_node.find('./Data').text)
                tmp_result = numpy.frombuffer(decoded, dtype=dtype)
                result[:, plane_index]  = tmp_result
                if __name__ == "__main__":
                    if tiff is None:
                        tiff = TiffIO.TiffIO(node_name + ".tif", "wb")
                    else:
                        tiff = TiffIO.TiffIO(node_name + ".tif", "rb+")
                tmp_result.shape = height, width
                tiff.writeImage(tmp_result, info={"Title":"Plane%d" % plane_index})
            tiff = None
            result.shape = height, width, 3
            if __name__ == "__main__":
                if name:
                    tiff = TiffIO.TiffIO(name + ".tif", "wb")
                    tiff.writeImage(result, info={"Title":container.attrib.get("Name", name)})
                    tiff = None
            # express in seconds
            images.append({"image":result,
                           "image_name":name,
                           "xScale": [0.0, x_calibration],
                           "yScale": [0.0, y_calibration]})
    return images

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
                result = True
        else:
            if eight_chars == "AAMVHFSS":
                if owner:
                    fid.close()
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
    print("total counts = ", stack.data.sum())
    print(stack.info)

