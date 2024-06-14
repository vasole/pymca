import sys
import os
import numpy
import logging
from PyMca5.PyMcaIO import SpecFileAbstractClass

_logger = logging.getLogger(__name__)

class BufferedFile(object):
    def __init__(self, filename):
        f = open(filename, 'r')
        self.__buffer = f.read()
        f.close()
        self.__buffer = self.__buffer.replace("\r", "\n")
        self.__buffer = self.__buffer.replace("\n\n", "\n")
        self.__buffer = self.__buffer.split("\n")
        self.__currentLine = 0

    def readline(self):
        if self.__currentLine >= len(self.__buffer):
            return ""
        line = self.__buffer[self.__currentLine]
        self.__currentLine += 1
        return line

    def close(self):
        self.__buffer = [""]
        self.__currentLine = 0
        return

class TASSpecFileParser(object):
    def __init__(self, filename, sum_all=False):
        _logger.debug("Starting TAS Spec file parsing")
        if not os.path.exists(filename):
            raise IOError("File %s does not exists"  % filename)
        _fileObject = BufferedFile(filename)

        #Only one measurement per file
        header = []
        header.append('#S 1  %s Unknown command' % os.path.basename(filename))

        #read the data
        line = _fileObject.readline()
        self.motorNames = []
        self.motorValues = []
        readingMetaData = False
        endReached = False
        readingData = False
        data = []
        while len(line)>1:
            if not readingData:
                header.append(line[:-1])
            #if readingMetaData:
            #    if '</MetaDataAtStart>' in line:
            #        readingMetaData = False
            ### Can probably use this block for TAS 
            ###    elif '=' in line:
            ###        key, value = line[:-1].split('=')
            ###        if 'datestring' in key:
            ###            header.append('#D %s' % value)
            ###        elif 'scancommand' in key:
            ###            header[0] = '#S 1 %s' % value
            ###        else:
            ###            self.motorNames.append(key)
            ###            self.motorValues.append(value)
            #elif '<MetaDataAtStart>' in line: ### Can replace this with "proposal"
            #    readingMetaData = True
            #elif '&END' in line:
            #    endReached = True
            #elif endReached:
                if readingData:
                    tmpLine = line[:-1].replace("\t", "  ").split("  ")
                    data.append([float(x) for x in tmpLine])
                else:
                    labels = line[:-1].replace("\t", "  ").split("  ")
                    readingData = True
            else:
                _logger.debug("Unhandled line %s", line[:-1])
            line = _fileObject.readline()
        header.append("#N %d" % len(labels))
        txt = "#L "
        for label in labels:
            txt += "  %s" % label
        header.append(txt + "\n")
        data = numpy.array(data)

        #create an abstract scan object
        self._scan = [TASSpecFileScan(data,
                              scantype='SCAN',
                              scanheader=header,
                              labels=labels,
                              #motor_values=self.motorValues, # what is this used for?
                              point=False)]
        _fileObject = None
        data = None

        #the methods below are called by PyMca on any SPEC file

    def __getitem__(self, item):
        return self._scan[item]

    def scanno(self):
        """
        Gives back the number of scans in the file
        """
        return len(self_scan)

    def list(self):
        return "1:1" #?

    def select(self, key):
        """
        key is of the from s.o
        scan number, scan order
        """
        n = key.split(".")
        return self.__getitem__(int(n[0])-1)

    def allmotors(self):
        return self.motorNames

class TASSpecFileScan(SpecFileAbstractClass.SpecFileAbstractScan):
    def __init__(self, data, scantype='SCAN',
                 identification="1.1", scanheader=None, labels=None,
                 motor_values=None,point=False):
        SpecFileAbstractClass.SpecFileAbstractScan.__init__(self,
                    data, scantype=scantype, identification=identification,
                    scanheader=scanheader, labels=labels,point=point)
        if motor_values is None:
            motor_values = []
        self.motorValues = motor_values

    def allmotorpos(self):
        return self.motorValues

def isTASSpecFile(filename):
    _logger.debug("Checking if the file is TAS Spec")
    f = open(filename, mode = 'r')
    line0 = f.readline()
    if line0[0:2] == '# ':
        f.close()
        return True
    else:
        f.close()
        return False

def test(filename):
    if isTASSpecFile(filename):
        sf=TASSpecFileParser(filename)
    else:
        print("Not a TAS Spec File")
    print(sf[0].header('S'))
    print(sf[0].header('D'))
    print(sf[0].alllabels())
    print(sf[0].allmotors())


if __name__ == "__main__":
    test(sys.argv[1])
