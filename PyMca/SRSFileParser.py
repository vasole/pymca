import sys
import os
import numpy
from PyMca import SpecFileAbstractClass

DEBUG = 0

class BufferedFile(object):
    def __init__(self, filename):
        f = open(filename, 'rb')
        self.__buffer = f.read()
        f.close()
        if sys.version < '3.0':
            self.__buffer=self.__buffer.replace("\r", "\n")
            self.__buffer=self.__buffer.replace("\n\n", "\n")
            self.__buffer = self.__buffer.split("\n")
        else:
            tmp = bytes("\n", 'utf-8')
            self.__buffer=self.__buffer.replace(bytes("\r", 'utf-8'), tmp)
            self.__buffer=self.__buffer.replace(bytes("\n\n", 'utf-8'), tmp)
            self.__buffer = self.__buffer.split(tmp)
        self.__currentLine = 0

    if sys.version < '3.0':
        def readline(self):
            if self.__currentLine >= len(self.__buffer):
                return ""
            line = self.__buffer[self.__currentLine] + "\n"
            self.__currentLine += 1
            return line
    else:
        def readline(self):
            if self.__currentLine >= len(self.__buffer):
                return bytes("", 'utf-8')
            line = self.__buffer[self.__currentLine] + bytes("\n", 'utf-8')
            self.__currentLine += 1
            return str(line, 'utf-8')

    def close(self):
        self.__currentLine = 0
        return

class SRSFileParser(object):
    def __init__(self, filename, sum_all=False):
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
            if readingMetaData:
                if '</MetaDataAtStart>' in line:
                    readingMetaData = False
                elif '=' in line:
                    key, value = line[:-1].split('=')
                    if 'datestring' in key:
                        header.append('#D %s' % value)
                    elif 'scancommand' in key:
                        header[0] = '#S 1 %s' % value
                    else:
                        self.motorNames.append(key)
                        self.motorValues.append(value)
            elif '<MetaDataAtStart>' in line:
                readingMetaData = True
            elif '&END' in line:
                endReached = True
            elif endReached:
                if readingData:
                    tmpLine = line[:-1].replace("\t", "  ").split("  ")
                    data.append([float(x) for x in tmpLine])
                else:
                    labels = line[:-1].replace("\t", "  ").split("  ")
                    readingData = True
            else:
                if DEBUG:
                    print("Unhandled line %s" % line[:-1])
            line = _fileObject.readline()
        header.append("#N %d" % len(labels))
        txt = "#L "
        for label in labels:
            txt += "  %s" % label
        header.append(txt + "\n")
        data = numpy.array(data)

        #create an abstract scan object
        self._scan = [SRSScan(data,
                              scantype='SCAN',
                              scanheader=header,
                              labels=labels,
                              motor_values=self.motorValues,
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
        return "1:1"

    def select(self, key):
        """
        key is of the from s.o
        scan number, scan order
        """
        n = key.split(".")
        return self.__getitem__(int(n[0])-1)

    def allmotors(self):
        return self.motorNames

class SRSScan(SpecFileAbstractClass.SpecFileAbstractScan):
    def __init__(self, data, scantype='MCA',
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

def isSRSFile(filename):
    f = open(filename, mode = 'rb')
    try:
        if sys.version < '3.0':
            if '&SRS' in f.readline():
                f.close()
                return True
        else:
            if '&SRS' in str(f.readline(), 'utf-8'):
                f.close()
                return True
    except:
        f.close()
        pass
    
    return False

def test(filename):
    if isSRSFile(filename):
        sf=SRSFileParser(filename)
    else:
        print("Not a SRS File")
    print(sf[0].header('S'))
    print(sf[0].header('D'))
    print(sf[0].alllabels())
    print(sf[0].allmotors())


if __name__ == "__main__":
    test(sys.argv[1])
