import sys
import os
import numpy
from PyMca import MEDFile
from PyMca import SpecFileAbstractClass

class APSMEDFileParser(object):
    def __init__(self, filename, sum_all=False):
        if not os.path.exists(filename):
            raise IOError("File %s does not exists"  % filename)

        self._medFileObject = MEDFile.MEDFile(filename)
        #read the data
        #potentially each detector could have a different number of channels
        #should I support that?
        header = []
        header.append('#S 1  %s Unknown command' % os.path.basename(filename))
        header.append('#D %s' % self._medFileObject.mcas[0].start_time)

        data = []
        for mca in self._medFileObject.mcas:
            header.append('#@CTIME %f %f %f' % (mca.realtime,
                                                mca.realtime,
                                                mca.livetime))
            header.append('#@CALIB %f %f %f' % (mca.cal_offset,
                                                mca.cal_slope,
                                                mca.cal_quad))
            data.append(mca.data)

        self.motorNames  = []
        motorValues = []
        for item in self._medFileObject.env:
            name, value  = item.split("=")
            self.motorNames.append(name)
            motorValues.append(value)
            header.append('#'+item)
                    
        #create an abstract scan object
        self._scan = [APSMEDScan(data, scanheader=header,
                                 motor_values=motorValues)]

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

class APSMEDScan(SpecFileAbstractClass.SpecFileAbstractScan):
    def __init__(self, data, scantype='MCA',
                 identification="1.1", scanheader=None, labels=None,
                 motor_values=None):
        SpecFileAbstractClass.SpecFileAbstractScan.__init__(self,
                    data, scantype=scantype, identification=identification,
                    scanheader=scanheader, labels=labels)
        if motor_values is None:
            motor_values = []
        self.motorValues = motor_values

    def allmotorpos(self):
        return self.motorValues

def isAPSMEDFile(filename):
    #Obviously I should put a better test than this one
    if not filename.upper().endswith(".XRF"):
        return False
    return True

def test(filename):
    if isAPSMEDFile(filename):
        sf=APSMEDFileParser(filename)
    else:
        print("Not a Fit2D .Chi File")
    print(sf[0].header('S'))
    print(sf[0].header('D'))
    print(sf[0].header('ID13ds'))
    print(sf[0].alllabels())
    print(dir(sf[0]))
    print("number of mcas = %s " % sf[0].nbmca())
    try:
        import pylab
        for i in range(sf[0].nbmca()):
            pylab.plot(sf[0].mca(i+1))
        pylab.show()
    except ImportError:
        pass


if __name__ == "__main__":
    test(sys.argv[1])
