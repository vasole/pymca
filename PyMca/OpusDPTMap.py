import DataObject
import numpy
import specfilewrapper as specfile

DEBUG = 0
SOURCE_TYPE="SpecFileStack"

class OpusDPTMap(DataObject.DataObject):
    def __init__(self, filename):
        DataObject.DataObject.__init__(self)
        sf = specfile.Specfile(filename)
        scan = sf[1]
        data = scan.data()
        nMca, nchannels = data.shape
        nMca = nMca - 1
        xValues = data[0,:] * 1
        xValues.shape = -1
        if 0:
            self.data = numpy.zeros((nMca, nchannels),numpy.float32)
            self.data[:,:] = data[1:,:]
            self.data.shape = 1, nMca, nchannels
        else:
            self.data = data[1:,:]
            self.data.shape = 1, nMca, nchannels
        data = None
        
        #perform a least squares adjustment to a line
        x = numpy.arange(nchannels).astype(numpy.float32)
        Sxy = numpy.dot(x, xValues.T)
        Sxx = numpy.dot(x, x.T)
        Sx  = x.sum()
        Sy  = xValues.sum()
        d = nchannels * Sxx - Sx * Sx
        zero = (Sxx * Sy - Sx * Sxy)/d
        gain = (nchannels * Sxy - Sx * Sy)/d

        #and fill the requested information to be identified as a stack
        self.info['SourceName'] = [filename]
        self.info["SourceType"] = "SpecFileStack"
        self.info["Size"]       = 1, nMca, nchannels
        self.info["NumberOfFiles"] = 1
        self.info["FileIndex"] = 0
        self.info["McaCalib"] = [zero, gain, 0.0]
        self.info["Channel0"] = 0.0
        
if __name__ == "__main__":
    import sys
    filename = None
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if filename is not None:
        DEBUG = 1
        w = OpusDPTMap(filename)
    else:
        print("Please supply input filename")
