#/*##########################################################################
# Copyright (C) 2004-2008 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem for you.
#############################################################################*/
import specfile
import os
import string
import numpy.oldnumeric as Numeric
try:
    import SPXFileParser
    SPX = True
except:
    SPX = False
DEBUG = 0

def Specfile(filename):
    if os.path.exists(filename):
        f = open(filename)
    else:
        return None
    line0  = f.readline()
    line = line0
    while(len(line)):
        if len(line) > 1:
            if line[0:2] == '#S':
                break
        line = f.readline()
    f.close()
    amptek = False
    qxas   = False
    if len(line):
        #it is a Specfile
        output=specfile.Specfile(filename)
    elif SPX and filename.upper().endswith("SPX"):
        #spx file
        output = SPXFileParser.SPXFileParser(filename)
    else:
        #print "this does not look as a specfile"
        if len(line0) > 7:
            if line0.startswith('$SPEC_ID') or\
               line0.startswith('$DATE_MEA') or\
               line0.startswith('$MEAS_TIM'):
                qxas = True
        if len(line0) >2:
            if line0[0:2] == '<<':
                amptek = True            
        output=specfilewrapper(filename, amptek=amptek, qxas = qxas)
    return output

class specfilewrapper:
    def __init__(self,filename,amptek=None, qxas = None):
        if amptek is None: amptek = False
        if qxas   is None: qxas   = False
        self.amptek = amptek
        self.qxas   = qxas
        self.header = []
        if self.qxas:
            f = open(filename)
        else:
            f = BufferedFile(filename)
        line = f.readline()
        outdata = []
        ncol0 = -1
        nlines= 0
        if amptek:
            while "<<DATA>>" not in line:
                self.header.append(line.replace("\n",""))
                line = f.readline()
        elif qxas:
            line.replace("\n","")
            line.replace("\x00","")
            self._qxasHeader = {}
            self._qxasHeader['S'] = '#S1 '+ " Unlabelled Spectrum"
            while 1:
                self.header.append(line)
                if line.startswith('$SPEC_ID:'):
                    line = f.readline().replace("\n","")
                    line.replace("\x00","")
                    self.header.append(line)
                    self._qxasHeader['S'] = '#S1 '+ line
                if line.startswith('$DATE_MEA'):
                    line = f.readline().replace("\n","")
                    self.header.append(line)
                    self._qxasHeader['D'] = line                    
                if line.startswith('$MEAS_TIM'):
                    line = f.readline().replace("\n","")
                    self.header.append(line)
                    preset, elapsed = [int(i) for i in line.split()]
                    self._qxasHeader['@CTIME'] = ['#@CTIME %f %f %f' % (preset, preset, elapsed)]                     
                if line.startswith('$DATA:'):
                    line = f.readline().replace("\n","")
                    self.header.append(line)
                    start, stop = [int(i) for i in line.split()]
                    self._qxasHeader['@CHANN'] = ['#@CHANN  %d  %d  %d  1' % (stop-start+1, start, stop)]
                    break
                line = f.readline().replace("\n","")
        if qxas:
            outdata = [float(x) for x in f.read().split()]
            nlines = len(outdata)
            f.close()
            self.data=Numeric.resize(Numeric.array(outdata).astype(Numeric.Float),(nlines,1))
        else:
            line = line.replace(",","  ")
            line = line.replace(";","  ")
            line = line.replace("\t","  ")
            line = line.replace("\r","\n")
            line = line.replace('"',"")
            line = line.replace('\n\n',"\n")
            while(len(line)):
                values = string.split(line)
                if len(values):
                    try:
                        reals = map(float,values)
                        ncols = len(reals)
                        if ncol0 < 0:ncol0 = ncols
                        if ncols == ncol0:
                            outdata.append(reals)
                            nlines += 1                    
                    except:
                        if len(line) > 1:
                            self.header.append(line.replace("\n",""))
                else:
                    if len(line) > 1:
                        self.header.append(line.replace("\n",""))
                line = f.readline()
                line = line.replace(",","  ")
                line = line.replace(";","  ")
                line = line.replace("\t","  ")
                line = line.replace("\r","\n")
                line = line.replace('"',"")
                line = line.replace('\n\n',"\n")
            f.close()
            self.data=Numeric.resize(Numeric.array(outdata).astype(Numeric.Float),(nlines,ncol0))
        if self.amptek:
            self.scandata=[myscandata(self.data,'MCA','1.1',scanheader=self.header)]
        elif self.qxas:
            self.scandata=[myscandata(self.data,'MCA','1.1',scanheader=self.header, qxas=self._qxasHeader)]
        else:
            labels = None
            if len(self.header) > 0:
                if len(self.header[0]) > 0:
                    labels = self.header[0].split("  ")
                    if len(labels) != ncol0:
                        labels = None
            self.scandata=[myscandata(self.data,'SCAN','1.1', labels=labels),myscandata(self.data,'MCA','2.1')]

    def list(self):
        if self.amptek or self.qxas:
            return "1:1"
        else:
            return "1:2"
        
    def __getitem__(self,item):
        return self.scandata[item]
        
    def select(self,i):
        n=string.split(i,".")
        return self.__getitem__(int(n[0])-1)
        
    def scanno(self):
        if self.amptek or self.qxas:
            return 1
        else:
            return 2

class myscandata:
    def __init__(self,data,scantype=None,identification=None, scanheader=None, qxas=None, labels=None):
        if identification is None:identification='1.1'
        if scantype is None:scantype='SCAN'
        self.qxas = qxas
        self.scanheader = scanheader
        #print Numeric.shape(data)
        (rows, cols) = Numeric.shape(data)
        if scantype == 'SCAN':
            self.__data = Numeric.zeros((rows, cols +1 ), Numeric.Float)
            self.__data[:,0] = Numeric.arange(rows) * 1.0
            self.__data[:,1:] = data * 1
            self.__cols = cols + 1
            self.labels = ['Point']
        else:
            self.__data = data
            self.__cols = cols
            self.labels = []
        self.scantype = scantype
        self.rows = rows
        if labels is None:
            for i in range(cols):
                self.labels.append('Column %d'  % i)
        else:
            for label in labels:
                self.labels.append('%s' % label)
        n = string.split(identification,".")
        self.__number = int(n[0])
        self.__order  = int(n[1])

    def alllabels(self):
        if self.scantype == 'SCAN':
            return self.labels
        else:
            return []

    def allmotorpos(self):
        return []
        
    def cols(self):
        return self.__cols
    
    def command(self):
        if DEBUG:
            print "command called"
        if self.qxas is not None:
            if self.qxas.has_key('S'):
                text = self.qxas['S']
        elif self.scanheader is not None:
            if len(self.scanheader):
                text = self.scanheader[0]
        return text
        
    def data(self):
        return Numeric.transpose(self.__data)
    
    def datacol(self,col):
        return self.__data[:,col]
        
    def dataline(self,line):
        return self.__data[line,:]
        
    
    def date(self):
        text = 'sometime'
        if self.qxas is not None:
            if self.qxas.has_key('D'):
                return self.qxas['D']
        elif self.scanheader is not None:
            for line in self.scanheader:
                if 'START_TIME' in line:
                    text = "%s" % line
                    break
        return text
            
    def fileheader(self):
        if DEBUG:
            print "file header called"
        labels = '#L '
        for label in self.labels:
            labels += '  '+label
        if self.scantype == 'SCAN':
            return ['#S1 Unknown command','#N %d' % self.cols,labels] 
        else:
            if self.scanheader is None:
                return ['#S1 Unknown command']
            else:
                if DEBUG:
                    print "returning ",self.scanheader
                return self.scanheader
    
    def header(self,key):
        if self.qxas is not None:
            if self.qxas.has_key(key):   return self.qxas[key]
            elif key == "" or key == " ":return self.fileheader()
        if   key == 'S': return self.fileheader()[0]
        elif key == 'N':return self.fileheader()[-2]
        elif key == 'L':return self.fileheader()[-1]
        elif key == '@CALIB':
            output = []
            if self.scanheader is None: return output
            if self.scanheader[0][0:2] == '<<':
                #amptek
                try:
                    if self.scanheader[-5][0:5] == 'LABEL':
                        values = string.split(self.scanheader[-4])
                        x0,y0 = map(float,values)
                        values = string.split(self.scanheader[-3])
                        x1,y1 = map(float,values)
                        gain = (y1-y0)/(x1-x0)
                        zero = y0 - gain * x0
                        output = ['#@CALIB  %g  %g  0' % (zero, gain)]
                except:
                    pass
            return output
        elif key == "" or key == " ":return self.fileheader()
        else:
            #print "requested key = ",key 
            return []
    
    def order(self):
        return self.__order
        
    def number(self):
        return self.__number
    
    def lines(self):
        if self.scantype == 'SCAN':
            return self.rows
        else:
            return 0
            
    def nbmca(self):
        if self.scantype == 'SCAN':
            return 0
        else:
            return self.__cols
        
    def mca(self,number):
        return self.__data[:,number-1]

class BufferedFile:
    def __init__(self, filename):
        f = open(filename, 'r')
        self.__buffer = f.read()
        f.close()
        self.__buffer=self.__buffer.replace("\r", "\n")
        self.__buffer=self.__buffer.replace("\n\n", "\n")
        self.__buffer = self.__buffer.split("\n")
        self.__currentLine = 0

    def readline(self):
        if self.__currentLine >= len(self.__buffer):
            return ""
        line = self.__buffer[self.__currentLine] + "\n"
        self.__currentLine += 1
        return line

    def close(self):
        self.__currentLine = 0
        return
            
if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    print filename
    sf=Specfile(filename)
    sf.list()
    print dir(sf[0])
