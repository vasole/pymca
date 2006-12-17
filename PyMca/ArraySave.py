import os

def save2DArrayListAsASCII(datalist, filename, labels = None):
    if type(datalist) != type([]):
        datalist = [datalist]
    r, c = datalist[0].shape
    ndata = len(datalist)
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except:
            pass
    if labels is None:
        labels = []
        for i in range(len(datalist)):
            labels.append("Array_%d" % i) 
    if len(labels) != len(datalist):
        raise "ValueError", "Incorrect number of labels"
    header = "row  column"
    for label in labels:
        header +="  %s" % label
    filehandle=open(filename,'w+')
    filehandle.write('%s\n' % header)
    fileline=""
    for row in range(r):
        for col in range(c):
            fileline += "%d" % row
            fileline += "  %d" % col
            for i in range(ndata):
                fileline +="  %g" % datalist[i][row, col]
            fileline += "\n"
            filehandle.write("%s" % fileline)
            fileline =""
    filehandle.write("\n") 
    filehandle.close()

