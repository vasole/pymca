import Numeric
import copy

class DataObject:
    def __init__(self):
        self.info = {}
        self.data = Numeric.array([])

    def getInfo(self):
        return self.info
    
    def getData(self):
        return self.data 
        
    def select(self,selection=None):
        if selection is None:
            return copy.deepcopy(self.data)
        else:
            print "Not implemented (yet)"
            #it will be a new array
            return copy.deepcopy(self.data)
