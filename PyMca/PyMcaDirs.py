import sys
import os

DEBUG = 0
inputDir  = None
outputDir = None
nativeFileDialogs = True

class __ModuleWrapper:
  def __init__(self, wrapped):
    self.__dict__["_ModuleWrapper__wrapped"] = wrapped

  def __getattr__(self, name):
    if DEBUG: print "getting ", name
    if name == "inputDir":
        if self.__wrapped.__dict__[name] is None:
            if self.__wrapped.__dict__['outputDir'] is not None:
                value = self.__wrapped.__dict__['outputDir']
            else:
                value = os.getcwd()
            if not os.path.isdir(value):
                value = os.getcwd()
            self.__setattr__('inputDir', value)
    elif name == "outputDir":
        if self.__wrapped.__dict__[name] is None:
            if self.__wrapped.__dict__['inputDir'] is not None:
                value = self.__wrapped.__dict__['inputDir']
            else:
                value = os.getcwd()
            if not os.path.isdir(value):
                value = os.getcwd()
            self.__setattr__('outputDir', value)
    if DEBUG:print "got ", name, getattr(self.__wrapped, name)
    return getattr(self.__wrapped, name)

  def __setattr__(self, name, value):
    if DEBUG: print "setting ", name, value
    if name == "inputDir":
        if os.path.isdir(value):
            self.__wrapped.__dict__[name]=value
        else:
            if not len("%s" % value):
                self.__wrapped.__dict__[name] = os.getcwd()
            else:  
                raise ValueError, "Non existing directory %s" % value
    elif name == "outputDir":
        if os.path.isdir(value):
            self.__wrapped.__dict__[name]=value
        else:
            if not len("%s" % value):
                self.__wrapped.__dict__[name] = os.getcwd()
            else:  
                raise ValueError, "Non existing directory %s" % value
    elif name == "nativeFileDialogs":
        self.__wrapped.__dict__[name]=value
    elif name.startswith("__"):
        self.__dict__[name]=value
    else:
        raise AttributeError, "Invalid attribute %s" % name
        #self.__wrapped.__dict__[name]=value

sys.modules[__name__]=__ModuleWrapper(sys.modules[__name__])


