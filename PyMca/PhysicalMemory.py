import sys
import os
import ctypes
import traceback

def loadCLibrary(name="libc.so"):
    try:
        libc = ctypes.CDLL(name)
    except OSError:
        text = traceback.format_exc()
        if "invalid ELF header" in text:
            library = text.split(": ")
            if len(library) > 1:
                libraryFile = library[1]
                if os.path.exists(libraryFile):
                    # to read some line
                    f = open(libraryFile, 'r')
                    for i in range(10):
                        line = f.readline()
                        if name in line:
                            f.close()
                            lineSplit = line.split(name)
                            libraryFile = lineSplit[0].split()[-1] +\
                                          name+\
                                          lineSplit[1].split()[0]
                            break
                    libraryFile = line[line.index(libraryFile):].split()[0]                    
                    libc = ctypes.CDLL(libraryFile)
        else:
            raise
    return libc

if sys.platform == 'win32':
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

        def __init__(self):
            # have to initialize this to the size of MEMORYSTATUSEX
            self.dwLength = ctypes.sizeof(self)
            super(MEMORYSTATUSEX, self).__init__()

    #print("MemoryLoad: %d%%" % (stat.dwMemoryLoad))
    #print("Physical memory = %d" % stat.ullTotalPhys)
    #print(stat.ullAvailPhys)
    def getPhysicalMemory():
        stat = MEMORYSTATUSEX()
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys

elif sys.platform.startswith('linux'):
    def getPhysicalMemory():
        return os.sysconf('SC_PAGESIZE') * os.sysconf('SC_PHYS_PAGES')
    
elif sys.platform == 'darwin':
    def getPhysicalMemory():
        libc = loadCLibrary("libc.dylib")
        memsize = ctypes.c_uint64(0)
        length = ctypes.c_size_t(ctypes.sizeof(memsize))
        libc.sysctlbyname("hw.memsize", ctypes.byref(memsize), ctypes.byref(length), None, 0)
        return memsize.value
else:
    def getPhysicalMemory():    
        return None

def getPhysicalMemoryOrNone():
    try:
        return getPhysicalMemory()
    except:
        return None

if __name__ == "__main__":
    print("Physical memory = %d" % getPhysicalMemory())
