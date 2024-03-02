from tiled.client import from_uri,from_profile

from databroker.queries import TimeRange

##fixme ... needs arg to connect to corret service
#def get_tiled_connection():
#    return from_uri("https://tiled-demo.blueskyproject.io")

class TiledAdaptor(object):
    @classmethod
    def get_nested(cls,d, keys):
        #FIXME to be done more efficient
        for k in keys:
            d = d[k]
        return d

    def __init__(self,host,prefix=None):
        print("init TiledAdaptor",host)
        if "http" in host:
            self._client = from_uri(host)
        else:
            self._client = from_profile(host)
        
        #specific hack for shen beamtime
        if host=="opls":
            #specific hack for shen beamtime, just to reduce the scope
            from tiled.queries import FullText
            self._client = self._client.search(TimeRange(since='2024-03-02 01:00', until='2024-03-02 02:00'))
        elif prefix:
            self._client = TiledAdaptor.get_nested(self._client,prefix.split("/"))

        self.__prefix=prefix
        print(self._client)

    def get_node(self,path=None,scan_id=None):
        
        if not scan_id is None:
            if path and "/" in path:
                return self.get_nested(self._client[scan_id],path.split("/"))
            elif not path:
                return self._client[scan_id]
            else:   
                return self._client[scan_id][path]
        else:
            if "/" in path:
                return self.get_nested(self._client,path.split("/"))
            elif not path:
                return self._client
            else:
                return self._client[path]
        

    @property
    def client(self):
        return self._client
    
    # @property
    # def _sourceName(self):
    #     return self.__sourceName
    
    # @_sourceName.setter
    # def _sourceName(self,sn):
    #     self.__sourceName = sn

    # def close(self):
    #     #not sure if there is anything that needs to happen here
    #     return
    
    # def getSourceInfo(self):
    #     """
    #     Returns a dictionary with the key "KeyList" (list of all available keys
    #     in this source). Each element in "KeyList" has the form 'n1.n2' where
    #     n1 is the source number and n2 entry number in file both starting at 1.
    #     """
    #     print("TiledAdaptor getSourceInfo")
    #     return {"KeyList":["1.1"]}
    
    # @property
    # def name(self):
    #     return "/"
    
    # @property
    # def filename(self):
    #     return "toto"

#very simplistic and "bad" coding ... just going global tiled connection for now

#just to be a little quicker ... only work against fxi first   
_TILED_CLIENT_opls=TiledAdaptor("opls")
_TILED_CLIENT_fxi=TiledAdaptor("https://tiled-demo.blueskyproject.io","fxi/raw")


def get_sessions_list():
    return ["opls","demo:fxi"]

def get_node(path):
    #path ... to be resolved from tiled...
    if path=="opls":
        return _TILED_CLIENT_opls
    if path=="demo:fxi":
        return _TILED_CLIENT_fxi
    else:
        print(">>> trying to access: ",path)
        raise