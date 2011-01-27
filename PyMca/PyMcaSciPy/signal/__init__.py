try:
    from median import *
except ImportError:
    from .median import medfilt2d, medfilt1d
