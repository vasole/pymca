import sys
import sip
if 'qt' not in sys.modules:
    try:
        from PyQt4.QtCore import *
        from PyQt4.QtGui import *
        try:
            #In case PyQwt is compiled with QtSvg
            from PyQt4.QtSvg import *
        except:
            pass
    except:
        from qt import *
else:
    from qt import *
