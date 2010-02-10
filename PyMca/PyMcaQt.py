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


class HorizontalSpacer(QWidget):
    def __init__(self, *args):
        QWidget.__init__(self, *args)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding,
                                          QSizePolicy.Fixed))

class VerticalSpacer(QWidget):
    def __init__(self, *args):
        QWidget.__init__(self, *args)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,
                                          QSizePolicy.Expanding))
