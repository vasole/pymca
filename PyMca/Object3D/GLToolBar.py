import sys
import Object3DQt as qt
import Object3DIcons
from HorizontalSpacer import HorizontalSpacer

DEBUG = 0

class GLToolBar(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QHBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self.build()

    def build(self):
        IconDict = Object3DIcons.IconDict
        self.cubeFrontIcon = qt.QIcon(qt.QPixmap(IconDict["cube_front"]))
        self.cubeBackIcon = qt.QIcon(qt.QPixmap(IconDict["cube_back"]))
        self.cubeTopIcon = qt.QIcon(qt.QPixmap(IconDict["cube_top"]))
        self.cubeBottomIcon = qt.QIcon(qt.QPixmap(IconDict["cube_bottom"]))
        self.cubeRightIcon = qt.QIcon(qt.QPixmap(IconDict["cube_right"]))
        self.cubeLeftIcon = qt.QIcon(qt.QPixmap(IconDict["cube_left"]))
        self.cube45Icon = qt.QIcon(qt.QPixmap(IconDict["cube_45"]))

        #the buttons
        self.cubeFront = qt.QToolButton(self)
        self.cubeFront.setIcon(self.cubeFrontIcon)

        self.cubeBack = qt.QToolButton(self)
        self.cubeBack.setIcon(self.cubeBackIcon)

        self.cubeTop = qt.QToolButton(self)
        self.cubeTop.setIcon(self.cubeTopIcon)

        self.cubeBottom = qt.QToolButton(self)
        self.cubeBottom.setIcon(self.cubeBottomIcon)

        self.cubeRight = qt.QToolButton(self)
        self.cubeRight.setIcon(self.cubeRightIcon)

        self.cubeLeft = qt.QToolButton(self)
        self.cubeLeft.setIcon(self.cubeLeftIcon)

        self.cube45 = qt.QToolButton(self)
        self.cube45.setIcon(self.cube45Icon)

        #the tool tips
        
        self.cubeFront.setToolTip("See from front (X+)")
        self.cubeBack.setToolTip("See from back (X-)")

        self.cubeTop.setToolTip("See from top (Z+)")
        self.cubeBottom.setToolTip("See from bottom (Z-)")

        self.cubeRight.setToolTip("See from right (Y+)")
        self.cubeLeft.setToolTip("See from left (Y-)")

        self.cube45.setToolTip("See from diagonal ( 1, 1, 1)")

        self.mainLayout.addWidget(self.cubeFront)
        self.mainLayout.addWidget(self.cubeBack)
        self.mainLayout.addWidget(self.cubeTop)
        self.mainLayout.addWidget(self.cubeBottom)
        self.mainLayout.addWidget(self.cubeRight)
        self.mainLayout.addWidget(self.cubeLeft)
        self.mainLayout.addWidget(self.cube45)

        self.connect(self.cubeFront, qt.SIGNAL('clicked()'),
                     self.cubeFrontSlot)

        self.connect(self.cubeBack, qt.SIGNAL('clicked()'),
                     self.cubeBackSlot)

        self.connect(self.cubeTop, qt.SIGNAL('clicked()'),
                     self.cubeTopSlot)

        self.connect(self.cubeBottom, qt.SIGNAL('clicked()'),
                     self.cubeBottomSlot)

        self.connect(self.cubeRight, qt.SIGNAL('clicked()'),
                     self.cubeRightSlot)

        self.connect(self.cubeLeft, qt.SIGNAL('clicked()'),
                     self.cubeLeftSlot)

        self.connect(self.cube45, qt.SIGNAL('clicked()'),
                     self.cube45Slot)

    def cubeFrontSlot(self):
        self.applyCube('front')

    def cubeBackSlot(self):
        self.applyCube('back')

    def cubeTopSlot(self):
        self.applyCube('top')

    def cubeBottomSlot(self):
        self.applyCube('bottom')

    def cubeRightSlot(self):
        self.applyCube('right')

    def cubeLeftSlot(self):
        self.applyCube('left')

    def cube45Slot(self):
        self.applyCube('d45')

    def applyCube(self, cubeFace):
        ddict = {}
        ddict['event'] = 'ApplyCubeClicked'
        ddict['face'] = cubeFace
        self.emit(qt.SIGNAL('GLToolBarSignal'), ddict)
        #print "to be implemented"

if __name__ == "__main__":
    app = qt.QApplication([])
    w = GLToolBar()
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                       app, qt.SLOT("quit()"))
    w.show()
    app.exec_()
