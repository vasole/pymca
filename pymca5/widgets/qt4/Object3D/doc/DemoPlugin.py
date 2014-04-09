from Object3D import Object3DBase
from OpenGL.GL import *

class Object3DDemo(Object3DBase.Object3D):
    def __init__(self, name="Demo"):
        Object3DBase.Object3D.__init__(self, name)
        
        # I have to give the limits I am going to use in order
        # to calculate a proper bounding box
        self.setLimits(-1.0, 0.0, 0.0, 1.0, 1.0, 0.0)
    
    def drawObject(self):
        # this is to handle transparency
        alpha = 1. - self._configuration['common']['transparency']

        #some simple drawing
        glShadeModel(GL_SMOOTH)
        glBegin(GL_TRIANGLES)
        glColor4f(  1.0, 0.0, 0.0, alpha)
        glVertex3f(-1.0, 0.0, 0.0)
        glColor4f(  0.0, 1.0, 0.0, alpha)
        glVertex3f( 0.0, 1.0, 0.0)
        glColor4f(  0.0, 0.0, 1.0, alpha)
        glVertex3f( 1.0, 0.0, 0.0)
        glEnd()

MENU_TEXT = 'Demo'
def getObject3DInstance(config=None):
    return Object3DDemo()
