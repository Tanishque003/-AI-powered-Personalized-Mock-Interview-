from OpenGL.GL import glGetString, GL_VERSION
import pygame
from pygame.locals import *

pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
version = glGetString(GL_VERSION)
print(f"OpenGL version: {version.decode('utf-8')}")
pygame.quit()

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBegin(GL_TRIANGLES)
    glVertex3f(-0.5, -0.5, 0)
    glVertex3f(0.5, -0.5, 0)
    glVertex3f(0, 0.5, 0)
    glEnd()
    glutSwapBuffers()

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutCreateWindow(b"OpenGL Test")
    glutDisplayFunc(draw)
    glutMainLoop()

if __name__ == "__main__":
    main()

