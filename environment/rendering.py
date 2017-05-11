# -*- encoding = utf8 -*-
from pyglet.gl import *


class Viewer:
    
    def __init__(self, width=400, height=400):
        self.window = pyglet.window.Window(width=width, height=height)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
    def render(self, geometories, is_save=False, path=None):
        glClearColor(1,1,1,1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        for geometroy in geometories:
            geometroy.render()
        if is_save:
            pyglet.image.get_buffer_manager().get_color_buffer().save(path)
        else:
            self.window.flip()
        
    def draw_floor(self, x, y, height, width):
        floor = Block([[x,y], [x+width, y], [x+width, y+height], [x, y+height]])
        floor.set_fill_color(1.0, 1.0, 1.0) # fill white
        floor.set_line_color(0.0, 0.0, 0.0) # line black
        floor.render()
        
    def _draw_wall(self):
        glColor4f(0.0, 0.0, 0.0, 1.0)
        glBegin(GL_POLYGON)
        # down
        glVertex2f(0, 0)
        glVertex2f(self.window.width, 0)
        glVertex2f(self.window.width, 10)
        glVertex2f(0, 10)
        glEnd()
        # up
        glBegin(GL_POLYGON)
        glVertex2f(0, self.window.height-10)
        glVertex2f(self.window.width, self.window.height-10)
        glVertex2f(self.window.width, self.window.height)
        glVertex2f(0, self.window.height)
        glEnd()
        
        
class Geometory:
    
    def __init__(self):
        self._color = (0, 0, 0, 1.0)
        
    def render(self):
        raise NotImplemented
    
    def set_color(self, r, g, b):
        self._color = (r, g, b, 1)
        

class Block(Geometory):
    
    def __init__(self, lefttop, height, width):
        Geometory.__init__(self)
        vertexs = [[lefttop[0], lefttop[1]], [lefttop[0]+height, lefttop[1]], \
                   [lefttop[0]+height, lefttop[1]+width], [lefttop[0], lefttop[1]+width]]
        self.vertexs = vertexs
        self._line_color = (0.0, 0.0, 0.0, 1.0)
        self._fill_color = (1.0, 1.0, 1.0, 1.0)
        
    def set_line_color(self, r, g, b):
        if r <= 1 and g <= 1 and b <= 1:
            self._line_color = (r, g, b, 1.0)
        else:
            self._line_color = (r/255.0, g/255.0, b/255.0, 1.0)
        
    def set_fill_color(self, r, g, b):
        if r <= 1 and g <= 1 and b <= 1:
            self._fill_color = (r, g, b, 1.0)
        else:
            self._fill_color = (r/255.0, g/255.0, b/255.0, 1.0)
        
    def render(self):
        # fill
        glColor4f(*self._fill_color)
        glBegin(GL_POLYGON)
        for vertex in self.vertexs:
            glVertex2f(vertex[0], vertex[1])
        glEnd()
        # line
        glColor4f(*self._line_color)
        for i in range(len(self.vertexs)-1):
            glBegin(GL_LINES)
            glVertex2f(*self.vertexs[i])
            glVertex2f(*self.vertexs[i+1])
            glEnd()
        glBegin(GL_LINES)
        glVertex2f(*self.vertexs[i+1])
        glVertex2f(*self.vertexs[0])
        glEnd()
        
        
class Agent(Geometory):
    
    def __init__(self, center, height, width):
        Geometory.__init__(self)
        vertexs = [[center[0]+width/3.0, center[1]+height/2.0], \
                   [center[0]-width/3.0, center[1]+height/2.0], \
                   [center[0]-2*width/3.0, center[1]], \
                   [center[0]-width/3.0, center[1]-height/2.0], \
                   [center[0]+width/3.0, center[1]-height/2.0], \
                   [center[0]+2*width/3.0, center[1]]]
        self.vertexs = vertexs
        self._fill_color = (1.0, 1.0, 1.0, 1.0)
        
    def set_fill_color(self, r, g, b, alpha=1.0):
        if r <= 1 and g <= 1 and b <= 1:
            self._fill_color = (r, g, b, alpha)
        else:
            self._fill_color = (r/255.0, g/255.0, b/255.0, alpha)
        
    def render(self):
        # fill
        glColor4f(*self._fill_color)
        glBegin(GL_POLYGON)
        for vertex in self.vertexs:
            glVertex2f(vertex[0], vertex[1])
        glEnd()
        

class Triangle(Geometory):
    
    def __init__(self, center, height, width):
        Geometory.__init__(self)
        vertexs = [[center[0], center[1]], [center[0]-width/2, center[1]+height/2], \
                   [center[0]+width/2, center[1]+height/2]]
        self.vertexs_up = vertexs
        vertexs = [[center[0], center[1]], [center[0]-width/2, center[1]-height/2], \
                   [center[0]+width/2, center[1]-height/2]]
        self.vertexs_down = vertexs
        vertexs = [[center[0], center[1]], [center[0]-width/2, center[1]+height/2], \
                   [center[0]-width/2, center[1]-height/2]]
        self.vertexs_left = vertexs
        vertexs = [[center[0], center[1]], [center[0]+width/2, center[1]+height/2], \
                   [center[0]+width/2, center[1]-height/2]]
        self.vertexs_right = vertexs
        self._fill_color = (1.0, 1.0, 1.0, 1.0)
        
    def set_fill_color_up(self, r, g, b, alpha=1.0):
        if r <= 1 and g <= 1 and b <= 1:
            self._fill_color_up = (r, g, b, alpha)
        else:
            self._fill_color_up = (r/255.0, g/255.0, b/255.0, alpha)
        
    def set_fill_color_down(self, r, g, b, alpha=1.0):
        if r <= 1 and g <= 1 and b <= 1:
            self._fill_color_down = (r, g, b, alpha)
        else:
            self._fill_color_down = (r/255.0, g/255.0, b/255.0, alpha)
        
    def set_fill_color_left(self, r, g, b, alpha=1.0):
        if r <= 1 and g <= 1 and b <= 1:
            self._fill_color_left = (r, g, b, alpha)
        else:
            self._fill_color_left = (r/255.0, g/255.0, b/255.0, alpha)
        
    def set_fill_color_right(self, r, g, b, alpha=1.0):
        if r <= 1 and g <= 1 and b <= 1:
            self._fill_color_right = (r, g, b, alpha)
        else:
            self._fill_color_right = (r/255.0, g/255.0, b/255.0, alpha)
        
    def render(self):
        # fill
        glColor4f(*self._fill_color_up)
        glBegin(GL_POLYGON)
        for vertex in self.vertexs_up:
            glVertex2f(vertex[0], vertex[1])
        glEnd()
        glColor4f(*self._fill_color_down)
        glBegin(GL_POLYGON)
        for vertex in self.vertexs_down:
            glVertex2f(vertex[0], vertex[1])
        glEnd()
        glColor4f(*self._fill_color_left)
        glBegin(GL_POLYGON)
        for vertex in self.vertexs_left:
            glVertex2f(vertex[0], vertex[1])
        glEnd()
        glColor4f(*self._fill_color_right)
        glBegin(GL_POLYGON)
        for vertex in self.vertexs_right:
            glVertex2f(vertex[0], vertex[1])
        glEnd()