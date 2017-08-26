import os
import math
import numpy as np
from kivy.app import App
from kivy.graphics.context_instructions import Color
from kivy.graphics.texture import Texture
from kivy.graphics.vertex_instructions import Line, Ellipse, Rectangle
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.uix.popup import Popup
from src.input.util import Util


class Picture(FloatLayout):
    texture = ObjectProperty(None)
    textureSize = ObjectProperty(None)
    pos = ObjectProperty(None)
    img = ObjectProperty(None)
    X = NumericProperty(0)
    Y = NumericProperty(0)
    value = NumericProperty(0)

    def on_touch_down(self, touch):
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        if touch.x >= 0 and touch.x < rows and touch.y >= 0 and \
                        touch.y < cols:
            # if self.collide_point(*touch.pos):
            self.X = math.floor(touch.x)
            self.Y = cols - math.floor(touch.y) - 1
            # Take element (y, x) from matrix
            self.value = int(self.img[self.Y, self.X])
            touch.ud['origin'] = (touch.x, touch.y)

    def on_touch_move(self, touch):
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        if touch.x >= 0 and touch.x < rows and touch.y >= 0 and \
                        touch.y < cols:
            originX = touch.ud['origin'][0]
            originY = touch.ud['origin'][1]
            height = math.fabs(touch.y - originY)
            base = math.fabs(touch.x - originX)
            pos = self.calculate_rectangle_pos(touch, originX, originY, touch.x, touch.y)

            self.canvas.remove_group('Awesome')
            with self.canvas:
                Color(1, 1, 1, 0.3)
                Rectangle(source='../resources/highlight.jpg', pos=(pos[0], pos[1]), size=(base, height),
                          group='Awesome')

    def calculate_rectangle_pos(self, touch, originX, originY, endX, endY):
        height = math.fabs(endY - originY)
        base = math.fabs(endX - originX)
        pos = (originX, originY)

        if endX > originX:
            if endY < originY:
                pos = (originX, originY - height)
        else:
            if endY > originY:
                pos = (originX - base, originY)
            else:
                pos = (endX, endY)
        return pos

    def on_touch_up(self, touch):
        self.canvas.remove_group('Awesome')


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

        # Use for Loading lena.ascii.pbm directly skipping Filechooser
        # source = '../resources/lena.ascii.pbm'
        # img = Util.load_gray_scale(source)
        # self.draw_gray_scale(img)

    def load(self, path, filename):
        source = os.path.join(path, filename[0])
        img = Util.load_gray_scale(source)
        self.draw_gray_scale(img)
        self.dismiss_popup()

    def draw_gray_scale(self, img):
        texture = Texture.create(size=img.shape, colorfmt='luminance')
        # Reverse the rows order (Because the (0,0) pixel from the matrix will be drawn in the
        # bottom-left border of the window)
        reversed_texture = np.flipud(img)
        buffer = reversed_texture.ravel()
        texture.blit_buffer(buffer, colorfmt='luminance', bufferfmt='ubyte')
        picture = Picture(img=img, texture=texture, pos=self.pos, textureSize=img.shape)
        self.add_widget(picture)


class ImageLoader(App):
    pass


if __name__ == '__main__':
    ImageLoader().run()
