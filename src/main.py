import numpy as np
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.graphics.vertex_instructions import Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
import matplotlib.pyplot as plt
from src.input.util import Util
from src.picture import Picture


class Root(FloatLayout):
    def __init__(self, **kwargs):
        super(Root, self).__init__(**kwargs)

        # Reference to the image matrix
        self.img = None
        self.transformed_img = None
        self.img_pos = (50, 50)
        self.transformed_img_pos = (600, 50)

        # Used for saving images
        self.image_number = 1
        self.selection_number = 1
        self.is_color = False

        # Reference to the kivy Layout
        self.picture = None
        self.transformed_picture = None

        # Button Groups
        self.button_bar = BoxLayout(size_hint=(1, .05), pos_hint={'top': 1})
        self.coordinates = BoxLayout(size_hint=(1, .05), pos_hint={'x': 0}, spacing=5)

        # Buttons
        self.load_button = Button(text='Load')
        self.edit_button = Button(text='Edit')
        self.noise_button = Button(text='Noise')
        self.filter_button = Button(text='Filter')
        self.save_button = Button(text='Save')
        self.edit_drop_down = DropDown()
        self.noise_drop_down = DropDown()
        self.filter_drop_down = DropDown()

        self.load_button.bind(on_release=self.load)
        self.edit_button.bind(on_release=self.edit_drop_down.open)
        self.noise_button.bind(on_release=self.noise_drop_down.open)
        self.filter_button.bind(on_release=self.filter_drop_down.open)
        self.save_button.bind(on_release=self.save)

        self.button_bar.add_widget(self.load_button)
        self.button_bar.add_widget(self.edit_button)
        self.button_bar.add_widget(self.noise_button)
        self.button_bar.add_widget(self.filter_button)
        self.button_bar.add_widget(self.save_button)

        self.add_widget(self.button_bar)

        # Edit Dropdown Buttons
        self.duplicate_btn = Button(text='Duplicate', size_hint=(1, None), height=30)
        self.negative_btn = Button(text='Negative', size_hint=(1, None), height=30)
        self.contrast_btn = Button(text='Contrast', size_hint=(1, None), height=30)
        self.compression_btn = Button(text='Compression', size_hint=(1, None), height=30)
        self.gamma_btn = Button(text='Gamma', size_hint=(1, None), height=30)
        self.histogram_btn = Button(text='Histogram', size_hint=(1, None), height=30)
        self.save_selection_btn = Button(text='Save Selection', size_hint=(1, None), height=30)

        # Edit Bindings
        self.duplicate_btn.bind(on_release=self.duplicate)
        self.negative_btn.bind(on_release=self.negative)
        self.contrast_btn.bind(on_release=self.contrast)
        self.compression_btn.bind(on_release=self.dynamic_compression)
        self.s1 = 70
        self.s2 = 150
        self.gamma_btn.bind(on_release=self.gamma_function)
        self.gamma = 0.5
        self.histogram_btn.bind(on_release=self.histogram)
        self.save_selection_btn.bind(on_release=self.save_selection)

        self.edit_drop_down.add_widget(self.duplicate_btn)
        self.edit_drop_down.add_widget(self.negative_btn)
        self.edit_drop_down.add_widget(self.contrast_btn)
        self.edit_drop_down.add_widget(self.compression_btn)
        self.edit_drop_down.add_widget(self.gamma_btn)
        self.edit_drop_down.add_widget(self.histogram_btn)
        self.edit_drop_down.add_widget(self.save_selection_btn)

        # Noise Dropdown Buttons
        self.normal_noise_btn = Button(text='Normal', size_hint=(1, None), height=30)
        self.rayleigh_noise_btn = Button(text='Rayleigh', size_hint=(1, None), height=30)
        self.exp_noise_btn = Button(text='Exp', size_hint=(1, None), height=30)
        self.salt_noise_btn = Button(text='Salt', size_hint=(1, None), height=30)

        # Noise Bindings
        self.normal_mu = 0
        self.normal_sigma = 1
        self.normal_prob = 0.5
        self.normal_noise_btn.bind(on_release=self.normal_noise)
        self.rayleigh_scale = 1
        self.rayleigh_noise_btn.bind(on_release=self.rayleigh_noise)
        self.exp_scale = 1
        self.exp_prob = 0.5
        self.exp_noise_btn.bind(on_release=self.exp_noise)
        self.salt_prob = 0.5
        self.salt_noise_btn.bind(on_release=self.salt_noise)

        self.noise_drop_down.add_widget(self.normal_noise_btn)
        self.noise_drop_down.add_widget(self.rayleigh_noise_btn)
        self.noise_drop_down.add_widget(self.exp_noise_btn)
        self.noise_drop_down.add_widget(self.salt_noise_btn)

        # Image Coordinates Labels and Inputs
        self.coordinates_label = Label(text='(x, y)')
        self.x_input = TextInput(text='0', multiline=False)
        self.y_input = TextInput(text='0', multiline=False)
        self.x_input.bind(text=self.update_pixel_value_input)
        self.y_input.bind(text=self.update_pixel_value_input)
        self.value_label = Label(text='Value')
        self.value_input = TextInput(text='0', multiline=False)
        self.set_pixel_btn = Button(text='Set')
        self.set_pixel_btn.bind(on_release=self.set_pixel)

        self.coordinates.add_widget(self.coordinates_label)
        self.coordinates.add_widget(self.x_input)
        self.coordinates.add_widget(self.y_input)
        self.coordinates.add_widget(self.value_label)
        self.coordinates.add_widget(self.value_input)
        self.coordinates.add_widget(self.set_pixel_btn)

        self.add_widget(self.coordinates)

    def load(self, *args):
        # self.source = '../resources/lena.ascii.pbm'
        self.source = '../resources/test/LENA.RAW'
        # self.source = '../resources/color.pbm'
        # self.source = '../resources/blur.raw'
        (self.img, self.is_color) = Util.load_raw(self.source, (256, 256)), True
        # (self.img, self.is_color) = Util.load_image(self.source)
        self.draw_main_picture(self.img, self.is_color, self.img_pos)

    def draw_main_picture(self, img, is_color, position):
        img_size = (img.shape[0], img.shape[1])
        texture = self.create_texture(img, is_color, img_size)

        self.picture = Picture(pos=position, size=img_size, img=img, is_color=is_color,
                               x_input=self.x_input, y_input=self.y_input, value_input=self.value_input)
        self.canvas.remove_group('main_image')
        with self.picture.canvas:
            Rectangle(texture=texture, pos=position, size=img_size, group='main_image')
        self.add_widget(self.picture)

    def draw_transformed_image(self, image, position):
        img = Util.linear_transform(image)
        img_size = (img.shape[0], img.shape[1])
        texture = self.create_texture(img, self.is_color, img_size)
        self.transformed_picture = BoxLayout(pos=position, size=img_size)
        self.canvas.remove_group('transform')
        with self.transformed_picture.canvas:
            Rectangle(texture=texture, pos=position, size=img_size, group='transform')
        self.add_widget(self.transformed_picture)

    def create_texture(self, img, is_color, img_size):
        if is_color:
            color_fmt = 'rgb'
        else:
            color_fmt = 'luminance'
        texture = Texture.create(size=img_size, colorfmt=color_fmt)
        # Reverse the rows order (Because the (0,0) pixel from the matrix will
        # be drawn in the bottom-left border of the window)
        reversed_img = np.flipud(img)
        buffer = reversed_img.ravel()
        texture.blit_buffer(buffer, colorfmt=color_fmt, bufferfmt='ubyte')
        return texture

    def save(self, *args):
        if self.transformed_img is not None:
            Util.save(self.transformed_img, '../resources/transformed_img_' + str(self.image_number))
            self.image_number += 1

    def duplicate(self, *args):
        if self.img is not None:
            self.transformed_img = np.copy(self.img)
            self.draw_transformed_image(self.img, self.transformed_img_pos)

    def negative(self, *args):
        if self.img is not None:
            self.transformed_img = Util.negative(self.img)
            self.draw_transformed_image(self.transformed_img, self.transformed_img_pos)

    def contrast(self, *args):
        if self.img is not None:
            r = Util.transformed_img = Util.contrast_increase(self.img[:, :, 0], self.s1, self.s2)
            g = Util.transformed_img = Util.contrast_increase(self.img[:, :, 1], self.s1, self.s2)
            b = Util.transformed_img = Util.contrast_increase(self.img[:, :, 2], self.s1, self.s2)
            self.transformed_img = self.merge_rgb(r, g, b)
            self.draw_transformed_image(self.transformed_img, self.transformed_img_pos)

    def dynamic_compression(self, *args):
        if self.img is not None:
            r = Util.dynamic_range_compression(self.img[:, :, 0])
            g = Util.dynamic_range_compression(self.img[:, :, 1])
            b = Util.dynamic_range_compression(self.img[:, :, 2])
            self.transformed_img = self.merge_rgb(r, g, b)
            self.draw_transformed_image(self.transformed_img, self.transformed_img_pos)

    def merge_rgb(self, r, g, b):
        ans = np.zeros((r.shape[0], r.shape[1], 3))
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                ans[i][j][0] = r[i][j]
                ans[i][j][1] = g[i][j]
                ans[i][j][2] = b[i][j]
        return ans

    def gamma_function(self, *args):
        if self.img is not None:
            self.transformed_img = Util.gamma_power(self.img, self.gamma)
            self.draw_transformed_image(self.transformed_img, self.transformed_img_pos)

    def normal_noise(self, *args):
        if self.img is not None:
            self.transformed_img = Util.add_additive_noise_normal(self.img, self.normal_mu,
                                                                  self.normal_sigma, self.salt_prob)
            self.draw_transformed_image(self.transformed_img, self.transformed_img_pos)

    def rayleigh_noise(self, *args):
        if self.img is not None:
            self.transformed_img = Util.add_noise_rayleigh(self.img, self.rayleigh_scale)
            self.draw_transformed_image(self.transformed_img, self.transformed_img_pos)

    def exp_noise(self, *args):
        if self.img is not None:
            self.transformed_img = Util.add_noise_exponential(self.img, self.exp_scale, self.exp_prob)
            self.draw_transformed_image(self.transformed_img, self.transformed_img_pos)

    def salt_noise(self, *args):
        if self.img is not None:
            self.transformed_img = Util.add_comino_and_sugar_noise(self.img, self.salt_prob)
            self.draw_transformed_image(self.transformed_img, self.transformed_img_pos)

    def histogram(self, *args):
        if self.img is not None:
            plt.hist(self.img.flatten(), bins=range(256))
            plt.show()

    def save_selection(self, *args):
        if self.picture is not None and self.picture.is_selected:
            pixels = self.picture.get_selection_coordinates()
            selected_img = Util.trim(self.img, pixels[0], pixels[1])
            Util.save(selected_img, '../resources/selected_img_' + str(self.selection_number))
            self.selection_number += 1

    def update_pixel_value_input(self, *args):
        if self.img is not None:
            coord = self.get_coordinates()
            if coord is not None:
                x = coord[0]
                y = coord[1]
                if self.is_color:
                    self.value_input.text = '{} {} {}'.format(self.img[y, x, 0], self.img[y, x, 1], self.img[y, x, 2])
                else:
                    self.value_input.text = str(self.img[y][x])

    def set_pixel(self, *args):
        if self.img is not None:
            coord = self.get_coordinates()
            value = self.get_pixel_value()
            if coord is not None and value is not None:
                x = coord[0]
                y = coord[1]
                if self.is_color:
                    self.img[y, x, 0] = value[0]
                    self.img[y, x, 1] = value[1]
                    self.img[y, x, 2] = value[2]
                else:
                    self.img[y][x] = value
                self.draw_main_picture(self.img, self.is_color, self.img_pos)

    def get_coordinates(self):
        try:
            x = int(self.x_input.text)
            y = int(self.y_input.text)
            if 0 <= x <= self.img.shape[0] and 0 <= y <= self.img.shape[1]:
                return x, y
            else:
                return None
        except ValueError:
            return None

    def get_pixel_value(self):
        try:
            if self.is_color:
                (r, g, b) = self.value_input.text.split()
                r = int(r)
                g = int(g)
                b = int(b)
                if 0 <= r <= 255 and 0 <= b <= 255 and 0 <= b <= 255:
                    return r, g, b
                else:
                    return None
            else:
                value = int(self.value_input.text)
                if 0 <= value <= 255:
                    return value
                else:
                    return None
        except ValueError:
            return None


class ImageEditorApp(App):
    def build(self):
        return Root()


if __name__ == '__main__':
    ImageEditorApp().run()
