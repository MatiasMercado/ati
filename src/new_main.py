import tkinter as tk
import tkinter.ttk as ttk
import PIL
from PIL import ImageTk
import numpy as np

from input.provider import Provider
from input.util import Util
from matplotlib import pyplot as plt


class ImageEditor(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid(sticky=tk.N + tk.S + tk.E + tk.W)
        self.image_number = 1
        self.active_window = tk.StringVar()
        self.create_menu()
        self.open_images = {}
        self.path = '../resources/test/BARCO.RAW'
        self.size = (290, 207)
        # File = askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')

        # Parameters
        # Contrast
        self.s1 = 70
        self.s2 = 150

        # Gamma
        self.gamma = 0.5

        # Binary
        self.binary_threshold = 100

    def create_menu(self):
        root = self.master

        # Menu Bar
        menu_bar = tk.Menu(root)
        # file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu = tk.Menu(menu_bar)
        transform_menu = tk.Menu(menu_bar)
        noise_menu = tk.Menu(menu_bar)
        filter_menu = tk.Menu(menu_bar)
        operations_menu = tk.Menu(menu_bar)

        # File Menu
        file_menu.add_command(label='Load', command=self.load_image)
        file_menu.add_command(label='Edit')
        file_menu.add_command(label='Save')
        file_menu.add_separator()
        file_menu.add_command(label='Save selection')
        file_menu.add_command(label='Histogram')
        file_menu.add_separator()
        file_menu.add_command(label='Quit', command=root.destroy)
        menu_bar.add_cascade(label='File', menu=file_menu)

        # Transform Menu
        transform_menu.add_command(label='Negative', command=self.negative)
        transform_menu.add_command(label='Contrast', command=self.contrast)
        transform_menu.add_command(label='Compression', command=self.dynamic_compression)
        transform_menu.add_command(label='Gamma', command=self.gamma_function)
        transform_menu.add_command(label='Binary', command=self.to_binary)
        transform_menu.add_command(label='Equalize', command=self.equalize)
        menu_bar.add_cascade(label='Transform', menu=transform_menu)

        # Noise Menu
        noise_menu.add_command(label='Normal')
        noise_menu.add_command(label='Rayleigh')
        noise_menu.add_command(label='Exp')
        noise_menu.add_command(label='Salt-pepper')
        menu_bar.add_cascade(label='Noise', menu=noise_menu)

        # Filter Menu
        filter_menu.add_command(label='Mean')
        filter_menu.add_command(label='Median')
        filter_menu.add_command(label='P. Median')
        filter_menu.add_command(label='Normal')
        filter_menu.add_command(label='Borders')
        menu_bar.add_cascade(label='Filter', menu=filter_menu)

        # Add menu Bar to Root Window
        root.config(menu=menu_bar)

    def load_image(self):
        (width, height) = self.size
        # IMPORTANT: Notice we exchange (w,h) to (h,w) to load the image correctly
        img_data = Util.load_raw(self.path, (height, width))
        self.create_new_image(img_data)

    def create_new_image(self, img_data):
        linear_img = Util.linear_transform(img_data)
        pil_img = PIL.Image.fromarray(linear_img, 'RGB')
        tk_img = ImageTk.PhotoImage(pil_img)

        # Matrix shape is (height, width, 3)
        height = img_data.shape[0]
        width = img_data.shape[1]

        new_window = tk.Toplevel()

        new_window.geometry('{}x{}'.format(width, height))
        new_window.resizable(width=False, height=False)
        new_window.title('Image {}'.format(self.image_number))

        new_window.bind('<ButtonRelease-1>', self.set_active_window)
        new_window.bind('<Destroy>', self.remove_open_image)

        self.image_number += 1
        canvas = tk.Canvas(new_window, width=width, height=height)
        canvas.grid(row=0, column=0)
        # canvas.create_image(-.5, -.5, image=tk_img, anchor=tk.NW)
        canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
        canvas.my_image = tk_img # Used only to prevent image being destroy by garbage collector
        self.open_images[new_window.title()] = img_data

    def set_active_window(self, event):
        self.active_window.set(event.widget.winfo_toplevel().title())

    def remove_open_image(self, event):
        self.open_images.pop(event.widget.winfo_toplevel().title(), None)

    def negative(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.negative(image)
        self.create_new_image(transformed_img)

    def negative(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.negative(image)
        self.create_new_image(transformed_img)

    def contrast(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        r = Util.contrast_increase(image[:, :, 0], self.s1, self.s2)
        g = Util.contrast_increase(image[:, :, 1], self.s1, self.s2)
        b = Util.contrast_increase(image[:, :, 2], self.s1, self.s2)
        transformed_img = self.merge_rgb(r, g, b)
        self.create_new_image(transformed_img)

    def dynamic_compression(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        r = Util.dynamic_range_compression(image[:, :, 0])
        g = Util.dynamic_range_compression(image[:, :, 1])
        b = Util.dynamic_range_compression(image[:, :, 2])
        transformed_img = self.merge_rgb(r, g, b)
        self.create_new_image(transformed_img)

    def gamma_function(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.gamma_power(image, self.gamma)
        self.create_new_image(transformed_img)

    def equalize(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Provider.equalize_histogram(image)
        self.create_new_image(transformed_img)

    def to_binary(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.to_binary(image, self.binary_threshold)
        self.create_new_image(transformed_img)

    def merge_rgb(self, r, g, b):
        ans = np.zeros((r.shape[0], r.shape[1], 3))
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                ans[i][j][0] = r[i][j]
                ans[i][j][1] = g[i][j]
                ans[i][j][2] = b[i][j]
        return ans

    def edit_image(self):
        top = self.winfo_toplevel()
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        img_data = self.load_image()
        pil_img = PIL.Image.fromarray(img_data, 'RGB')
        tk_img = ImageTk.PhotoImage(pil_img)
        self.label = tk.Label(self.master, image=self.tk_img).grid(row=0, column=0)
        self.label.my_image = tk_img  # Used only to prevent image being destroy by garbage collector

if __name__ == '__main__':
    app = ImageEditor()
    app.master.title('Image Editor')
    app.master.geometry('250x250')
    app.mainloop()

