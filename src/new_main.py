import tkinter as tk
import tkinter.ttk as ttk
import PIL
from PIL import ImageTk
import numpy as np
from input.util import Util
from matplotlib import pyplot as plt


class ImageEditor(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid(sticky=tk.N + tk.S + tk.E + tk.W)
        # self.create_menu()
        self.image_number = 1
        self.create_menu()
        self.open_images = {}
        self.path = '../resources/test/BARCO.RAW'
        self.size = (290, 207)

        #     File = askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')


        # Create Image Button
        self.quitButton = tk.Button(self, text='Click MEEEE', command=self.create_new_image).grid(row=0, column=0)

    def create_menu(self):
        root = self.master

        # Menu Bar
        menu_bar = tk.Menu(root)
        file_menu = tk.Menu(menu_bar)
        # file_menu = tk.Menu(menu_bar, tearoff=0)
        edit_menu = tk.Menu(menu_bar)

        # File Menu
        file_menu.add_command(label='Load')
        file_menu.add_command(label='Save')
        file_menu.add_separator()
        file_menu.add_command(label='Quit', command=root.destroy)
        menu_bar.add_cascade(label='File', menu=file_menu)

        # Edit Menu
        edit_menu.add_command(label='Negative')
        edit_menu.add_command(label='Contrast')
        edit_menu.add_command(label='Gamma')
        menu_bar.add_cascade(label='Edit', menu=edit_menu)

        # Add menu Bar to Root Window
        root.config(menu=menu_bar)

    def create_new_image(self):
        img_data = self.load_image(self.path, self.size)
        pil_img = PIL.Image.fromarray(img_data, 'RGB')
        tk_img = ImageTk.PhotoImage(pil_img)

        # Matrix shape is (height, width, 3)
        height = img_data.shape[0]
        width = img_data.shape[1]

        new_window = tk.Toplevel()
        new_window.geometry('{}x{}'.format(width, height))
        new_window.resizable(width=False, height=False)
        # new_window.title('Image {}'.format(self.image_number))
        new_window.title('Image {}'.format(self.image_number))
        self.image_number += 1
        canvas = tk.Canvas(new_window, width=width, height=height)
        canvas = tk.Canvas(new_window)
        canvas.grid(row=0, column=0)
        canvas.create_image(-.5, -.5, image=tk_img, anchor=tk.NW)
        canvas.my_image = tk_img # Used only to prevent image being destroy by garbage collector
        self.open_images[new_window.title] = tk_img

    def create_image(self):
        top = self.winfo_toplevel()
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        img_data = self.load_image(self.path, self.size)
        pil_img = PIL.Image.fromarray(img_data, 'RGB')
        self.tk_img = ImageTk.PhotoImage(pil_img)
        label = tk.Label(self.master, image=self.tk_img).grid(row=0, column=0)

    def load_image(self, path, size):
        (width, height) = size
        # IMPORTANT: Notice we exchange (w,h) to (h,w) to load the image correctly
        return Util.load_raw(path, (height, width))

if __name__ == '__main__':
    app = ImageEditor()
    app.master.title('Image Editor')
    # app.master.geometry('800x600')
    app.mainloop()

