import tkinter as tk
import tkinter.ttk as ttk
import PIL
from PIL import ImageTk
import numpy as np

from input.filter_provider import FilterProvider
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

        # Parameters

        # Contrast
        self.s1 = 70
        self.s2 = 150

        # Gamma
        self.gamma = 0.5

        # Binary
        self.binary_threshold = 10

        # Scalar Product
        self.scalar = 1.2

        # Noise
        self.normal_mu = 0
        self.normal_sigma = 1
        self.normal_prob = 0.5
        self.rayleigh_scale = 1
        self.exp_scale = 1
        self.exp_prob = 0.5
        self.salt_pepper_prob = (0.25, 0.25)

        # Filters
        self.mean_filter_size = (3, 3)
        self.median_filter_mask = np.matrix([[1, 3, 1], [3, 5, 3], [1, 3, 1]])
        self.p_median_filter_mask = np.matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.normal_filter_size = (3, 3)
        self.normal_filter_sigma = 0.5

    def create_menu(self):
        root = self.master

        # Menu Bar
        menu_bar = tk.Menu(root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        transform_menu = tk.Menu(menu_bar, tearoff=0)
        noise_menu = tk.Menu(menu_bar, tearoff=0)
        filter_menu = tk.Menu(menu_bar, tearoff=0)
        operations_menu = tk.Menu(menu_bar, tearoff=0)

        # File Menu
        file_menu.add_command(label='Load', command=self.load_image)
        file_menu.add_command(label='Edit')
        file_menu.add_command(label='Save', command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label='Save selection')
        file_menu.add_command(label='Histogram', command=self.histogram)
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

        # Operations Menu
        operations_menu.add_command(label='Add', command=self.add)
        operations_menu.add_command(label='Difference', command=self.difference)
        operations_menu.add_command(label='Product', command=self.multiply)
        operations_menu.add_command(label='Scalar Product', command=self.scalar_product)
        menu_bar.add_cascade(label='Operations', menu=operations_menu)

        # Noise Menu
        noise_menu.add_command(label='Normal', command=self.normal_noise)
        noise_menu.add_command(label='Rayleigh', command=self.rayleigh_noise)
        noise_menu.add_command(label='Exp', command=self.exp_noise)
        noise_menu.add_command(label='Salt-pepper', command=self.salt_pepper_noise)
        menu_bar.add_cascade(label='Noise', menu=noise_menu)

        # Filter Menu
        filter_menu.add_command(label='Mean', command=self.mean_filter)
        filter_menu.add_command(label='Median', command=self.median_filter)
        filter_menu.add_command(label='P. Median', command=self.p_median_filter)
        filter_menu.add_command(label='Normal', command=self.normal_filter)
        filter_menu.add_command(label='Borders', command=self.borders_filter)
        menu_bar.add_cascade(label='Filter', menu=filter_menu)

        # Add menu Bar to Root Window
        root.config(menu=menu_bar)

    # File Menu Functions

    def load_image(self):
        img_path = tk.filedialog.askopenfilename(initialdir='../resources/test', title='Select Image')
        # TODO: Read this from a file or a static map
        # size = (290, 207)   # Size of BARCO.RAW
        size = (256, 256)   # Size of BARCO.RAW
        (width, height) = size
        # IMPORTANT: Notice we exchange (w,h) to (h,w) to load the image correctly
        # TODO: Change this for a generic load method that checks on the img_path extension
        img_data = Util.load_raw(img_path, (height, width))
        self.create_new_image(img_data)

    def save_image(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        img_path = tk.filedialog.asksaveasfilename(initialdir='../resources/test', title='Save Image')
        linear_image = Util.linear_transform(image)
        # TODO: Change this for a generic save method that checks on the img_path extension
        Util.save_raw(linear_image, img_path)

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
        canvas = tk.Canvas(new_window, width=width, height=height, borderwidth=0, highlightthickness=0)
        canvas.grid(row=0, column=0)
        canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
        canvas.my_image = tk_img # Used only to prevent image being destroy by garbage collector
        self.open_images[new_window.title()] = img_data

    def set_active_window(self, event):
        self.active_window.set(event.widget.winfo_toplevel().title())

    def remove_open_image(self, event):
        self.open_images.pop(event.widget.winfo_toplevel().title(), None)

    def histogram(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        plt.hist(image.flatten(), bins=range(256))
        plt.show()

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

    # Transform Menu Functions

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
        transformed_img = self.__merge_rgb(r, g, b)
        self.create_new_image(transformed_img)

    def dynamic_compression(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        r = Util.dynamic_range_compression(image[:, :, 0])
        g = Util.dynamic_range_compression(image[:, :, 1])
        b = Util.dynamic_range_compression(image[:, :, 2])
        transformed_img = self.__merge_rgb(r, g, b)
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

    # Operations Functions

    def add(self):
        self.wait_variable(self.active_window)
        image1 = self.open_images[self.active_window.get()]
        self.wait_variable(self.active_window)
        image2 = self.open_images[self.active_window.get()]
        transformed_img = Util.sum(image1, image2)
        self.create_new_image(transformed_img)

    def difference(self):
        self.wait_variable(self.active_window)
        image1 = self.open_images[self.active_window.get()]
        self.wait_variable(self.active_window)
        image2 = self.open_images[self.active_window.get()]
        transformed_img = Util.difference(image1, image2)
        self.create_new_image(transformed_img)

    def multiply(self):
        self.wait_variable(self.active_window)
        image1 = self.open_images[self.active_window.get()]
        self.wait_variable(self.active_window)
        image2 = self.open_images[self.active_window.get()]
        transformed_img = Util.multiply(image1, image2)
        self.create_new_image(transformed_img)

    def scalar_product(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.scalar_prod(image, self.scalar)
        self.create_new_image(transformed_img)

    # Noise Functions

    def normal_noise(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.add_additive_noise_normal(image, self.normal_mu, self.normal_sigma,
                                                         self.normal_prob)
        self.create_new_image(transformed_img)

    def rayleigh_noise(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.add_noise_rayleigh(image, self.rayleigh_scale)
        self.create_new_image(transformed_img)

    def exp_noise(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.add_noise_exponential(image, self.exp_scale, self.exp_prob)
        self.create_new_image(transformed_img)

    def salt_pepper_noise(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.add_comino_and_sugar_noise(image, self.salt_pepper_prob)
        self.create_new_image(transformed_img)

    # Filter Functions

    def mean_filter(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.blur(image, self.mean_filter_size)
        self.create_new_image(transformed_img)

    def median_filter(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.median_filter(image, self.median_filter_mask, False)
        self.create_new_image(transformed_img)

    def p_median_filter(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.sliding_window_median(image, self.p_median_filter_mask, True)
        self.create_new_image(transformed_img)

    def normal_filter(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.gauss_blur(image, self.normal_filter_size, self.normal_filter_sigma)
        self.create_new_image(transformed_img)

    def borders_filter(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.pasa_altos(image)
        self.create_new_image(transformed_img)

    # Private Functions

    def __merge_rgb(self, r, g, b):
        ans = np.zeros((r.shape[0], r.shape[1], 3))
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                ans[i][j][0] = r[i][j]
                ans[i][j][1] = g[i][j]
                ans[i][j][2] = b[i][j]
        return ans

if __name__ == '__main__':
    app = ImageEditor()
    app.master.title('Image Editor')
    app.master.geometry('350x250')
    app.mainloop()

