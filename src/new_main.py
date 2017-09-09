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

        # Make the root window and the main frame responsive to resize
        self.grid(sticky=tk.N + tk.S + tk.E + tk.W)
        top = self.winfo_toplevel()
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Create Widgets
        self.init_default_settings()
        self.settings = self.create_settings()
        self.create_menu()

        # State Variables
        self.active_window = tk.StringVar()
        self.image_number = 1
        self.open_images = {}

        # Display Settings
        # self.show_settings()

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
        noise_menu.add_command(label='Gauss', command=self.normal_noise)
        noise_menu.add_command(label='Rayleigh', command=self.rayleigh_noise)
        noise_menu.add_command(label='Exp', command=self.exp_noise)
        noise_menu.add_command(label='Salt-pepper', command=self.salt_pepper_noise)
        menu_bar.add_cascade(label='Noise', menu=noise_menu)

        # Filter Menu
        filter_menu.add_command(label='Mean', command=self.mean_filter)
        filter_menu.add_command(label='Median', command=self.median_filter)
        filter_menu.add_command(label='P. Median', command=self.p_median_filter)
        filter_menu.add_command(label='Gauss', command=self.normal_filter)
        filter_menu.add_command(label='Borders', command=self.borders_filter)
        menu_bar.add_cascade(label='Filter', menu=filter_menu)

        # Settings
        menu_bar.add_command(label='Settings', command=self.show_settings)

        # Add menu Bar to Root Window
        root.config(menu=menu_bar)

    def init_default_settings(self):
        # Contrast
        self.s1 = tk.DoubleVar()
        self.s1.set(10)
        self.s2 = tk.DoubleVar()
        self.s2.set(50)

        # Gamma
        self.gamma = tk.DoubleVar()
        self.gamma.set(0.5)

        # Binary
        self.binary_threshold = tk.DoubleVar()
        self.binary_threshold.set(100)

        # Scalar Product
        self.scalar = tk.DoubleVar()
        self.scalar.set(1.2)

        # Noise
        self.normal_mu = tk.DoubleVar()
        self.normal_mu .set(0) # THIS SHOULD ALWAYS BE 0
        self.normal_sigma = tk.DoubleVar()
        self.normal_sigma.set(20)
        self.normal_prob = tk.DoubleVar()
        self.normal_prob.set(1)
        self.rayleigh_scale = tk.DoubleVar()
        self.rayleigh_scale.set(0.25)
        self.exp_scale = tk.DoubleVar()
        self.exp_scale.set(0.25)
        self.exp_prob = tk.DoubleVar()
        self.exp_prob.set(1)
        self.salt_pepper_p0 = tk.DoubleVar()
        self.salt_pepper_p0.set(0.1)
        self.salt_pepper_p1 = tk.DoubleVar()
        self.salt_pepper_p1.set(0.9)

        # Filters
        self.mean_filter_size = tk.StringVar()
        self.mean_filter_size.set('3 3')
        self.median_filter_mask = tk.StringVar()
        self.median_filter_mask.set('[[1, 3, 1]; [3, 5, 3]; [1, 3, 1]]')
        self.normal_filter_size = tk.StringVar()
        self.normal_filter_size.set('3 3')
        self.normal_filter_sigma = tk.DoubleVar()
        self.normal_filter_sigma.set(20)

    def create_settings(self):
        settings_frame = tk.Frame(self)
        # Title
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=0, columnspan=2, sticky=(tk.W, tk.E))
        tk.Label(settings_frame, text='Settings').grid(row=0, columnspan=2)

        # Contrast
        tk.Label(settings_frame, text='Contrast').grid(row=1, column=0)
        tk.Label(settings_frame, text='S1').grid(row=2, column=0)
        tk.Entry(settings_frame, text=self.s1, textvariable=self.s1).grid(row=2, column=1)
        tk.Label(settings_frame, text='S2').grid(row=3, column=0)
        tk.Entry(settings_frame, text=self.s2, textvariable=self.s2).grid(row=3, column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=4, columnspan=2, sticky=(tk.W, tk.E))

        # Gamma
        tk.Label(settings_frame, text='Gamma').grid(row=5, column=0)
        # tk.Scale(settings_frame, variable=self.gamma, from_=0, to=3, orient=tk.HORIZONTAL).grid(row=4, column=1)
        tk.Entry(settings_frame, text=self.gamma, textvariable=self.gamma).grid(row=5, column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=6, columnspan=2, sticky=(tk.W, tk.E))

        # Binary
        tk.Label(settings_frame, text='Binary Threshold').grid(row=7, column=0)
        tk.Entry(settings_frame, text=self.binary_threshold, textvariable=self.binary_threshold).grid(row=7, column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=8, columnspan=2, sticky=(tk.W, tk.E))

        # Scalar Product
        tk.Label(settings_frame, text='Scalar Product').grid(row=9, column=0)
        tk.Entry(settings_frame, text=self.scalar, textvariable=self.scalar).grid(row=9, column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=10, columnspan=2, sticky=(tk.W, tk.E))

        # Noise
        tk.Label(settings_frame, text='Gauss Noise').grid(row=11, column=0)
        tk.Label(settings_frame, text='Deviation').grid(row=12, column=0)
        tk.Entry(settings_frame, text=self.normal_sigma, textvariable=self.normal_sigma).grid(row=12, column=1)

        tk.Label(settings_frame, text='Probability').grid(row=13, column=0)
        tk.Entry(settings_frame, text=self.normal_prob, textvariable=self.normal_prob).grid(row=13, column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=14, columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Rayleigh Scale').grid(row=15, column=0)
        tk.Entry(settings_frame, text=self.rayleigh_scale, textvariable=self.rayleigh_scale).grid(row=15, column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=16, columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Exp Noise').grid(row=17, column=0)
        tk.Label(settings_frame, text='Exp Scale').grid(row=18, column=0)
        tk.Entry(settings_frame, text=self.exp_scale, textvariable=self.exp_scale).grid(row=18, column=1)

        tk.Label(settings_frame, text='Exp Probability').grid(row=19, column=0)
        tk.Entry(settings_frame, text=self.exp_prob, textvariable=self.exp_prob).grid(row=19, column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=20, columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Salt Pepper Noise').grid(row=21, column=0)
        tk.Label(settings_frame, text='P0').grid(row=22, column=0)
        tk.Entry(settings_frame, text=self.salt_pepper_p0, textvariable=self.salt_pepper_p0).grid(row=22, column=1)
        tk.Label(settings_frame, text='P1').grid(row=23, column=0)
        tk.Entry(settings_frame, text=self.salt_pepper_p1, textvariable=self.salt_pepper_p1).grid(row=23, column=1)

        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=24, columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Mean Filter Size').grid(row=25, column=0)
        tk.Entry(settings_frame, text=self.mean_filter_size, textvariable=self.mean_filter_size).grid(row=25, column=1)

        tk.Label(settings_frame, text='Median Filter Mask').grid(row=26, column=0)
        tk.Entry(settings_frame, text=self.median_filter_mask, textvariable=self.median_filter_mask).grid(row=26, column=1)

        tk.Label(settings_frame, text='Gauss Filter Size').grid(row=27, column=0)
        tk.Entry(settings_frame, text=self.normal_filter_size, textvariable=self.normal_filter_size).grid(row=27, column=1)

        tk.Label(settings_frame, text='Gauss Filter Deviation').grid(row=28, column=0)
        tk.Entry(settings_frame, text=self.normal_filter_sigma, textvariable=self.normal_filter_sigma).grid(row=28, column=1)

        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(columnspan=2, sticky=(tk.W, tk.E))
        tk.Button(settings_frame, text='Return', command=self.hide_settings).grid(columnspan=2, sticky=(tk.W, tk.E))
        return settings_frame

    def show_settings(self):
        self.settings.grid()
        # self.settings.grid(sticky=tk.N + tk.S + tk.E + tk.W)

    def hide_settings(self):
        self.settings.grid_remove()

    # File Menu Functions
    def load_image(self):
        img_path = tk.filedialog.askopenfilename(initialdir='../resources/test', title='Select Image')
        img_data = Util.load_image(img_path)
        print(img_data)
        print(img_data.shape)
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
        # r = Util.contrast_increase(image[:, :, 0], self.s1.get(), self.s2.get())
        # g = Util.contrast_increase(image[:, :, 1], self.s1.get(), self.s2.get())
        # b = Util.contrast_increase(image[:, :, 2], self.s1.get(), self.s2.get())
        # transformed_img = self.__merge_rgb(r, g, b)
        transformed_img = Util.contrast_increase(image, self.s1.get(), self.s2.get())
        self.create_new_image(transformed_img)

    def dynamic_compression(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        # r = Util.dynamic_range_compression(image[:, :, 0])
        # g = Util.dynamic_range_compression(image[:, :, 1])
        # b = Util.dynamic_range_compression(image[:, :, 2])
        # transformed_img = self.__merge_rgb(r, g, b)
        transformed_img = Util.dynamic_range_compression(image)
        self.create_new_image(transformed_img)

    def gamma_function(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.gamma_power(image, self.gamma.get())
        self.create_new_image(transformed_img)

    def equalize(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Provider.equalize_histogram(image)
        self.create_new_image(transformed_img)

    def to_binary(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.to_binary(image, self.binary_threshold.get())
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
        print('Scalar Product With: {}'.format(self.scalar.get()))
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.scalar_prod(image, self.scalar.get())
        self.create_new_image(transformed_img)

    # Noise Functions
    def normal_noise(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.add_additive_noise_normal(image, self.normal_mu.get(), self.normal_sigma.get(),
                                                         self.normal_prob.get())
        self.create_new_image(transformed_img)

    def rayleigh_noise(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.add_noise_rayleigh(image, self.rayleigh_scale.get())
        self.create_new_image(transformed_img)

    def exp_noise(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.add_noise_exponential(image, self.exp_scale.get(), self.exp_prob.get())
        self.create_new_image(transformed_img)

    def salt_pepper_noise(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.add_comino_and_sugar_noise(image, self.salt_pepper_p0.get(), self.salt_pepper_p1.get())
        self.create_new_image(transformed_img)

    # Filter Functions
    def mean_filter(self):
        size = self.mean_filter_size.get().split()
        mean_filter_size = (int(size[0]), int(size[1]))
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.blur(image, mean_filter_size)
        self.create_new_image(transformed_img)

    def median_filter(self):
        print(self.median_filter_mask.get())
        median_filter_mask = np.matrix(self.median_filter_mask.get())
        print(median_filter_mask.shape)
        print(median_filter_mask)
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.median_filter(image, median_filter_mask, False)
        self.create_new_image(transformed_img)

    def p_median_filter(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.median_filter(image, weighted=True)
        self.create_new_image(transformed_img)

    def normal_filter(self):
        size = self.normal_filter_size.get().split()
        normal_filter_size = (int(size[0]), int(size[1]))
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.gauss_blur(image, normal_filter_size, self.normal_filter_sigma.get())
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
    app.master.geometry('400x500')
    app.mainloop()

