import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from src.input.util import Util
from src.input.filter_provider import FilterProvider
from src.input.provider import Provider
from PIL import ImageTk
import tkinter as tk
import tkinter.ttk as ttk
import PIL
import numpy as np


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
        self.settings = self.create_settings_window()
        self.create_menu()

        # State Variables
        self.active_window = tk.StringVar()
        self.image_number = 1
        self.open_images = {}

        self.x_coord = tk.DoubleVar();
        self.y_coord = tk.DoubleVar();
        self.r_value = tk.DoubleVar();
        self.g_value = tk.DoubleVar();
        self.b_value = tk.DoubleVar();
        # is_selected
        # selection_x1, y1, x2, y2

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
        file_menu.add_command(label='Edit', command=self.edit_image)
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
        filter_menu.add_command(label='W. Median', command=self.w_median_filter)
        filter_menu.add_command(label='Gauss', command=self.gauss_filter)
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
        self.normal_mu.set(0)  # THIS SHOULD ALWAYS BE 0
        self.normal_sigma = tk.DoubleVar()
        self.normal_sigma.set(20)
        self.normal_prob = tk.DoubleVar()
        self.normal_prob.set(1)
        self.rayleigh_scale = tk.DoubleVar()
        self.rayleigh_scale.set(0.25)
        self.rayleigh_prob = tk.DoubleVar()
        self.rayleigh_prob.set(1)
        self.exp_scale = tk.DoubleVar()
        self.exp_scale.set(0.25)
        self.exp_prob = tk.DoubleVar()
        self.exp_prob.set(1)
        self.salt_pepper_p0 = tk.DoubleVar()
        self.salt_pepper_p0.set(0.05)
        self.salt_pepper_p1 = tk.DoubleVar()
        self.salt_pepper_p1.set(0.95)
        self.salt_pepper_density = tk.DoubleVar()
        self.salt_pepper_density.set(1)

        # Filters
        self.mean_filter_size = tk.StringVar()
        self.mean_filter_size.set('3 3')
        self.gauss_filter_size = tk.StringVar()
        self.gauss_filter_size.set('3 3')
        self.gauss_filter_sigma = tk.DoubleVar()
        self.gauss_filter_sigma.set(20)

    def create_settings_window(self):
        settings_frame = tk.Frame(self)
        self.settings_row = 0;

        def curr_row():
            return self.settings_row;

        def next_row():
            self.settings_row = self.settings_row + 1
            return self.settings_row;

        # Title
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=curr_row(), columnspan=2, sticky=(tk.W, tk.E))
        tk.Label(settings_frame, text='Settings').grid(row=curr_row(), columnspan=2)

        # Contrast
        tk.Label(settings_frame, text='Contrast').grid(row=next_row(), column=0)
        tk.Label(settings_frame, text='S1').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.s1, textvariable=self.s1).grid(row=curr_row(), column=1)
        tk.Label(settings_frame, text='S2').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.s2, textvariable=self.s2).grid(row=curr_row(), column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        # Gamma
        tk.Label(settings_frame, text='Gamma').grid(row=next_row(), column=0)
        # tk.Scale(settings_frame, variable=self.gamma, from_=0, to=3, orient=tk.HORIZONTAL).grid(row=4, column=1)
        tk.Entry(settings_frame, text=self.gamma, textvariable=self.gamma).grid(row=curr_row(), column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        # Binary
        tk.Label(settings_frame, text='Binary Threshold').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.binary_threshold, textvariable=self.binary_threshold).grid(row=curr_row(), column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        # Scalar Product
        tk.Label(settings_frame, text='Scalar Product').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.scalar, textvariable=self.scalar).grid(row=curr_row(), column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        # Noise
        tk.Label(settings_frame, text='Gauss Noise').grid(row=next_row(), column=0)
        tk.Label(settings_frame, text='Deviation').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.normal_sigma, textvariable=self.normal_sigma).grid(row=curr_row(), column=1)

        tk.Label(settings_frame, text='Density').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.normal_prob, textvariable=self.normal_prob).grid(row=curr_row(), column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Rayleigh Noise').grid(row=next_row(), column=0)
        tk.Label(settings_frame, text='Scale').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.rayleigh_scale, textvariable=self.rayleigh_scale).grid(row=curr_row(), column=1)

        tk.Label(settings_frame, text='Density').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.rayleigh_prob, textvariable=self.rayleigh_prob).grid(row=curr_row(), column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Exp Noise').grid(row=next_row(), column=0)
        tk.Label(settings_frame, text='Scale').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.exp_scale, textvariable=self.exp_scale).grid(row=curr_row(), column=1)

        tk.Label(settings_frame, text='Density').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.exp_prob, textvariable=self.exp_prob).grid(row=curr_row(), column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Salt Pepper Noise').grid(row=next_row(), column=0)
        tk.Label(settings_frame, text='P0').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.salt_pepper_p0, textvariable=self.salt_pepper_p0).grid(row=curr_row(), column=1)
        tk.Label(settings_frame, text='P1').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.salt_pepper_p1, textvariable=self.salt_pepper_p1).grid(row=curr_row(), column=1)

        tk.Label(settings_frame, text='Density').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.salt_pepper_density, textvariable=self.salt_pepper_density).grid(row=curr_row(), column=1)

        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Mean Filter Size').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.mean_filter_size, textvariable=self.mean_filter_size).grid(row=curr_row(), column=1)

        tk.Label(settings_frame, text='Gauss Filter Size').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.gauss_filter_size, textvariable=self.gauss_filter_size).grid(row=curr_row(), column=1)

        tk.Label(settings_frame, text='Gauss Filter Deviation').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.gauss_filter_sigma, textvariable=self.gauss_filter_sigma).grid(row=curr_row(), column=1)

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
        canvas.my_image = tk_img  # Used only to prevent image being destroy by garbage collector
        self.open_images[new_window.title()] = img_data

    def set_active_window(self, event):
        self.active_window.set(event.widget.winfo_toplevel().title())

    def remove_open_image(self, event):
        self.open_images.pop(event.widget.winfo_toplevel().title(), None)

    def histogram(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        plt.hist(image.flatten(), bins='auto', normed=1)
        plt.show()

    def edit_image(self):
        self.wait_variable(self.active_window)
        img_data = self.open_images[self.active_window.get()]

        linear_img = Util.linear_transform(img_data)
        pil_img = PIL.Image.fromarray(linear_img, 'RGB')
        tk_img = ImageTk.PhotoImage(pil_img)
        height = img_data.shape[0]
        width = img_data.shape[1]

        root = self.master;
        canvas = tk.Canvas(root, width=width, height=height, borderwidth=0, highlightthickness=0)
        canvas.grid(row=0, column=1)
        canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
        canvas.my_image = tk_img  # Used only to prevent image being destroy by garbage collector

        self.edited_img_data = np.copy(img_data).astype(float)
        # self.edited_img_canvas = canvas

        coord_frame = tk.Frame(self)
        tk.Label(coord_frame, text='(x,y)').grid(row=0, column=0, columnspan=3)
        tk.Entry(coord_frame, text=self.x_coord, textvariable=self.x_coord, width=5).grid(row=1,column=0)
        tk.Entry(coord_frame, text=self.y_coord, textvariable=self.y_coord, width=5).grid(row=1,column=1)
        tk.Label(coord_frame, text='(r,g,b)').grid(row=2, column=0, columnspan=3)
        tk.Entry(coord_frame, text=self.r_value, textvariable=self.r_value, width=5).grid(row=3, column=0)
        tk.Entry(coord_frame, text=self.g_value, textvariable=self.g_value, width=5).grid(row=3, column=1)
        tk.Entry(coord_frame, text=self.b_value, textvariable=self.b_value, width=5).grid(row=3, column=2)
        tk.Button(coord_frame, text='Set Value').grid(row = 1, column = 2, padx=10)

        ttk.Separator(coord_frame, orient=tk.HORIZONTAL).grid(row=5, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        tk.Button(coord_frame, text='Cancel').grid(row=6, column=0, columnspan=1, pady=10)
        tk.Button(coord_frame, text='Save').grid(row=6, column=2, columnspan=1, pady=10)

        # TODO: return coord_frame and save it to hide it later (copy create_settings method)
        coord_frame.grid(row=0, column=0)


    # def create_edit_panel(self):
    #     edit_frame = tk.Frame(self, bd=1)
    #     # self.edited_image_label = tk.Label(edit_frame).grid()
    #     tk.Button(edit_frame, text='Discard', command=lambda: self.hide_edit_panel)\
    #         .grid(columnspan=2, sticky=(tk.W, tk.E))
    #     tk.Button(edit_frame, text='Finish', command=lambda: self.create_new_image(self.edited_image))\
    #         .grid(columnspan=2, sticky=(tk.W, tk.E))
    #     return edit_frame
    #     # ----------------- DELETE ----------------------------
    #     # top = self.winfo_toplevel()
    #     # top.rowconfigure(0, weight=1)
    #     # top.columnconfigure(0, weight=1)
    #     # self.rowconfigure(0, weight=1)
    #     # self.columnconfigure(0, weight=1)
    #     # img_data = self.load_image()
    #     # pil_img = PIL.Image.fromarray(img_data, 'RGB')
    #     # tk_img = ImageTk.PhotoImage(pil_img)
    #     # self.label = tk.Label(self.master, image=self.tk_img).grid(row=0, column=0)
    #     # self.label.my_image = tk_img  # Used only to prevent image being destroy by garbage collector
    #     # -----------------------------------------------
    #
    # def show_edit_panel(self):
    #     edit_frame = tk.Frame(self, bd=1)
    #     self.wait_variable(self.active_window)
    #     image = self.open_images[self.active_window.get()]
    #     self.edited_image = np.copy(image).astype(float)
    #     linear_image = Util.linear_transform(self.edited_image)
    #     tk.Label(edit_frame, image=linear_image).grid()
    #     # edited_image_label.config(image = self.edited_image)
    #     self.edit_panel = edit_frame
    #     self.edit_panel.grid()
    #     tk.Button(edit_frame, text='Discard', command=lambda: self.hide_edit_panel) \
    #         .grid(columnspan=2, sticky=(tk.W, tk.E))
    #     tk.Button(edit_frame, text='Finish', command=lambda: self.create_new_image(self.edited_image)) \
    #         .grid(columnspan=2, sticky=(tk.W, tk.E))
    #     return edit_frame
    #
    # def hide_edit_panel(self):
    #     self.edit_panel.grid_remove()

    # Transform Menu Functions
    def negative(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.negative(image)
        self.create_new_image(transformed_img)

    def contrast(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.contrast_increase(image, self.s1.get(), self.s2.get())
        self.create_new_image(transformed_img)

    def dynamic_compression(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.dynamic_range_compression(image)

        sad = transformed_img.flatten()
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
        transformed_img = Util.add_noise_rayleigh(image, self.rayleigh_scale.get(), self.rayleigh_prob.get())
        self.create_new_image(transformed_img)

    def exp_noise(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.add_noise_exponential(image, self.exp_scale.get(), self.exp_prob.get())
        self.create_new_image(transformed_img)

    def salt_pepper_noise(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = Util.add_comino_and_sugar_noise(image, self.salt_pepper_p0.get(), self.salt_pepper_p1.get(),
                                                          self.salt_pepper_density.get())
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
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.median_filter(image)
        self.create_new_image(transformed_img)

    def w_median_filter(self):
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.median_filter(image, weighted=True)
        self.create_new_image(transformed_img)

    def gauss_filter(self):
        size = self.gauss_filter_size.get().split()
        gauss_filter_size = (int(size[0]), int(size[1]))
        self.wait_variable(self.active_window)
        image = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.gauss_blur(image, gauss_filter_size, self.gauss_filter_sigma.get())
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
    app.master.geometry('500x530')
    app.mainloop()
