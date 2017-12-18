import threading
import time

# import cv2
import matplotlib
from cv2 import cv2

from input.logGabor import LogGabor

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from src.input.util import Util
from src.input.filter_provider import FilterProvider
from src.input.vector_util import VectorUtil
from src.input.provider import Provider
from src.input.border_detectors import BorderDetector
from PIL import ImageTk
from src.input.feature_detector import FeaturesDetector
import tkinter as tk
import tkinter.ttk as ttk
import PIL
import numpy as np
import os

SUSAN_BORDER_DETECTOR = 0
SUSAN_CORNER_DETECTOR = 1
SUSAN_BORDER_CORNER_DETECTOR = 2


class ImageEditor(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        # Make the root window and the main frame responsive to resize
        self.changed = False
        self.test = 56
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

        self.x_coord = tk.IntVar()
        self.y_coord = tk.IntVar()
        self.r_value = tk.DoubleVar()
        self.g_value = tk.DoubleVar()
        self.b_value = tk.DoubleVar()
        self.r_average = tk.DoubleVar()
        self.g_average = tk.DoubleVar()
        self.b_average = tk.DoubleVar()
        self.selection_square = None
        self.is_selected = False

    def create_menu(self):
        root = self.master

        # Menu Bar
        menu_bar = tk.Menu(root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        generate_menu = tk.Menu(menu_bar, tearoff=0)
        operations_menu = tk.Menu(menu_bar, tearoff=0)
        transform_menu = tk.Menu(menu_bar, tearoff=0)
        noise_menu = tk.Menu(menu_bar, tearoff=0)
        filter_menu = tk.Menu(menu_bar, tearoff=0)
        borders_menu = tk.Menu(menu_bar, tearoff=0)
        feature_detectors_menu = tk.Menu(menu_bar, tearoff=0)

        # File Menu
        file_menu.add_command(label='Load', command=self.load_image)
        file_menu.add_command(label='Load Color', command=self.load_color_image)
        file_menu.add_command(label='Load Sequence', command=self.load_sequence)
        file_menu.add_command(label='Edit', command=self.edit_image)
        file_menu.add_command(label='Save', command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label='Save selection')
        file_menu.add_command(label='Histogram', command=self.histogram)
        file_menu.add_separator()
        file_menu.add_command(label='Quit', command=root.destroy)
        menu_bar.add_cascade(label='File', menu=file_menu)

        # Generate Menu
        generate_menu.add_command(label='Gray Gradient', command=self.sinthetic_image_builder(Provider.gray_gradient))
        generate_menu.add_command(label='Color Gradient', command=self.sinthetic_image_builder(Provider.color_gradient))
        generate_menu.add_command(label='Circle', command=self.sinthetic_image_builder(Provider.draw_circle))
        generate_menu.add_command(label='Square', command=self.sinthetic_image_builder(Provider.draw_square))
        menu_bar.add_cascade(label='Generate', menu=generate_menu)

        # Transform Menu
        transform_menu.add_command(label='Negative', command=self.negative)
        transform_menu.add_command(label='Contrast', command=self.contrast)
        transform_menu.add_command(label='Compression', command=self.dynamic_compression)
        transform_menu.add_command(label='Gamma', command=self.gamma_function)
        transform_menu.add_command(label='Binary', command=self.to_binary)
        transform_menu.add_command(label='Equalize', command=self.equalize)
        transform_menu.add_command(label='Global Threshold', command=self.thresholdg)
        transform_menu.add_command(label='Otsu Threshold', command=self.thresholdo)
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
        filter_menu.add_command(label='Anisotropic', command=self.anisotropic_filter)
        menu_bar.add_cascade(label='Filter', menu=filter_menu)

        # Borders Menu
        borders_menu.add_command(label='Prewitt', command=self.directional_borders_prewitt)
        borders_menu.add_command(label='Sobel', command=self.directional_borders_sobel)
        borders_menu.add_command(label='Laplace', command=self.laplace_borders)
        borders_menu.add_command(label='Gaussian Laplace', command=self.gaussian_laplace_borders)
        borders_menu.add_command(label='Susan', command=self.susan_border_detector)
        borders_menu.add_command(label='Hough', command=self.hough_transform)
        borders_menu.add_command(label='Canny', command=self.canny_edges)
        borders_menu.add_command(label='Active Contours', command=self.active_contours)
        borders_menu.add_command(label='Harris', command=self.harris_corner_detector)
        menu_bar.add_cascade(label='Borders', menu=borders_menu)

        # Features Detector Menu
        feature_detectors_menu.add_command(label='SIFT compare', command=self.SIFT_compare)
        feature_detectors_menu.add_command(label='SIFT', command=self.SIFT_single)
        feature_detectors_menu.add_command(label='Iris Template', command=self.iris_detector)
        feature_detectors_menu.add_command(label='Iris Compare', command=self.compare_iris)
        menu_bar.add_cascade(label='Features Detectors', menu=feature_detectors_menu)

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
        self.gauss_filter_sigma = tk.DoubleVar()
        self.gauss_filter_sigma.set(1)
        self.anisotropic_iter = tk.IntVar()
        self.anisotropic_iter.set(5)
        self.anisotropic_m = tk.IntVar()
        self.anisotropic_m.set(0)
        self.anisotropic_sigma = tk.DoubleVar()
        self.anisotropic_sigma.set(3)

        # Global Threshold
        self.threshold = tk.DoubleVar()
        self.threshold.set(100)
        self.delta = tk.DoubleVar()
        self.delta.set(1)
        self.deltaT = tk.DoubleVar()
        self.deltaT.set(200)

        # Borders
        self.gaussian_laplace_sigma = tk.DoubleVar()
        self.gaussian_laplace_sigma.set(1)
        self.laplace_threshold = tk.DoubleVar()
        self.laplace_threshold.set(4)
        self.borders_detectors_directions = tk.StringVar()
        self.borders_detectors_directions.set('0 1 2 3')

        # Susan
        self.susan_type = tk.IntVar()
        self.susan_type.set(2)
        self.susan_delta = tk.DoubleVar()
        self.susan_delta.set(0.15)

        # Hough
        self.hough_theta_steps = tk.IntVar()
        self.hough_theta_steps.set(4)
        self.hough_p_steps = tk.IntVar()
        self.hough_p_steps.set(20)
        self.hough_threshold = tk.DoubleVar()
        self.hough_threshold.set(0.5)

        # Canny
        self.canny_sigma1 = tk.IntVar()
        self.canny_sigma1.set(2)
        self.canny_sigma2 = tk.IntVar()
        self.canny_sigma2.set(2)

        # Canny
        self.harris_threshold = tk.DoubleVar()
        self.harris_threshold.set(0.008)

    def create_settings_window(self):
        settings_frame = tk.Frame(self)
        self.settings_row = 0

        def curr_row():
            return self.settings_row

        def next_row():
            self.settings_row = self.settings_row + 1
            return self.settings_row

        # Title
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=curr_row(), columnspan=4, sticky=(tk.W, tk.E))
        tk.Label(settings_frame, text='Settings').grid(row=curr_row(), columnspan=4)

        # Contrast
        tk.Label(settings_frame, text='Contrast S1').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.s1, textvariable=self.s1).grid(row=curr_row(), column=1)
        tk.Label(settings_frame, text='Contrast S2').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.s2, textvariable=self.s2).grid(row=curr_row(), column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        # Gamma
        tk.Label(settings_frame, text='Gamma').grid(row=next_row(), column=0)
        # tk.Scale(settings_frame, variable=self.gamma, from_=0, to=3, orient=tk.HORIZONTAL).grid(row=4, column=1)
        tk.Entry(settings_frame, text=self.gamma, textvariable=self.gamma).grid(row=curr_row(), column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        # Binary
        tk.Label(settings_frame, text='Binary Threshold').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.binary_threshold, textvariable=self.binary_threshold).grid(row=curr_row(),
                                                                                                      column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        # Scalar Product
        tk.Label(settings_frame, text='Scalar Product').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.scalar, textvariable=self.scalar).grid(row=curr_row(), column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        # Noise
        tk.Label(settings_frame, text='Gauss Noise Deviation').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.normal_sigma, textvariable=self.normal_sigma).grid(row=curr_row(), column=1)

        tk.Label(settings_frame, text='Gauss Noise Density').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.normal_prob, textvariable=self.normal_prob).grid(row=curr_row(), column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Rayleigh Noise Scale').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.rayleigh_scale, textvariable=self.rayleigh_scale).grid(row=curr_row(),
                                                                                                  column=1)

        tk.Label(settings_frame, text='Rayleigh Noise Density').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.rayleigh_prob, textvariable=self.rayleigh_prob).grid(row=curr_row(),
                                                                                                column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Exp Noise Scale').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.exp_scale, textvariable=self.exp_scale).grid(row=curr_row(), column=1)

        tk.Label(settings_frame, text='Exp Noise Density').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.exp_prob, textvariable=self.exp_prob).grid(row=curr_row(), column=1)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Salt Pepper Noise P0').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.salt_pepper_p0, textvariable=self.salt_pepper_p0).grid(row=curr_row(),
                                                                                                  column=1)
        tk.Label(settings_frame, text='Salt Pepper Noise P1').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.salt_pepper_p1, textvariable=self.salt_pepper_p1).grid(row=curr_row(),
                                                                                                  column=1)

        tk.Label(settings_frame, text='Salt Pepper Noise Density').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.salt_pepper_density, textvariable=self.salt_pepper_density).grid(
            row=curr_row(), column=1)

        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Mean Filter Size').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.mean_filter_size, textvariable=self.mean_filter_size).grid(row=curr_row(),
                                                                                                      column=1)

        tk.Label(settings_frame, text='Gauss Filter Deviation').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.gauss_filter_sigma, textvariable=self.gauss_filter_sigma).grid(
            row=curr_row(), column=1)

        tk.Label(settings_frame, text='Anisotropic M').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.anisotropic_m, textvariable=self.anisotropic_m).grid(row=curr_row(),
                                                                                                column=1)
        tk.Label(settings_frame, text='Anisotropic Tmax').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.anisotropic_iter, textvariable=self.anisotropic_iter).grid(row=curr_row(),
                                                                                                      column=1)
        tk.Label(settings_frame, text='Anisotropic Deviation').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.anisotropic_sigma, textvariable=self.anisotropic_sigma).grid(row=curr_row(),
                                                                                                        column=1)
        # Global Threshold
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Global Threshold').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.threshold, textvariable=self.threshold).grid(row=curr_row(), column=1)

        tk.Label(settings_frame, text='Global Threshold Delta').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.delta, textvariable=self.delta).grid(row=curr_row(), column=1)

        # tk.Label(settings_frame, text='Global Threshold DeltaT').grid(row=next_row(), column=0)
        # tk.Entry(settings_frame, text=self.deltaT, textvariable=self.deltaT).grid(row=curr_row(), column=1)

        # Border Detectors
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=2, sticky=(tk.W, tk.E))

        tk.Label(settings_frame, text='Prewitt/Sobel Directions').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.borders_detectors_directions, textvariable=self.borders_detectors_directions) \
            .grid(row=curr_row(), column=1)

        tk.Label(settings_frame, text='Gaussian Laplace Deviation').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.gaussian_laplace_sigma, textvariable=self.gaussian_laplace_sigma).grid(
            row=curr_row(),
            column=1)

        tk.Label(settings_frame, text='Laplace Threshold').grid(row=next_row(), column=0)
        tk.Entry(settings_frame, text=self.laplace_threshold, textvariable=self.laplace_threshold) \
            .grid(row=curr_row(), column=1)

        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(columnspan=4, sticky=(tk.W, tk.E))
        tk.Button(settings_frame, text='Return', command=self.hide_settings).grid(columnspan=4, sticky=(tk.W, tk.E))

        ####  Column 3&4  ####
        self.settings_row = 1

        # Susan Detector
        tk.Label(settings_frame, text='Susan Detector').grid(row=curr_row(), column=2)
        tk.Entry(settings_frame, text=self.susan_type, textvariable=self.susan_type).grid(row=curr_row(),
                                                                                          column=3)
        tk.Label(settings_frame, text='Susan Delta').grid(row=next_row(), column=2)
        tk.Entry(settings_frame, text=self.susan_delta, textvariable=self.susan_delta).grid(row=curr_row(),
                                                                                            column=3)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), column=2, columnspan=2,
                                                                 sticky=(tk.W, tk.E))
        # Hough
        tk.Label(settings_frame, text='Hough Theta Steps').grid(row=next_row(), column=2)
        tk.Entry(settings_frame, text=self.hough_theta_steps, textvariable=self.hough_theta_steps).grid(row=curr_row(),
                                                                                                        column=3)
        tk.Label(settings_frame, text='Hough P Steps').grid(row=next_row(), column=2)
        tk.Entry(settings_frame, text=self.hough_p_steps, textvariable=self.hough_p_steps).grid(row=curr_row(),
                                                                                                column=3)
        tk.Label(settings_frame, text='Hough Threshold').grid(row=next_row(), column=2)
        tk.Entry(settings_frame, text=self.hough_threshold, textvariable=self.hough_threshold).grid(row=curr_row(),
                                                                                                    column=3)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), column=2, columnspan=2,
                                                                 sticky=(tk.W, tk.E))
        # Canny
        tk.Label(settings_frame, text='Canny Sigma 1').grid(row=next_row(), column=2)
        tk.Entry(settings_frame, text=self.canny_sigma1, textvariable=self.canny_sigma1).grid(row=curr_row(), column=3)
        tk.Label(settings_frame, text='Canny Sigma 2').grid(row=next_row(), column=2)
        tk.Entry(settings_frame, text=self.canny_sigma2, textvariable=self.canny_sigma2).grid(row=curr_row(), column=3)
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=next_row(), column=2, columnspan=2,
                                                                 sticky=(tk.W, tk.E))

        # Harris
        tk.Label(settings_frame, text='Harris Threshold').grid(row=next_row(), column=2)
        tk.Entry(settings_frame, text=self.harris_threshold, textvariable=self.harris_threshold).grid(row=curr_row(),
                                                                                                      column=3)

        return settings_frame

    def show_settings(self):
        self.settings.grid()
        # self.settings.grid(sticky=tk.N + tk.S + tk.E + tk.W)

    def hide_settings(self):
        self.settings.grid_remove()

    # File Menu Functions
    def load_image(self, color=False):
        img_path = tk.filedialog.askopenfilename(initialdir='../resources/test', title='Select Image')
        img_data = Util.load_image(img_path)
        title = img_path.split('/')
        title = title[len(title) - 1]
        self.create_new_image(img_data, title, color)

    def load_color_image(self):
        self.load_image(True)

    def load_sequence(self):
        directory_path = tk.filedialog.askdirectory()
        directory = os.fsencode(directory_path)
        images = []
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if not filename.startswith('.'):
                images.append(Util.load_image(directory_path + '/' + filename))
        self.create_new_video(images, directory_path, color=True)

    def save_image(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        img_path = tk.filedialog.asksaveasfilename(initialdir='../resources/test', title='Save Image')
        linear_image = Util.linear_transform(image)
        # TODO: Change this for a generic save method that checks on the img_path extension
        color = False
        Util.save(linear_image, img_path, color)

    def create_new_image(self, img_data, title='', color=False):
        linear_img = Util.linear_transform(img_data)
        pil_img = PIL.Image.fromarray(linear_img, 'RGB')
        tk_img = ImageTk.PhotoImage(pil_img)

        # Matrix shape is (height, width, 3)
        height = img_data.shape[0]
        width = img_data.shape[1]

        new_window = tk.Toplevel()
        new_window.geometry('{}x{}'.format(width, height))
        new_window.resizable(width=False, height=False)
        if title.__len__() == 0:
            new_window.title('Image {}'.format(self.image_number))
        else:
            new_window.title('{} {}'.format(title, self.image_number))
        new_window.bind('<ButtonRelease-1>', self.set_active_window)
        new_window.bind('<Destroy>', self.remove_open_image)

        self.image_number += 1
        canvas = tk.Canvas(new_window, width=width, height=height, borderwidth=0, highlightthickness=0)
        canvas.grid(row=0, column=0)
        canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
        canvas.my_image = tk_img  # Used only to prevent image being destroy by garbage collector
        self.open_images[new_window.title()] = img_data, color, canvas
        return self.open_images[new_window.title()]

    def create_new_video(self, img_data_array, title='', color=False, play=False):
        linear_img_array = []
        for img_data in img_data_array:
            linear_img_array.append(Util.linear_transform(img_data))

        tk_img_array = []
        for linear_img in linear_img_array:
            pil_img = PIL.Image.fromarray(linear_img, 'RGB')
            tk_img_array.append(ImageTk.PhotoImage(pil_img))

        # Matrix shape is (height, width, 3)
        height = img_data_array[0].shape[0]
        width = img_data_array[0].shape[1]

        new_window = tk.Toplevel()
        new_window.geometry('{}x{}'.format(width, height))
        new_window.resizable(width=False, height=False)
        if title.__len__() == 0:
            new_window.title('Image {}'.format(self.image_number))
        else:
            new_window.title('{} {}'.format(title, self.image_number))
        new_window.bind('<ButtonRelease-1>', self.set_active_window)
        new_window.bind('<Destroy>', self.remove_open_image)

        self.image_number += 1
        canvas = tk.Canvas(new_window, width=width, height=height, borderwidth=0, highlightthickness=0)
        canvas.grid(row=0, column=0)
        canvas.create_image(0, 0, image=tk_img_array[0], anchor=tk.NW)
        canvas.my_image = tk_img_array[0]  # Used only to prevent image being destroy by garbage collector

        self.open_images[new_window.title()] = img_data_array, color, canvas
        if play:
            class CanvasRefresher(threading.Thread):
                def __init__(self, video_canvas, video_images):
                    threading.Thread.__init__(self)
                    self.canvas = video_canvas
                    self.images = video_images

                def run(self):
                    while 1:
                        for frame in self.images:
                            time.sleep(1)
                            self.canvas.create_image(0, 0, image=frame, anchor=tk.NW)
                            self.canvas.my_image = frame

            thread1 = CanvasRefresher(canvas, tk_img_array)
            thread1.start()

    def set_active_window(self, event):
        self.active_window.set(event.widget.winfo_toplevel().title())

    def remove_open_image(self, event):
        self.open_images.pop(event.widget.winfo_toplevel().title(), None)

    def histogram(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        plt.hist(image.flatten(), bins='auto', normed=1)
        plt.show()

    def edit_image(self):
        self.wait_variable(self.active_window)
        img_data = self.open_images[self.active_window.get()]

        # Auxiliar functions for building the grid
        self.edit_panel_row = 0

        def curr_row():
            return self.edit_panel_row

        def next_row():
            self.edit_panel_row = self.edit_panel_row + 1
            return self.edit_panel_row

        linear_img = Util.linear_transform(img_data)
        pil_img = PIL.Image.fromarray(linear_img, 'RGB')
        tk_img = ImageTk.PhotoImage(pil_img)
        height = img_data.shape[0]
        width = img_data.shape[1]

        root = self.master
        canvas = tk.Canvas(root, width=width, height=height, borderwidth=0, highlightthickness=0)
        canvas.grid(row=curr_row(), column=1)
        canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)

        canvas.my_image = tk_img  # Used only to prevent image being destroy by garbage collector

        self.edited_img = tk_img
        self.edited_img_data = np.copy(img_data).astype(float)
        self.edited_img_canvas = canvas

        self.coord_frame = tk.Frame(self)
        tk.Label(self.coord_frame, text='(x,y)').grid(row=curr_row(), column=0, columnspan=3)
        tk.Entry(self.coord_frame, text=self.x_coord, textvariable=self.x_coord, width=5).grid(row=next_row(), column=0)
        tk.Entry(self.coord_frame, text=self.y_coord, textvariable=self.y_coord, width=5).grid(row=curr_row(), column=1)
        tk.Button(self.coord_frame, text='Set Value', command=self.__set_value).grid(row=curr_row(), column=2, padx=10)

        tk.Label(self.coord_frame, text='(r,g,b)').grid(row=next_row(), column=0, columnspan=3)
        tk.Entry(self.coord_frame, text=self.r_value, textvariable=self.r_value, width=5).grid(row=next_row(), column=0)
        tk.Entry(self.coord_frame, text=self.g_value, textvariable=self.g_value, width=5).grid(row=curr_row(), column=1)
        tk.Entry(self.coord_frame, text=self.b_value, textvariable=self.b_value, width=5).grid(row=curr_row(), column=2)

        tk.Label(self.coord_frame, text='Gray Level Average').grid(row=next_row(), column=0, columnspan=3)
        tk.Label(self.coord_frame, text=self.r_average, textvariable=self.r_average, width=5).grid(row=next_row(),
                                                                                                   column=0)
        tk.Label(self.coord_frame, text=self.g_average, textvariable=self.g_average, width=5).grid(row=curr_row(),
                                                                                                   column=1)
        tk.Label(self.coord_frame, text=self.b_average, textvariable=self.b_average, width=5).grid(row=curr_row(),
                                                                                                   column=2)

        ttk.Separator(self.coord_frame, orient=tk.HORIZONTAL).grid(row=next_row(), columnspan=3, sticky=(tk.W, tk.E),
                                                                   pady=10)

        tk.Button(self.coord_frame, text='Cancel', command=self.hide_edit_panel).grid(row=next_row(), column=0,
                                                                                      columnspan=1, pady=10)
        tk.Button(self.coord_frame, text='Save', command=self.save_edition).grid(row=curr_row(), column=2, columnspan=1,
                                                                                 pady=10)

        # TODO: return coord_frame and save it to hide it later (copy create_settings method)
        self.coord_frame.grid(row=0, column=0)

        canvas.bind("<ButtonPress>", self.__pixel_selection)
        canvas.bind("<B1-Motion>", self.__range_selection)

    # Edit Private Functions
    def __pixel_selection(self, event):
        self.is_selected = False
        self.edited_img_canvas.delete(self.selection_square)
        self.x_coord.set(event.x)
        self.y_coord.set(event.y)
        self.r_value.set(self.edited_img_data[self.y_coord.get(), self.x_coord.get(), 0])
        self.g_value.set(self.edited_img_data[self.y_coord.get(), self.x_coord.get(), 1])
        self.b_value.set(self.edited_img_data[self.y_coord.get(), self.x_coord.get(), 2])

    def __clear_selection_square(self):
        self.is_selected = False
        self.edited_img_canvas.delete(self.selection_square)

    def __range_selection(self, event):
        width = self.edited_img_data.shape[0]
        height = self.edited_img_data.shape[1]
        self.is_selected = True
        self.last_x = event.x
        self.last_y = event.y

        # Fix the last point of the square to be inside the image
        if self.last_x >= width:
            self.last_x = width - 1
        elif self.last_x < 0:
            self.last_x = 0

        if self.last_y >= height:
            self.last_y = height - 1
        elif self.last_y < 0:
            self.last_y = 0

        self.edited_img_canvas.delete(self.selection_square)
        self.selection_square = self.edited_img_canvas.create_rectangle(self.x_coord.get(), self.y_coord.get(),
                                                                        self.last_x, self.last_y, fill="#8E3840",
                                                                        width=0, stipple="gray50")

        # Util.color_average(self.edited_img_data, (self.x_coord.get(), self.y_coord.get()), (self.last_x, self.last_y))

        # TODO: Fix get_info method to work with 3D matrix. Show it's value on labels.
        # Maybe convert last_x and last_y to IntVar and show the values on screen to know where the average is
        # Add a button to save the selection
        # Add padding to Gray Level Average and replace the text for the (x1, y1), (x2, y2) or smth

    def __set_value(self):
        x, y = self.x_coord.get(), self.y_coord.get()
        r, g, b = self.r_value.get(), self.g_value.get(), self.b_value.get()
        self.edited_img_data[y, x, 0] = r
        self.edited_img_data[y, x, 1] = g
        self.edited_img_data[y, x, 2] = b

        linear_img = Util.linear_transform(self.edited_img_data)
        pil_img = PIL.Image.fromarray(linear_img, 'RGB')
        tk_img = ImageTk.PhotoImage(pil_img)
        self.edited_img_canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
        self.edited_img_canvas.my_image = tk_img  # Used only to prevent image being destroy by garbage collector

    def hide_edit_panel(self):
        self.edited_img_canvas.delete('all')
        self.coord_frame.grid_remove()
        # self.edit_panel.grid_remove()

    def save_edition(self):
        self.create_new_image(self.edited_img_data)

    # Generate Menu Functions
    def sinthetic_image_builder(self, draw_function):
        def draw():
            image = draw_function()
            self.create_new_image(image)

        return draw

    # Transform Menu Functions
    def negative(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Util.negative(image)
        self.create_new_image(transformed_img)

    def contrast(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Util.contrast_increase(image, self.s1.get(), self.s2.get(), color)
        self.create_new_image(transformed_img)

    def dynamic_compression(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Util.dynamic_range_compression(image, color)

        sad = transformed_img.flatten()
        self.create_new_image(transformed_img)

    def gamma_function(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Util.gamma_power(image, self.gamma.get(), color)
        self.create_new_image(transformed_img)

    def equalize(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Provider.equalize_histogram(image)
        self.create_new_image(transformed_img)

    def to_binary(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Util.to_binary(image, self.binary_threshold.get())
        self.create_new_image(transformed_img)

    # Operations Functions
    def add(self):
        self.wait_variable(self.active_window)
        image1, color, canvas = self.open_images[self.active_window.get()]
        self.wait_variable(self.active_window)
        image2, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Util.sum(image1, image2, color)
        self.create_new_image(transformed_img)

    def difference(self):
        self.wait_variable(self.active_window)
        image1, color, canvas = self.open_images[self.active_window.get()]
        self.wait_variable(self.active_window)
        image2, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Util.difference(image1, image2)
        self.create_new_image(transformed_img)

    def multiply(self):
        self.wait_variable(self.active_window)
        image1, color, canvas = self.open_images[self.active_window.get()]
        self.wait_variable(self.active_window)
        image2, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Util.multiply(image1, image2, color)
        self.create_new_image(transformed_img)

    def scalar_product(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Util.scalar_prod(image, self.scalar.get())
        self.create_new_image(transformed_img)

    # Noise Functions
    def normal_noise(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Util.add_additive_noise_normal(image,
                                                         self.normal_mu.get(), self.normal_sigma.get(),
                                                         self.normal_prob.get(), color)
        self.create_new_image(transformed_img)

    def rayleigh_noise(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Util.add_noise_rayleigh(image, self.rayleigh_scale.get(), self.rayleigh_prob.get(), color)
        self.create_new_image(transformed_img)

    def exp_noise(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Util.add_noise_exponential(image, self.exp_scale.get(), self.exp_prob.get(), color)
        self.create_new_image(transformed_img)

    def salt_pepper_noise(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = Util.add_comino_and_sugar_noise(image, self.salt_pepper_p0.get(), self.salt_pepper_p1.get(),
                                                          self.salt_pepper_density.get())
        self.create_new_image(transformed_img)

    # Filter Functions
    def mean_filter(self):
        size = self.mean_filter_size.get().split()
        mean_filter_size = (int(size[0]), int(size[1]))
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.blur(image, mean_filter_size, color)
        self.create_new_image(transformed_img)

    def median_filter(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.median_filter(image, independent_layer=color)
        self.create_new_image(transformed_img)

    def w_median_filter(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.median_filter(image, weighted=True, independent_layer=color)
        self.create_new_image(transformed_img)

    def gauss_filter(self):
        sigma = int(self.gauss_filter_sigma.get())
        gauss_filter_size = (2 * sigma + 1, 2 * sigma + 1)
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.gauss_blur(image, gauss_filter_size, sigma, color)
        self.create_new_image(transformed_img)

    def anisotropic_filter(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.anisotropic_filter(
            image, self.anisotropic_iter.get(), self.anisotropic_m.get(), self.anisotropic_sigma.get(), color)
        self.create_new_image(transformed_img)

    def borders_filter(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = FilterProvider.pasa_altos(image)
        self.create_new_image(transformed_img)

    # The arrow points to the negative numbers
    # 0: DOWN, 1: DOWN-LEFT, 2: LEFT, 3: UP-LEFT
    def directional_borders_prewitt(self, weighted=False):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        directions_str = self.borders_detectors_directions.get().split()
        directions = []
        for i in range(len(directions_str)):
            directions.append(int(directions_str[i]))
        transformed_img = FilterProvider.directional_border_detector(image, weighted, directions,
                                                                     independent_layer=color)
        self.create_new_image(transformed_img)

    def directional_borders_sobel(self):
        self.directional_borders_prewitt(True)

    def laplace_borders(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = BorderDetector.laplacian_detector(image, self.laplace_threshold.get(), color)
        self.create_new_image(transformed_img)

    def gaussian_laplace_borders(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = BorderDetector.laplacian_gaussian_detector(image,
                                                                     self.gaussian_laplace_sigma.get(),
                                                                     self.laplace_threshold.get(), color)
        self.create_new_image(transformed_img)

    # 0: BORDER_DETECTOR, 1: CORNER_DETECTOR, 2: BORDER AND CORNER
    def susan_border_detector(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = BorderDetector.susan_border_detector(
            image=image, independent_layer=color,
            detector_type=self.susan_type.get(), delta=self.susan_delta.get())
        self.create_new_image(transformed_img)

    def harris_corner_detector(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        # Someday
        # if color:
        # gray_image = cv2.cvtColor(image.astype('B'), cv2.COLOR_BGR2GRAY)
        # print(gray_image.shape)
        # print(gray_image)
        # self.create_new_image(img_data=gray_image, color=color)
        # else:
        #     gray_image = image
        transformed_img = BorderDetector.harris_corner_detector(image=image, independent_layer=False)
        transformed_img = self.harris_draw_corners(image, transformed_img, self.harris_threshold.get())
        self.create_new_image(transformed_img)

    def harris_draw_corners(self, image, transformed_img, p=0.1):
        threshold = np.max(transformed_img) * p
        print('[INFO] Harris Corner Detector Calculated Threshold: ' + str(threshold))
        ans = np.copy(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if transformed_img[i, j, 0] >= threshold:
                    ans[i, j, 0] = 255
                    ans[i, j, 1] = 0
                    ans[i, j, 2] = 0
        return ans

    # [WARN] Pass only greyscale images!
    def hough_transform(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        lines, points, theta_range, p_range = BorderDetector.hough_transform(
            image, self.hough_theta_steps.get(), self.hough_p_steps.get())
        new_image, new_color, new_canvas = self.create_new_image(np.copy(image))
        threshold = np.max(lines) * self.hough_threshold.get()
        for i in range(theta_range.size):
            for j in range(p_range.size):
                if lines[i, j] > threshold:
                    self.draw_single_hough_line(points[(i, j)], theta_range[i], p_range[j], new_canvas)
        print('[FINISHED] Hough Transform')

    def draw_single_hough_line(self, points, theta, p, canvas):
        mini = points[0]
        maxi = points[len(points) - 1]
        canvas.create_line(mini[1], mini[0], maxi[1], maxi[0], fill='red', width=5)

    def canny_edges(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = BorderDetector.canny_edges(
            self.canny_sigma1.get(), self.canny_sigma2.get(), image=image, color=color)
        self.create_new_image(transformed_img)

    # todo
    def active_contours(self):
        print('active contours')
        self.wait_variable(self.active_window)
        print('window found')

        start_point = None
        end_point = None
        object_rectangle = None
        object_color = None
        background_color = None

        def start_object_selection(event):
            self.__pixel_selection(event)
            global start_point
            start_point = (event.y, event.x)
            print('object', event.y, event.x)
            print('find me', self.test)
            canvas.bind('<ButtonPress-1>', start_object_selection)
            canvas.bind('<ButtonRelease-1>', end_object_selection)

        def end_object_selection(event):
            self.__clear_selection_square()
            global end_point
            global object_rectangle
            global start_point
            global object_color
            print('object', event.y, event.x)
            print('start point object', start_point[0], start_point[1])
            end_point = (event.y, event.x)
            object_rectangle = (start_point, end_point)
            color_acu = [0, 0, 0]
            min_x = np.min([start_point[0], end_point[0]])
            max_x = np.max([start_point[0], end_point[0]])
            min_y = np.min([start_point[1], end_point[1]])
            max_y = np.max([start_point[1], end_point[1]])
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    color_acu[0] += image[x, y, 0]
                    color_acu[1] += image[x, y, 1]
                    color_acu[2] += image[x, y, 2]
            pixels_count = ((max_x - min_x + 1) *
                            (max_y - min_y + 1))
            object_color = [color_acu[0] / pixels_count, color_acu[1] / pixels_count, color_acu[2] / pixels_count]
            canvas.unbind('<ButtonPress-1>')
            canvas.unbind('<ButtonRelease-1>')
            canvas.bind('<ButtonPress-1>', start_background_selection)
            canvas.bind('<ButtonRelease-1>', end_background_selection)

        def start_background_selection(event):
            self.__pixel_selection(event)
            global start_point
            start_point = (event.y, event.x)
            print('background', event.y, event.x)

        def end_background_selection(event):
            self.__clear_selection_square()
            global start_point
            global end_point
            global background_color
            global object_color
            global object_rectangle
            end_point = (event.y, event.x)
            print('background', event.y, event.x)

            color_acu = [0, 0, 0]
            min_x = np.min([start_point[0], end_point[0]])
            max_x = np.max([start_point[0], end_point[0]])
            min_y = np.min([start_point[1], end_point[1]])
            max_y = np.max([start_point[1], end_point[1]])
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    color_acu[0] += image[x, y, 0]
                    color_acu[1] += image[x, y, 1]
                    color_acu[2] += image[x, y, 2]
            pixels_count = ((max_x - min_x + 1) *
                            (max_y - min_y + 1))
            background_color = [color_acu[0] / pixels_count, color_acu[1] / pixels_count, color_acu[2] / pixels_count]

            print('object color', object_color)
            print('background color', background_color)

            canvas.unbind('<ButtonPress-1>')
            canvas.unbind('<ButtonRelease-1>')
            canvas.bind('<ButtonRelease-1>', self.set_active_window)
            initial_state = BorderDetector.generate_active_contour_initial_state(image, object_rectangle)

            self.create_new_video(
                BorderDetector.active_contour_sequence(images, initial_state, background_color, object_color,
                                                       algorithm=1), color=True, play=True)

        def set_edited_image(img_data, canvas):
            self.edited_img = canvas.my_image
            self.edited_img_data = np.copy(img_data).astype(float)
            self.edited_img_canvas = canvas

        images, color, canvas = self.open_images[self.active_window.get()]
        image = images[0]
        set_edited_image(image, canvas)
        canvas.bind('<ButtonPress-1>', start_object_selection)
        canvas.bind("<B1-Motion>", self.__range_selection)
        canvas.bind('<ButtonRelease-1>', end_object_selection)

        print('active contours end')

    def thresholdg(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = np.zeros(image.shape)
        aux = []
        (width, height, layers) = image.shape
        for d in range(layers):
            t = BorderDetector.global_threshold(image[:, :, d], self.threshold.get(), self.delta.get(),
                                                self.deltaT.get())
            print('Global threshold: {}'.format(t))
            aux.append(Util.to_binary(image[:, :, d], t))
            if not color:
                aux.append(aux[0])
                aux.append(aux[0])
                break

        for x in range(width):
            for y in range(height):
                pixel = []
                for d in range(layers):
                    pixel.append(aux[d][x, y])
                transformed_img[x, y] = pixel
        self.create_new_image(transformed_img)

    def thresholdo(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = np.zeros(image.shape)
        aux = []
        (width, height, layers) = image.shape
        for d in range(layers):
            t = BorderDetector.otsu_threshold(image[:, :, d])
            print('Otsu threshold: {}'.format(t))
            aux.append(Util.to_binary(image[:, :, d], t))
            if (not color):
                aux.append(aux[0])
                aux.append(aux[0])
                break

        for x in range(width):
            for y in range(height):
                pixel = []
                for d in range(layers):
                    pixel.append(aux[d][x, y])
                transformed_img[x, y] = pixel

        # t = BorderDetector.otsu_threshold(image)
        # # Don't delete this print, it gives info. about the image
        # print('Otsu threshold: {}'.format(t))
        # transformed_img = Util.to_binary(image, t)
        self.create_new_image(transformed_img)

    def SIFT_compare(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        self.wait_variable(self.active_window)
        image2, color2, canvas2 = self.open_images[self.active_window.get()]
        transformed_img = FeaturesDetector.SIFT(image, image2)
        self.create_new_image(transformed_img)

    def SIFT_single(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        transformed_img = FeaturesDetector.SIFT_single(image)
        self.create_new_image(transformed_img)

    def iris_detector(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        image = cv2.cvtColor(image.astype('B'), cv2.COLOR_BGR2GRAY)

        def select_circle_center(handle_select_release, handle_select_radius):
            def select_center(event):
                select_circle_center.center = (event.x, event.y)
                canvas.unbind('<ButtonPress-1>')
                canvas.bind("<B1-Motion>", handle_select_radius)
                canvas.bind('<ButtonRelease-1>', handle_select_release)
            return select_center

        def handle_iris_motion(event):
            center_x, center_y = select_circle_center.center
            radius = \
                int(np.sqrt(VectorUtil.sqr_euclidean_distance(select_circle_center.center, (event.x, event.y))))
            canvas.delete(handle_iris_motion.id)
            handle_iris_motion.id = canvas.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius, outline='#eef442')

        def handle_pupil_motion(event):
            center_x, center_y = select_circle_center.center
            radius = \
                int(np.sqrt(VectorUtil.sqr_euclidean_distance(select_circle_center.center, (event.x, event.y))))
            canvas.delete(handle_pupil_motion.id)
            handle_pupil_motion.id = canvas.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius, outline='#d8112f')

        def handle_iris_release(event):
            center_x, center_y = select_circle_center.center
            canvas.unbind("<B1-Motion>")
            canvas.unbind('<ButtonRelease-1>')
            radius = \
                int(np.sqrt(VectorUtil.sqr_euclidean_distance((center_y, center_x), (event.y, event.x))))
            handle_iris_release.initial_state = Provider.get_circle_coordinates(radius, (center_y, center_x))
            canvas.bind('<ButtonPress-1>', select_circle_center(handle_pupil_release, handle_pupil_motion))

        def handle_pupil_release(event):
            center_x, center_y = select_circle_center.center
            canvas.unbind("<B1-Motion>")
            canvas.unbind('<ButtonRelease-1>')
            radius = \
                int(np.sqrt(VectorUtil.sqr_euclidean_distance((center_y, center_x), (event.y, event.x))))
            pupil_initial_state = Provider.get_circle_coordinates(radius, (center_y, center_x))
            features, iris, pupil = FeaturesDetector.iris_detector(image, handle_iris_release.initial_state, pupil_initial_state)
            canvas.features = features
            transformed_img = self.draw_control_points(image, iris, pupil)
            self.create_new_image(transformed_img)

        handle_iris_motion.id = 0
        handle_iris_motion.radius = 0
        handle_pupil_motion.id = 0
        handle_pupil_motion.radius = 0
        select_circle_center.center = (0,0)
        select_circle_center.center = (0,0)
        handle_iris_release.initial_state = []
        canvas.bind('<ButtonPress-1>', select_circle_center(handle_iris_release, handle_iris_motion))

    def draw_control_points(self, image, iris, pupil):
        width, height = image.shape
        copy = np.copy(image)
        ans = np.zeros((image.shape[0], image.shape[1], 3))
        ans[:, :, 0] = copy
        ans[:, :, 1] = copy
        ans[:, :, 2] = copy
        for point in iris:
            x, y = point
            if 0 <= x < width and 0 <= y < height:
                ans[x][y][0] = 0
                ans[x][y][1] = 255
                ans[x][y][2] = 0
        for point in pupil:
            x, y = point
            if 0 <= x < width and 0 <= y < height:
                ans[x][y][0] = 0
                ans[x][y][1] = 0
                ans[x][y][2] = 255
        return ans

    def compare_iris(self):
        self.wait_variable(self.active_window)
        image, color, canvas = self.open_images[self.active_window.get()]
        image1Name = self.active_window.get().split()[0]
        self.wait_variable(self.active_window)
        image2, color2, canvas2 = self.open_images[self.active_window.get()]
        image2Name = self.active_window.get().split()[0]
        equals = LogGabor.compare_templates_w_euclidean(canvas.features, canvas2.features, image1Name, image2Name)

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
    app.master.geometry('700x650')
    app.mainloop()
