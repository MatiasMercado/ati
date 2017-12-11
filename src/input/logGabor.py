import numpy as np
import argparse
import cv2
import math
from LogGabor import LogGabor
from LogGabor import imread
import matplotlib as plt

def normalization(image, inner, outter):
    iris=np.zeros((360,64))

    innerimax = max(inner[:][:])
    innerimin = min(inner[:][:])
    center = (int((innerimax[0]-innerimin[0])/2), int(innerimin[1]+(innerimax[1]-innerimin[1])/2))
    print(center)

    innerradius = max(inner[:][:])[0] - center[0]
    outterradius = max(outter[:][:])[0] - center[0]
    print (innerradius)
    print (outterradius)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pointradius = math.sqrt((center[0]-i)**2+(center[1]-j)**2)
            if pointradius > innerradius and pointradius < outterradius:
                theta = int(math.degrees(math.atan2(j - center[0], i - center[1])))
                iris[theta][pointradius]=image[i][j]
    return iris

def interest_degrees(image):
    ans = np.zeros(110,64) #Resize al tamaño de interes donde hay menos ruido
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i > 180 and i < 246 or i > 314 and i < 359:
                ans[i][j]=image[i][j]
    return ans

def build_filters(): #Gabor NO LogGabor
    filters = []
    ksize = 32
    for theta in np.arange(0, np.pi, np.pi / 8):
        print(math.degrees(theta))
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    # kern 0°
    fimg0 = cv2.filter2D(img, cv2.CV_8UC3, filters[0])
    # kern 22.5°
    fimg1 = cv2.filter2D(img, cv2.CV_8UC3, filters[1])
    # kern 45.0°
    fimg2 = cv2.filter2D(img, cv2.CV_8UC3, filters[2])
    # kern 67.5°
    fimg3 = cv2.filter2D(img, cv2.CV_8UC3, filters[3])
    # kern 90.0°
    fimg4 = cv2.filter2D(img, cv2.CV_8UC3, filters[4])
    # kern 112.5°
    fimg5 = cv2.filter2D(img, cv2.CV_8UC3, filters[5])
    # kern 135.0°
    fimg6 = cv2.filter2D(img, cv2.CV_8UC3, filters[6])
    # kern 157.5°
    fimg7 = cv2.filter2D(img, cv2.CV_8UC3, filters[7])
    print(fimg0.shape, fimg1.shape, fimg2.shape, fimg3.shape, fimg4.shape, fimg5.shape, fimg6.shape, fimg7.shape)
    print(img.shape)
    return fimg0

def create_template(image): #Aun no obtiene el angulo del Feature Extraction
    parameterfile = 'C:/Users/Josimar/PycharmProjects/ati-master/src/input/default_param.py'
    lg = LogGabor(parameterfile)
    lg.loggabor_image()
    orientation = np.zeros(image.shape)
    template = np.zeros(image.shape)
    print(image.shape)
    print(lg.frequency_angle())
    print(lg.frequency_radius().shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if orientation[i][j] > 0 and orientation[i][j] < 90:
                template[i][j]= '11'
            if orientation[i][j] > 90 and orientation[i][j] < 180:
                template[i][j]= '01'
            if orientation[i][j] > 180 and orientation[i][j] < 270:
                template[i][j]= '00'
            if orientation[i][j] > 270 and orientation[i][j] < 360:
                template[i][j]= '10'
    return lg.frequency_angle()

def compare_templates(basetemplate, template):
    acumm=0
    total=basetemplate.shape[0]*basetemplate.shape[1]
    for i in range(basetemplate.shape[0]):
        for j in range(basetemplate.shape[1]):
           total += 1
           if basetemplate[i][j] != template [i][j]:
               acumm += 1
    hammingdist = acumm * 1 / total
    return hammingdist

image=imread('lena.jpg')

