import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from src.input.util import Util
from src.input.provider import Provider

def normalization(image, inner, outter):
    iris=np.zeros((36,360,3))
    innerimax = max(inner[:][:])
    innerimin = min(inner[:][:])
    center = (int((innerimax[1]+innerimin[1])/2), int((innerimax[0]+innerimin[0])/2))
    innerradius = int(math.sqrt((center[0]-inner[0][1])**2+(center[1]-inner[0][0])**2))
    outterradius = int(math.sqrt((center[0]-outter[0][1])**2+(center[1]-outter[0][0])**2))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
           for k in range(image.shape[2]):
             pointradius = int(math.sqrt((center[0]-i)**2+(center[1]-j)**2))
             if (pointradius > innerradius) and (pointradius < outterradius):
               theta = int(math.degrees(math.atan2(j - center[1], i - center[0])))
               if theta < 0:
                 theta += 360
               iris[pointradius-innerradius][theta][k]=image[i][j][k]
    return iris

def interest_degrees(image):
    ans = np.zeros((110,34,3)) #Resize al tamaño de interes donde hay menos ruido
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
           for k in range(image.shape[2]):
              if i > 180 and i < 246 or i > 314 and i < 359:
                ans[i][j]=image[i][j]
    return ans

def build_filters():
    filters = []
    ksize = 32
    for theta in np.arange(0, np.pi, np.pi / 8):
        print(math.degrees(theta))
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    template = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if fimg[i][j] == 1:
                    template[i][j] = 1
    return template

def ceropercent(template):
    acumm=0
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            if template[i][j] == 1:
                acumm += 1
    print(acumm/(template.shape[0]*template.shape[1]))

def compare_templates(basetemplate, template):
    acumm=0
    total=basetemplate.shape[0]*basetemplate.shape[1]
    for i in range(basetemplate.shape[0]):
        for j in range(basetemplate.shape[1]):
           total += 1
           if basetemplate[i][j] != template [i][j]:
               acumm += 1
    hammingdist = acumm / total
    return hammingdist

