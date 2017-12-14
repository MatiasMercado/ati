import numpy as np
import cv2
import math
from LogGabor import imread

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

image=imread('lena.jpg')
ceropercent(process(image,build_filters()))


