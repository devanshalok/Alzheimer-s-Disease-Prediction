import nibabel as nib
import scipy.misc
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#noise reduction
img = cv.imread('hello2.png') 
img_median = cv.medianBlur(img, 5) 
scipy.misc.imsave('noise.png',img_median)
plt.imshow(img_median)
