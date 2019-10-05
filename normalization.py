import nibabel as nib
import scipy.misc
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#normalization
normalizedImg = np.zeros((800, 800))
normalizedImg = cv.normalize(img,  normalizedImg, 0, 255, cv.NORM_MINMAX)
scipy.misc.imsave('normalized.png',normalizedImg)
plt.imshow(normalizedImg)
