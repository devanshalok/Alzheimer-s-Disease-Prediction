import nibabel as nib
import scipy.misc
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#edge detection
edges = cv.Canny(img_weighted,100,200)
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
scipy.misc.imsave('edged.png',edges)


pic = plt.imread('edged.png')/255  # dividing by 255 to bring the pixel values between 0 and 1
print(pic.shape)
plt.imshow(pic)
pic_n = pic.reshape(pic.shape[0]*pic.shape[1],)
scipy.misc.imsave('sharpened2.png',pic)
