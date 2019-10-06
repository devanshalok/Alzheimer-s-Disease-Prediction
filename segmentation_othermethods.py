#SEGMENTATION

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import cv2 as cv
from sklearn.cluster import KMeans
import imageio

#import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
#import skimage.draw as draw
#import skimage.color as color

os.chdir('C:/Users/aasth/Desktop')

text = cv.imread('sharp.png')
#Convert into a 2D image
text = text[:, :, 0]
print(text.shape)

#Supervised Thresholding: histogram plot analysis to set threshold 
fig, ax = plt.subplots(1, 1)
ax.hist(text.ravel(), bins=32, range=[0, 256])
ax.set_xlim(0, 256)
text_threshold = 10

#Unsupervised Thresholding:
text_threshold = filters.threshold_li(text)

#Method 1:
text_segmented = text < text_threshold
plt.imshow(text_segmented)
imageio.imwrite('segmented.png', text_segmented)

#Method 2: Watershed Algorithm
text = cv.imread('sharp.png')
#Convert into a 2D image
text = text[:, :, 0]

markers = np.zeros_like(text)
markers[text < 30] = 1
markers[text > 150] = 2

from skimage.filters import sobel
elevation_map = sobel(text)

from skimage.morphology import watershed
segmentation = watershed(elevation_map, markers)
scipy.misc.imsave('seg1.png',segmentation)

from scipy import ndimage as ndi
segmentation = ndi.binary_fill_holes(segmentation - 1)
lc, _ = ndi.label(segmentation)
scipy.misc.imsave('seg2.png', lc)

plt.imshow(lc)
cv.waitKey(1)

'''
#Method 3: Suprervised Segmentation
#Active Contour

#Unsupervised Segmentation thorugh SLIC( Simple Linear Iterative Clustering, it uses KMeans:
image_slic = seg.slic(text,n_segments=155)
plt.imshow(image_slic)
'''

text = cv.imread('sharp.png')
text = text[:, :, 0]
e = text/255
arr = e.reshape(e.shape[0]*e.shape[1])
#print(arr.mean())
for i in range(arr.shape[0]):
    if arr[i] > arr.mean():
        arr[i] = 2
    elif arr[i] > 0.13:
        arr[i] = 1
    else:
        arr[i] = 0
e = arr.reshape(e.shape[0],e.shape[1])
plt.imshow(e)
e = e.astype(np.uint8)
imageio.imwrite('segmented123.png', e)


#K-means
text = cv.imread('sharp.png')
#text = text[:, :, 0]
pic_n = text.reshape(text.shape[0]*text.shape[1], text.shape[2])
pic_n = np.reshape(text, (-1, 1))
pic_n.shape
kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = pic2show.reshape(text.shape[0], text.shape[1], text.shape[2])
plt.imshow(cluster_pic)
