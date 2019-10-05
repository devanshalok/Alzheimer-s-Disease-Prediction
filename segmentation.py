import nibabel as nib
import scipy.misc
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#segmentation
kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]
scipy.misc.imsave('segmented.png',pic2show)
plt.imshow(pic2show)
