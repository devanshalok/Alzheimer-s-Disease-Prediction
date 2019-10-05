import nibabel as nib
import scipy.misc
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#sharpening
img_weighted = cv.addWeighted(img, 1.0 + 3.0, img_median, -3.0, 0) # im1 = im + 3.0*(im - im_blurred)
plt.figure(figsize=(20,10))
scipy.misc.imsave('sharp.png',img_weighted)
plt.subplot(122),plt.imshow(cv.cvtColor(img_weighted, cv.COLOR_BGR2RGB)), plt.axis('off'), plt.title('', size=10)
plt.show()
