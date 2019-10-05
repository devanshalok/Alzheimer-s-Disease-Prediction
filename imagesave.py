import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.misc
import glob
from sklearn.cluster import KMeans
import cv2 as cv

#Save the 92 index slices out of 256 2D slices of the 3D MRI image
basepath = 'C:/Users/aasth/Desktop/adni dataset'
outpath = 'C:/Users/aasth/Desktop/adni dataset/final'
os.chdir(basepath)

i=1
for entry in glob.glob('*.hdr'):
    image_array= nib.load(entry).get_data()
    data=np.rot90(image_array[:, 92,:,0])
    image_name='adni'+ str(i) +'.png'
    final = os.path.join(outpath, image_name)
    scipy.misc.imsave(final, data)
    i =i+1

