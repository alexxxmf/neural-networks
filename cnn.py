import skimage.data
from skimage.viewer import ImageViewer
import numpy as np
import matplotlib.pyplot as plt

img = skimage.data.chelsea()
img = skimage.color.rgb2gray(img)

plt.imshow(img, cmap='gray')