import skimage.data
from skimage.viewer import ImageViewer
import numpy as np
import matplotlib.pyplot as plt
import sys

img = skimage.data.chelsea()
img = skimage.color.rgb2gray(img)

plt.imshow(img, cmap='gray')

# 2 filters of 3x3. Image is gray, so there is no depth
# image can be represented as a 2D array/tensor (matrix)
l1_filter = np.zeros((2,3,3))

print(l1_filter.shape)
print(img.shape)

# let's play with the filters for a bit
l1_filter[0, :, :] = np.array([[[-1, 0, 1], 
                                   [-1, 0, 1], 
                                   [-1, 0, 1]]])
l1_filter[1, :, :] = np.array([[[1,   1,  1], 
                                   [0,   0,  0], 
                                   [-1, -1, -1]]])
# conv_filter (number of filters, dimension 1, dimension 2, more?)
def convolve(img, conv_filter):

  if len(img.shape) > 2 or len(conv_filter.shape) > 3:
  
    if img.shape[-1] != conv_filter.shape[-1]:
      print("Error: Number of channels in both image and filter must match.")
      sys.exit()
    
  # Check if filter dimensions are equal
  if conv_filter.shape[1] != conv_filter.shape[2]:
    print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
    sys.exit()
