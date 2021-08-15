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

def conv_(img, conv_filter):
  filter_size = conv_filter.shape[1]
  result = np.zeros((img.shape))

  #Looping through the image to apply the convolution operation.
  for r in np.uint16(np.arange(filter_size/2.0, img.shape[0]-filter_size/2.0+1)):
    for c in np.uint16(np.arange(filter_size/2.0, img.shape[1]-filter_size/2.0+1)):
      """
      Getting the current region to get multiplied with the filter.
      How to loop through the image and get the region based on 
      the image and filer sizes is the most tricky part of convolution.
      """
      curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)), 
                  c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]
      
      #Element-wise multipliplication between the current region and the filter.
      curr_result = curr_region * conv_filter
      conv_sum = np.sum(curr_result) #Summing the result of multiplication.
      result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.

  #Clipping the outliers of the result matrix.
  final_result = result[np.uint16(filter_size/2.0):result.shape[0]-np.uint16(filter_size/2.0), 
                        np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]
  return final_result

# conv_filter (number of filters, dimension 1, dimension 2, more?)
def convolve(img, conv_filter):

  if len(img.shape) > 2 or len(conv_filter.shape) > 3:
  
    if img.shape[-1] != conv_filter.shape[-1]:
      print("Error: Number of channels in both image and filter must match.")
      sys.exit()
    
  # Check if filter is a square
  if conv_filter.shape[1] != conv_filter.shape[2]:
    print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
    sys.exit()
  
  # Check if filter diemnsions are odd
  # Here there is a great explanation by the user Dynamic Stardust on the odd sizing of filters
  # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
  if conv_filter.shape[1] % 2 == 0: 
    print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
    sys.exit()
  
  # An empty feature map to hold the output of convolving the filters with the image.
  # conv_filter is squared hence the triviality between choosing between conv_filter.shape[1 or 2]
  feature_maps = np.zeros((img.shape[0] - conv_filter.shape[1] + 1, 
                              img.shape[1] - conv_filter.shape[1] + 1, 
                              conv_filter.shape[0]))

  for filter_num in range(conv_filter.shape[0]):
    print("Filter ", filter_num + 1)
    # getting a filter from the bank assuming conv_filters (num of filters, d1, d2...)
    curr_filter = conv_filter[filter_num, :]
    """ 
    Checking if there are multiple channels for the single filter.
    If so, then each channel will convolve the image.
    The result of all convolutions are summed to return a single feature map.
    """
    if len(curr_filter.shape) > 2:
      # Array holding the sum of all feature maps
      conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])
      # Convolving each channel with the image and summing the results.
      for ch_num in range(1, curr_filter.shape[-1]):
        conv_map = conv_map + conv_(img[:, :, ch_num], 
                          curr_filter[:, :, ch_num])
    # Just a single channel in the filter.
    else:
      conv_map = conv_(img, curr_filter)
      
    feature_maps[:, :, filter_num] = conv_map
  
  return feature_maps

feature_maps = convolve(img, l1_filter)

# Printing both feature maps just to take a look
plt.imshow(feature_maps[:,:,0], cmap='gray')
plt.imshow(feature_maps[:,:,1], cmap='gray')

def relu_forward(f_maps):
  relu_out = np.zeros(f_maps.shape)

  for i in range(f_maps.shape[-1]):
    relu_out[:,:,i] = np.maximum(0, f_maps[:,:,i])

  return relu_out

feature_maps_relu = relu_forward(feature_maps)

