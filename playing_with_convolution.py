import skimage.data
import numpy as np
import matplotlib.pyplot as plt
import scipy

img = skimage.data.chelsea()
img = skimage.color.rgb2gray(img)

plt.imshow(img, cmap='gray')

sobel_filter_v = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_filter_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# Result after applying vertical edge detection
r_fv = scipy.ndimage.convolve(img, sobel_filter_v)
plt.imshow(r_fv, cmap='gray')
# Result after applying horizontal edge detection
r_fh = scipy.ndimage.convolve(img, sobel_filter_h)
plt.imshow(r_fh, cmap='gray')

# Here we combine bode edge detection results
r_comb = np.sqrt(np.square(r_fv) + np.square(r_fh))
plt.imshow(r_comb, cmap='gray')