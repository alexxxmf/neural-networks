import os
import numpy as np
from posix import listdir
import urllib
import urllib.request
from zipfile import ZipFile
import cv2
import matplotlib.pyplot as plt
from utils.data_fns import create_data_mnist

np.set_printoptions(linewidth=200)

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

if not os.path.isfile(FILE):
  print(f'Downloading {URL} and saving as {FILE}...')
  urllib.request.urlretrieve(URL, FILE)

if not os.path.isdir(FOLDER):
  print('Unzipping images')
  with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)

  print('Done!')

X, y, X_test, y_test = create_data_mnist(FOLDER)

X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

print(X.shape)