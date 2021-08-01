import os
import numpy as np
from posix import listdir
import urllib
import urllib.request
from zipfile import ZipFile
import cv2
import matplotlib.pyplot as plt

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

image_data = cv2.imread('fashion_mnist_images/train/7/0002.png', cv2.IMREAD_UNCHANGED)

labels = os.listdir(f'{FOLDER}/train')

print(image_data)

X = []
y = []


