import torch
import os
import sys
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision
import torchvision.transforms as transforms
import cv2
from sklearn.metrics import confusion_matrix
import itertools

train_dataset = torchvision.datasets.FashionMNIST(
    root='./datasets',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./datasets',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

