import torch, pdb
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def show(tensor, channels=1, size=(28,28), num=16):
  data = tensor.detach().cpu().view(-1, channels, *size)
  grid = make_grid(data[:num], n_rows=4).permute(1,2,0)
  plt.imshow(grid)
  plt.show()