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

epochs = 300
current_step = 0
print_every = 100
mean_generator_loss = 0
mean_discriminator_loss = 0

z_dim = 64
learning_rate = 0.00001
loss_func = nn.BCEWithLogitsLoss()

batch_size = 128
device = 'cuda'

dataloader = DataLoader(MNIST('.', download=True, transform=transforms.ToTensor()), shuffle=True, batch_size=batch_size)

# no steps 60000 images / 128 ~ 468

def genBlock(input, output):
  return nn.Sequential(
    nn.Linear(input, output),
    nn.BatchNorm1d(output),
    nn.ReLU(inplace=True)
  )

class Generator(nn.Module):

  def __init__(self, z_dim=64, input_dim=784, hidden_dim=128):
    super().__init__()
    self.gen = nn.Sequential(
        genBlock(z_dim, hidden_dim), # in 64, out 128
        genBlock(hidden_dim, hidden_dim*2), # in 128, out 256
        genBlock(hidden_dim*2, hidden_dim*4), # in 256, out 512
        genBlock(hidden_dim*4, hidden_dim*8), # in 512, out 1024
        nn.Linear(hidden_dim*8, input_dim), # in 1024, out 784 (28x28)
        nn.Sigmoid()
    )

  def forward(self, noise):
    return self.gen(noise)

def gen_noise(number, z_dim):
  return torch.randn(number, z_dim).to(device)