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
  grid = make_grid(data[:num], nrow=4).permute(1,2,0)
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

noise_image_example = torch.randn(256, 256)
plt.imshow(noise_image_example, cmap='gray')

def generator_block(input, output):
  return nn.Sequential(
    nn.Linear(input, output),
    nn.BatchNorm1d(output),
    nn.ReLU(inplace=True)
  )

class Generator(nn.Module):

  def __init__(self, z_dim=64, input_dim=784, hidden_dim=128):
    super().__init__()
    self.gen = nn.Sequential(
        generator_block(z_dim, hidden_dim), # in 64, out 128
        generator_block(hidden_dim, hidden_dim*2), # in 128, out 256
        generator_block(hidden_dim*2, hidden_dim*4), # in 256, out 512
        generator_block(hidden_dim*4, hidden_dim*8), # in 512, out 1024
        nn.Linear(hidden_dim*8, input_dim), # in 1024, out 784 (28x28)
        nn.Sigmoid()
    )

  def forward(self, noise):
    return self.gen(noise)

def gen_noise(number, z_dim):
  return torch.randn(number, z_dim).to(device)


def discriminator_block(input, output):
  return nn.Sequential(
    nn.Linear(input, output),
    nn.LeakyReLU(0.2)
  )

class Discriminator(nn.Module):
  def __init__(self, input_dim=784, hidden_dim=256):
    super().__init__()
    self.discriminator = nn.Sequential(
      discriminator_block(input_dim, hidden_dim * 4), # in 784, out 1024
      discriminator_block(hidden_dim * 4, hidden_dim * 2), # in 1024, out 512
      discriminator_block(hidden_dim * 2, hidden_dim), # in 512, out 256
      nn.Linear(hidden_dim, 1) # in 256, out 1
    )

  def forward(self, image):
    return  self.discriminator(image)


generator = Generator(z_dim).to(device)
generator_opt = torch.optim.Adam(generator.parameters(), lr=learning_rate)
discriminator = Discriminator(z_dim).to(device)
discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)


x,y = next(iter(dataloader))
print(x.shape, y.shape)
print(y[:10])

noise = gen_noise(batch_size, z_dim)
fake = generator(noise)
show(fake)


def calc_generator_loss(loss_func, generator, discriminator, number, z_dim):
  noise = gen_noise(number, z_dim)
  fake = generator(noise)
  predictions = discriminator(fake)
  targets = torch.ones_like(predictions)
  generator_loss = loss_func(predictions, targets)

  return generator_loss

def calc_discriminator_loss(loss_func, generator, discriminator, number, real, z_dim):
  noise = gen_noise(number, z_dim)
  fake = generator(noise)
  discriminator_fake = discriminator(fake.detach())
  discriminator_fake_targets = torch.zeros_like(discriminator_fake)
  discriminator_fake_loss = loss_func(discriminator_fake, discriminator_fake_targets)

  discriminator_real = discriminator(real)
  discriminator_real_targets = torch.ones_like(discriminator_real)
  discriminator_real_loss = loss_func(discriminator_real, discriminator_real_targets)

  discriminator_loss = (discriminator_fake_loss + discriminator_real_loss)/2

  return discriminator_loss

# 469 steps to process 128 images (except the last one)

for epoch in range(epochs):
  # we don t need the labels hence the underscoe _
  for real, _ in tqdm(dataloader):
    # discriminator
    discriminator_opt.zero_grad()

    current_batch_size = len(real) # 128 x 1 x 28 x 28
    real = real.view(current_batch_size, -1) # 128 x 784
    real = real.to(device)

    discriminator_loss = calc_discriminator_loss(
        loss_func, generator, discriminator, current_batch_size, real, z_dim)

    discriminator_loss.backward(retain_graph=True)
    discriminator_opt.step()

      # generator
    generator_opt.zero_grad()
    generator_loss = calc_generator_loss(loss_func, generator, discriminator, current_batch_size, z_dim)
    generator_loss.backward(retain_graph=True)
    generator_opt.step()

    mean_discriminator_loss += discriminator_loss.item()/print_every
    mean_generator_loss += discriminator_loss.item()/print_every

    if (current_step % print_every == 0 and current_step > 0):
      fake_noise = gen_noise(current_batch_size, z_dim)
      fake = generator(fake_noise)
      show(fake)
      show(real)
      print(f'{epoch}: step {current_step} / Generator loss: {mean_generator_loss} / discriminator loss: {mean_discriminator_loss}')

    current_step+=1