import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


N = 1000
X = np.random.random((N, 2)) * 6 - 3
Y = np.cos(2*X[:,0]) + np.cos(3*X[:,1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

model = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

n_epochs = 3000

recorded_losses = np.zeros(3000)

for i in range(n_epochs):
  epoch = i + 1
  optimizer.zero_grad()

  output = model(torch.from_numpy(X.astype(np.float32)))
  loss = loss_function(output, torch.from_numpy(Y.astype(np.float32)).reshape(-1, 1))

  loss.backward()
  optimizer.step()

  recorded_losses[i] = loss.item()

  if epoch % 500:
    print(f'Epoch: {i+1}, Loss: {loss.item()}')

plt.plot(recorded_losses)