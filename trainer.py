
#!/usr/bin/env python3
import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from vae import VAE

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

epochs = 1
batch_size = 64
lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
transform = transforms.Compose([
    transforms.ToTensor(),
])

# train and validation data
train_data = datasets.MNIST(
    root='../input/data',
    train=True,
    download=True,
    transform=transform
)
val_data = datasets.MNIST(
    root='../input/data',
    train=False,
    download=True,
    transform=transform
)

# %%
# training and validation data loaders
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False
)

#%%
criterion = nn.BCELoss(reduction='sum')
model = VAE(input_dim = 784, reconstruction_loss_f=criterion).to(device) 
optimizer = optim.Adam(model.parameters(), lr=lr)

# def final_loss(bce_loss, mu, logvar):
#     """
#     This function will add the reconstruction loss (BCELoss) and the 
#     KL-Divergence.
#     KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     :param bce_loss: recontruction loss
#     :param mu: the mean from the latent vector
#     :param logvar: log variance from the latent vector
#     """
#     BCE = bce_loss 
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD

def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        # bce_loss = criterion(reconstruction, data)
        # loss = final_loss(bce_loss, mu, logvar)
        loss = model.loss_function(data, reconstruction, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss

#%%
train_loss = []
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_epoch_loss = fit(model, train_loader)
    train_loss.append(train_epoch_loss)
    print(f"\nTrain Loss: {train_epoch_loss:.4f}")

# #%%
# import matplotlib.pyplot as plt
# %matplotlib inline

# sample, _ = next(iter(val_loader))
# reconstruction, _ , _ = model(sample)

# def image_from_tensor(tensor, index):
#     return tensor[index].detach().numpy().reshape(28, -1)

# sample_index = 3
# image = image_from_tensor(sample, sample_index)
# reconstructed_image = image_from_tensor(reconstruction, sample_index)
# fig = plt.figure(figsize = (8,8))
# fig.add_subplot(2,1,1)
# plt.imshow(image)
# fig.add_subplot(2,1,2)
# plt.imshow(reconstructed_image)

# plt.show()

# %%
