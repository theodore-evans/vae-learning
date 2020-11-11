
#!/usr/bin/env python3

#%%
import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from vae import vae

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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
model = vae(input_dim = 784, reconstruction_loss_f=criterion).to(device) 
optimizer = optim.Adam(model.parameters(), lr=lr)

def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        loss = model.loss_function(data, reconstruction, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss

#%%
epochs = 10
checkpoint_filepath = "weights.pt"

def train(model, optimizer, train_loader, epochs, checkpoint_filepath):
    train_loss = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_epoch_loss = fit(model, train_loader)
        print(f"\nTrain Loss: {train_epoch_loss:.4f}")

        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : train_epoch_loss
            }, checkpoint_filepath)

        train_loss.append(train_epoch_loss)

train(model, optimizer, train_loader, epochs, checkpoint_filepath)
# %%
