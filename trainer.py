
#!/usr/bin/env python3

#%%
import argparse
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

#%%

import sys
sys.argv=['']
del sys

parser = argparse.ArgumentParser(description='Train a VAE')
parser.add_argument("-r", action='store_true', help="When true, resume training from checkpoint file")
parser.add_argument("-b", "--batchsize", default="64", help="Batch size")
parser.add_argument("-l", "--lr", default=1e-4, help="Learning rate")
parser.add_argument("-e", "--epochs", default="10", help="Number of epochs to train")
parser.add_argument("-c", "--checkpoint", default="weights.pt", help="Checkpoint file")

options = parser.parse_args()

resume_training = options.r
batch_size = int(options.batchsize)
lr = float(options.lr)
epochs = int(options.epochs)
checkpoint_filepath = options.checkpoint

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

if resume_training:
    checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print("Loaded from checkpoint file {checkpoint_filepath}, current loss: {loss}")

train(model, optimizer, train_loader, epochs, checkpoint_filepath)

# %%
