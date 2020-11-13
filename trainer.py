
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

# cyclic annealing taken from https://arxiv.org/abs/1903.10145
def beta_anneal(iteration, total_iterations, start_beta, end_beta, burn_in, cycles_per_epoch):
    iterations_per_cycle = round(total_iterations/cycles_per_epoch)
    burn_in_period = iterations_per_cycle * burn_in
    beta = min(start_beta + ((iteration % iterations_per_cycle) / burn_in_period) * end_beta, end_beta)
    return beta

def fit(model, dataloader, epoch, beta_schedule = lambda i, total_i: 0.25):
    model.train()
    running_loss = (0.0, 0.0)
    total_iterations = int(len(train_data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=total_iterations):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        beta = beta_schedule(total_iterations * epoch + i, total_iterations)
        recons_loss, kld_loss = model.loss_function(data, reconstruction, mu, logvar, beta)
        loss = recons_loss + beta * kld_loss
        running_loss += (recons_loss.item(), kld_loss.item())
        loss.backward()
        optimizer.step()
    train_loss = [loss/len(dataloader.dataset) for loss in running_loss]
    return train_loss

#%%
def train(model, optimizer, train_loader, epochs, checkpoint_filepath, resume_training, beta_anneal_schedule=lambda epoch: 0.25):
    
    if resume_training:
        checkpoint = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_loss = checkpoint['loss']
        print(f"Loaded from checkpoint file {checkpoint_filepath}, train loss: {train_loss[-1]:.4f}")
    else: 
        start_epoch = 0
        train_loss = []

    end_epoch = start_epoch + epochs

    for epoch in range(start_epoch, end_epoch):
        print(f"\nEpoch {epoch + 1}/{end_epoch}") 
        train_epoch_loss = fit(model, train_loader, epoch, beta_schedule)
        print(f"\nReconstruction Loss: {train_epoch_loss[0]:.4f}")

        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : train_loss
            }, checkpoint_filepath)

        train_loss.append(train_epoch_loss)

beta_schedule = lambda iteration, total_iterations: beta_anneal(iteration, total_iterations, start_beta=0.0, end_beta=1.0, burn_in = 0.75, cycles_per_epoch = 1.5)
train(model, optimizer, train_loader, epochs, checkpoint_filepath, resume_training, beta_schedule)