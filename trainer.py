import torch
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from vae import VAE

from tqdm import tqdm
import numpy as np

from loaddata import LoadData

class Trainer(): 
    def __init__(self, model, optimizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer

    def fit(self, load_data, epoch, beta_schedule = lambda i, total_i: 0.25):
        self.model.train()
        running_loss = [0.0, 0.0]

        total_iterations = int(len(load_data.train_data)/load_data.batch_size)
        dataloader = load_data.train_loader

        for i, data in tqdm(enumerate(dataloader), total=total_iterations):
            data, _ = data
            data = data.to(self.device)
            data = data.view(data.size(0), -1)
            self.optimizer.zero_grad()
            reconstruction, mu, logvar = self.model(data)
            beta = beta_schedule(total_iterations * epoch + i, total_iterations)
            recons_loss, kld_loss = self.model.loss_function(data, reconstruction, mu, logvar, beta)
            running_loss[0] += recons_loss.item()
            running_loss[1] += kld_loss.item()
            loss = recons_loss + beta * kld_loss
            loss.backward()
            self.optimizer.step()

            self.recons_loss_log.append(recons_loss/len(dataloader.dataset))
            self.kld_loss_log.append(kld_loss/len(dataloader.dataset))
            self.beta_log.append(beta)

        train_loss = list(map(lambda x: x/len(dataloader.dataset), running_loss))
        return train_loss

    def train(self, load_data, epochs, checkpoint_filepath, resume_training, beta_schedule=lambda epoch: 0.25):
        if resume_training:
            checkpoint = torch.load(checkpoint_filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_loss = checkpoint['loss']
            print(f"Loaded from checkpoint file {checkpoint_filepath}, Reconstruction Loss: {train_loss[-1][0]:.4f}, KL-Divergence loss: {train_loss[-1][1]:.4f}")
        else: 
            start_epoch = 0
            train_loss = []

        self.recons_loss_log = []
        self.kld_loss_log = []
        self.beta_log = []

        end_epoch = start_epoch + epochs

        for epoch in range(start_epoch, end_epoch):
            print(f"\nEpoch {epoch + 1}/{end_epoch}") 
            train_epoch_loss = self.fit(load_data, epoch, beta_schedule)
            print(f"\nReconstruction Loss: {train_epoch_loss[0]:.4f}, KL-Divergence loss: {train_epoch_loss[1]:.4f}")

            torch.save({
                'epoch' : epoch,
                'model_state_dict' : self.model.state_dict(),
                'optimizer_state_dict' : self.optimizer.state_dict(),
                'loss' : train_loss
                }, checkpoint_filepath)

            train_loss.append(train_epoch_loss)

        with open('trainlogs.npy', 'wb') as f:
            np.save(f, np.array(self.recons_loss_log))
            np.save(f, np.array(self.kld_loss_log))
            np.save(f, np.array(self.beta_log))

        return train_loss
