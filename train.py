
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
from torchvision import datasets
from torch.utils.data import DataLoader

from vae import VAE
from loaddata import LoadData
from trainer import Trainer

from tqdm import tqdm

from torchvision.utils import save_image

#%%
parser = argparse.ArgumentParser(description='Train a VAE')
parser.add_argument("-r", action='store_true', help="When true, resume training from checkpoint file")
parser.add_argument("-b", "--batchsize", default="64", help="Batch size")
parser.add_argument("-l", "--lr", default=1e-4, help="Learning rate")
parser.add_argument("-e", "--epochs", default="10", help="Number of epochs to train")
parser.add_argument("-c", "--checkpoint", default="weights.pt", help="Checkpoint file")
parser.add_argument("--beta-start", default = "0.0")
parser.add_argument("--beta-end", default = "1.0")
parser.add_argument("--beta-cycles", default = "1.5")
parser.add_argument("--beta-burnin", default = "0.75")

options = parser.parse_args()

resume_training = options.r
batch_size = int(options.batchsize)
lr = float(options.lr)
epochs = int(options.epochs)
checkpoint_filepath = options.checkpoint

start_beta = float(options.beta_start)
end_beta = float(options.beta_end)
cycles_per_epoch = float(options.beta_cycles)
burn_in = float(options.beta_burnin)

#%%
data = LoadData(datasets.MNIST, batch_size = batch_size)
criterion = nn.BCELoss(reduction='sum')
model = VAE(input_dim = 784, reconstruction_loss_f=criterion)

# cyclic annealing taken from https://arxiv.org/abs/1903.10145
def beta_anneal(iteration, total_iterations, start_beta, end_beta, cycles_per_epoch, burn_in):
    iterations_per_cycle = round(total_iterations/cycles_per_epoch)
    burn_in_period = iterations_per_cycle * burn_in
    beta = min(start_beta + ((iteration % iterations_per_cycle) / burn_in_period) * end_beta, end_beta)
    return beta

beta_schedule = lambda iteration, total_iterations: beta_anneal(iteration, total_iterations, start_beta, end_beta, cycles_per_epoch, burn_in)

print(options)
trainer = Trainer(model, optim.Adam(model.parameters(), lr=lr))
trainer.train(data, epochs, checkpoint_filepath, resume_training, beta_schedule)
# %%
