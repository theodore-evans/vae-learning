import torch
from torch import nn
from torch.nn import functional as F

class vae(nn.Module):
    def __init__(self, input_dim = 1024, z_dim = 16, hidden_dims=[256, 128], beta = 0.25, reconstruction_loss_f = F.mse_loss):
        super(vae, self).__init__()
        modules = []

        self.reconstruction_loss_f = reconstruction_loss_f

        self.data_dim = input_dim
        self.beta = beta
        self.z_dim = z_dim

        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_dim, h_dim))
            input_dim = h_dim
        
        modules.append(nn.Linear(hidden_dims[-1], 2 * z_dim))
        modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*modules)

        modules = []

        input_dim = z_dim
        for h_dim in hidden_dims[::-1]:
            modules.append(nn.Linear(input_dim, h_dim))
            input_dim = h_dim

        modules.append(nn.Linear(hidden_dims[0], self.data_dim))

        modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*modules)

    def encode(self, input):
        input = input.view(-1,self.data_dim)
        x = self.encoder(input).view(-1,2,self.z_dim)
        mu = x[:, 0, :]
        log_var = x[:, 1,:]
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def decode(self, z):
        x = self.decoder(z)
        return x
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)

        return x_hat, mu, log_var

    def loss_function(self, x, x_hat, mu, log_var):
        recons_loss = self.reconstruction_loss_f(x_hat, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim=0)

        return recons_loss + self.beta * kld_loss
