import pickle

import numpy as np
import torch
from torch import nn, distributions
from torch.nn import functional as F

from blocks.decoders import ConvDecoder, FcDecoder
from blocks.encoders import ConvEncoder, FcEncoder
from blocks.layers import Lambda


########################################################
class VadeCNN(nn.Module):
    def __init__(self, h_dim=64, z_dim=16, n_cls=10):
        super(VadeCNN, self).__init__()

        # self.encoder = ConvEncoder(z_dim=z_dim, h_dim=h_dim)
        # self.decoder = ConvDecoder(z_dim=z_dim)
        self.encoder = FcEncoder(z_dim=z_dim)
        self.decoder = FcDecoder(z_dim=z_dim)

        self.reparam = Lambda(self.reparameterize)

        self.pi = nn.Parameter(torch.FloatTensor(n_cls, ).fill_(1) / n_cls, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(n_cls, z_dim).fill_(0), requires_grad=True)
        self.logvar_c = nn.Parameter(torch.FloatTensor(n_cls, z_dim).fill_(0), requires_grad=True)

    def initialize_gmm_params(self, gmm):
        pi = torch.from_numpy(gmm.weights_).cuda().float()
        self.pi.data = torch.log(pi / (1 - pi))
        self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
        self.logvar_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())

    def reparameterize(self, arguments):
        mu, logvar = arguments
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)

        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z
        
    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        return z, mu, logvar

    def decode(self, x):
        x_hat = self.decoder(z)
        return x_hat

    def pc_given_z(self, z):
        std_c = torch.exp(self.logvar_c / 2)
        pi = distributions.Categorical(torch.sigmoid(self.pi)).probs
        log_pz_given_c = distributions.Normal(self.mu_c, std_c[None, :, :]).log_prob(z[:, None, :]).sum(dim=2)
        log_pz_and_c = torch.log(pi)[None, :] + log_pz_given_c
        pc_given_z = torch.exp(log_pz_and_c - torch.logsumexp(log_pz_and_c, dim=1)[:, None])
        return pc_given_z

    def predict(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        pc_given_z_np = self.pc_given_z(z).detach().cpu().numpy()
        return np.argmax(pc_given_z_np, axis=1)

    def losses(self, x, x_hat, mu_z, logvar_z, z):
        std_z = torch.exp(logvar_z / 2)
        std_c = torch.exp(self.logvar_c / 2)
        pi = distributions.Categorical(torch.sigmoid(self.pi)).probs
        pc_given_z = self.pc_given_z(z)

        BCE = F.binary_cross_entropy(x_hat, x, reduction='mean') * 784
        KLD = torch.sum(pc_given_z * distributions.kl_divergence(
            distributions.Independent(distributions.Normal(mu_z[:, None, :], std_z[:, None, :]), reinterpreted_batch_ndims=1),
            distributions.Independent(distributions.Normal(self.mu_c[None, :, :], std_c[None, :, :]), reinterpreted_batch_ndims=1)
        ), dim=1).mean()
        KLD_c = distributions.kl_divergence(distributions.Categorical(pc_given_z), distributions.Categorical(pi[None, :])).mean()

        return BCE, KLD, KLD_c, torch.tensor(0).float()
