import itertools
import os
import time

import torch
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from analyse import Analyser
from blocks.vade import VadeCNN
from dataloader import get_mnist

args = {
    "z_dim": 10,
    "batch_size": 1024,
    "debug": True,
    "analysis_after_epochs": 10,
    "train_epochs": 500,
    "pre_train_epochs": 50
}
################
print("Running with seed: " + str(torch.initial_seed()))

DL, DS = get_mnist(data_dir='../data', batch_size=args["batch_size"])
model = nn.DataParallel(VadeCNN(z_dim=args["z_dim"], h_dim=200, n_cls=10).cuda(), device_ids=range(1))
pre_train_optimizer = Adam(itertools.chain(model.module.encoder.parameters(), model.module.decoder.parameters()), lr=1e-3)
train_optimizer = Adam(model.parameters(), lr=2e-3)
lr_scheduler = StepLR(train_optimizer, step_size=10, gamma=0.95)
analyser = Analyser(model=model, DL=DL)


def pre_train(save_vars=False):
    # mse = nn.MSELoss()
    for x, y in DL:
        pre_train_optimizer.zero_grad()
        x = x.view((-1, 1, 28, 28)).cuda()
        mu, _ = model.module.encoder(x)
        x_hat = model.module.decoder(mu)
        L = 784 * F.binary_cross_entropy(x_hat, x, reduction='none').mean()
        # L = mse(x, x_hat)
        analyser.add_to_loss_variables({'L': L}, normalization_factor=len(DL))
        L.backward()
        pre_train_optimizer.step()

        if save_vars:
            analyser.add_to_intermediate_variables({"x": x, "y": y, 'mu': mu})


def train(save_vars=False):
    for x, y in DL:
        train_optimizer.zero_grad()
        x = x.view((-1, 1, 28, 28)).cuda()
        x_hat, mu, logvar, z = model(x)
        BCE, KLD, KLD_c, L_sparsity = model.module.losses(x, x_hat, mu, logvar, z)
        batch_loss = KLD + BCE + KLD_c + L_sparsity
        analyser.add_to_loss_variables({'BCE': BCE, 'KLD': KLD, 'KLD_c': KLD_c, 'L_sp': L_sparsity, 'L': batch_loss}, normalization_factor=len(DL))
        batch_loss.backward()
        train_optimizer.step()

        if save_vars:
            analyser.add_to_intermediate_variables({"x": x, "y": y, 'mu': mu})

    y_pred = model.cpu().float().module.predict(DS["X"].view((-1, 1, 28, 28)))
    analyser.add_to_loss_variables({"ACC": Analyser.cluster_acc(y_pred, DS["Y"].numpy())[0] * 100})
    model.cuda()


if not os.path.exists('./pretrain_model.pk'):
    for epoch in range(args["pre_train_epochs"]):

        start_time = time.time()
        analyser.flush_epoch_variables()
        pre_train(save_vars=(epoch % args["analysis_after_epochs"] == 0))

        if epoch % args["analysis_after_epochs"] == 0:
            # analyser.write_state(epoch, weights=False)
            Mu, _ = model.cpu().float().module.encoder(DS["X"].view((-1, 1, 28, 28)))
            gmm = GaussianMixture(n_components=10, covariance_type='diag', n_init=5)
            pre = gmm.fit_predict(Mu.detach().cpu().numpy())
            analyser.add_to_loss_variables({"ACC": Analyser.cluster_acc(pre, DS["Y"].numpy())[0] * 100})
            model.cuda()

        end_time = time.time()
        print(analyser.update_str(epoch=epoch, epoch_time=(end_time - start_time)))

    print('Acc={:.4f}%'.format(Analyser.cluster_acc(pre, DS["Y"].numpy())[0] * 100))

    model.module.initialize_gmm_params(gmm)
    torch.save(model.state_dict(), './pretrain_model.pk')
else:
    model.load_state_dict(torch.load('./pretrain_model.pk'))


for epoch in range(args["train_epochs"]):
    start_time = time.time()
    analyser.flush_epoch_variables()
    analyser.add_to_loss_variables({"LR": lr_scheduler.get_lr()[0]})
    train(save_vars=(epoch % args["analysis_after_epochs"] == 0))
    lr_scheduler.step(epoch)

    if epoch % args["analysis_after_epochs"] == 0:
        analyser.write_state(epoch, weights=False)
    end_time = time.time()
    print(analyser.update_str(epoch=epoch, epoch_time=(end_time - start_time)))
