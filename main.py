import itertools
import os
import time
import datetime

import torch

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.vade_archs import Vade2D
from dataset import return_data
from miscellaneous.args_parsing import args_parsing

################


def pre_train(data_loader,save_vars=False):
    # mse = nn.MSELoss()
    for x, y in data_loader:
        pre_train_optimizer.zero_grad()
        x = x.view((-1, 1, dim, dim)).cuda()
        mu, _ = model.module.encoder(x)
        x_hat = model.module.decoder(mu)
        L = dim*dim * F.binary_cross_entropy(x_hat, x, reduction='none').mean()
        # L = mse(x, x_hat)
        analyser.add_to_loss_variables({'L': L}, normalization_factor=len(data_loader))
        L.backward()
        pre_train_optimizer.step()

        if save_vars:
            analyser.add_to_intermediate_variables({"x": x, "y": y, 'mu': mu})





print("Running with seed: " + str(torch.initial_seed()))
args = args_parsing()
#model = nn.DataParallel(Vade2D(z_dim=args.z_dim, h_dim=200, n_cls=3).cuda(), device_ids=range(1))
model = Vade2D(z_dim=args.z_dim, h_dim=200, n_cls=3).cuda()
#pre_train_optimizer = Adam(itertools.chain(model.module.encoder.parameters(), model.module.decoder.parameters()), lr=1e-3)
train_optimizer = Adam(model.parameters(), lr=2e-3)
lr_scheduler = StepLR(train_optimizer, step_size=10, gamma=0.95)
data_loader = return_data(args)
model.load_state_dict(torch.load('./model_dsprites.pk'))
todays_date = datetime.datetime.today()
output_dir = os.path.join(args.output_dir, str(todays_date))
writer = SummaryWriter(os.path.join(output_dir,"tensorboard"))

dim = 64

# if not os.path.exists('./pretrain_model_dsprites.pk'):
#     for epoch in range(args.pre_train_epochs):

#         start_time = time.time()
#         analyser.flush_epoch_variables()
#         pre_train(data_loader,save_vars=(epoch % args.analysis_after_epochs == 0))

#         if epoch % args.analysis_after_epochs == 0:
#             # analyser.write_state(epoch, weights=False)
#             Mu, _ = model.cpu().float().module.encoder(DS["X"].view((-1, 1, dim, dim)))
#             gmm = GaussianMixture(n_components=3, covariance_type='diag', n_init=5)
#             pre = gmm.fit_predict(Mu.detach().cpu().numpy())
#             analyser.add_to_loss_variables({"ACC": Analyser.cluster_acc(pre, DS["Y"].numpy())[0] * 100})
#             model.cuda()

#         end_time = time.time()
#         print(analyser.update_str(epoch=epoch, epoch_time=(end_time - start_time)))

#     print('Acc={:.4f}%'.format(Analyser.cluster_acc(pre, DS["Y"].numpy())[0] * 100))

#     model.module.initialize_gmm_params(gmm)
#     torch.save(model.state_dict(), './pretrain_model_dsprites.pk')
# else:
#     model.load_state_dict(torch.load('./pretrain_model_dsprites.pk'))


def train(data_loader, epoch,save_vars=False):
    


    for i ,(x, y) in tqdm(enumerate(data_loader, 0),total=len(data_loader),smoothing=0.9):
        #print("Inside the training loop")
        train_optimizer.zero_grad()
        x = x.cuda()
      
        x_hat, mu, logvar, z = model(x)
        #BCE, KLD, KLD_c, L_sparsity = model.module.losses(x, x_hat, mu, logvar, z)
        BCE, KLD, KLD_c, L_sparsity = model.losses(x, x_hat, mu, logvar, z)
        batch_loss = KLD + BCE + KLD_c + L_sparsity
        #analyser.add_to_loss_variables({'BCE': BCE, 'KLD': KLD, 'KLD_c': KLD_c, 'L_sp': L_sparsity, 'L': batch_loss}, normalization_factor=len(data_loader))

        
        batch_loss.backward()
        train_optimizer.step()
        

        #if save_vars:
            #analyser.add_to_intermediate_variables({"x": x, "y": y, 'mu': mu})
    if i >=1:
        print("last losses BCE:{}, KLD:{}, KLDc:{}, L_sparsity{}".format(BCE, KLD, KLD_c, L_sparsity))
    writer.add_scalar("Loss/ita",batch_loss,epoch)
    writer.add_scalar("BCE/ita",BCE,epoch)
    writer.add_scalar("KLD/ita",KLD,epoch)
    writer.add_scalar("KLD_catigorical/ita",KLD_c,epoch)
    writer.add_scalar("L_sparsity/ita",L_sparsity,epoch)
    
    #y_pred = model.cpu().float().module.predict(DS["X"].view((-1, 1, dim, dim)))
    #analyser.add_to_loss_variables({"ACC": Analyser.cluster_acc(y_pred, DS["Y"].numpy())[0] * 100})
    
for epoch in range(args.epochs):
    print("epoch number{}".format(epoch))
    start_time = time.time()
    #analyser.flush_epoch_variables()
    #analyser.add_to_loss_variables({"LR": lr_scheduler.get_lr()[0]})
    
    train(data_loader,epoch,save_vars=(epoch % args.analysis_after_epochs == 0))
    lr_scheduler.step(epoch)

    
    KLDs = model.kld_unit_guassians_per_cluster()
    for i in range(KLDs.shape[0]):
        dic = {}
        for j in range(KLDs.shape[1]):
            dic['u_{}'.format(j)] = KLDs[i,j]
        writer.add_scalars("KLD cluster {}/ita".format(i),dic, epoch)
    dic.clear()
    for i in range(KLDs.shape[0]):
        dic = {}
        for j in range(KLDs.shape[1]):

            dic['var_{}'.format(j)] = torch.exp(model.logvar_c[i,j])
        writer.add_scalars("Var cluster {}/ita".format(i),dic, epoch)
    dic.clear()
    #if epoch % args.analysis_after_epochs == 0:
        #analyser.write_state(epoch, weights=False)
    end_time = time.time()
    #print(analyser.update_str(epoch=epoch, epoch_time=(end_time - start_time)))
    torch.save(model.state_dict(), './model_dsprites.pk')
data_loader = iter(data_loader)
images, labels = data_loader.next()
writer.add_graph(model,images)