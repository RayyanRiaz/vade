import datetime
import os
import random

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from models.vade_archs import Vade_mnist, Vade2D

LOGS_DIR_TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
RESULTS_DIR = os.path.dirname(os.path.realpath(__file__)) + "/results/" + LOGS_DIR_TIMESTAMP + "/"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


class Analyser:
    def __init__(self, model: Vade2D, DL, z_dim, debug=True):
        self.model = model
        random.seed(0)
        self.debug = debug
        self.losses = {}
        self.intermediate_variables = {}
        self.DL = DL
        self.summary_writer = SummaryWriter(log_dir=RESULTS_DIR)
        self.z_dim = z_dim

    def add_to_loss_variables(self, losses, normalization_factor=1):
        self.losses.update({
            k: (v.item() if type(v) is torch.Tensor else v) + (self.losses[k] if k in self.losses else 0) / normalization_factor
            for k, v in losses.items()})

    def add_to_intermediate_variables(self, variables):
        for k, v in variables.items():
            if k in self.intermediate_variables:
                self.intermediate_variables[k] = torch.cat([self.intermediate_variables[k], v])
            else:
                self.intermediate_variables[k] = v

    def save_weights(self, epoch):
        torch.save(self.model.state_dict(), RESULTS_DIR + "model_weights_" + str(epoch) + ".pk")

    def flush_epoch_variables(self):
        self.losses = {k: 0 for k in self.losses.keys()}
        self.intermediate_variables = {}

    def update_str(self, epoch, epoch_time):
        losses_str = (": {:4.4f}\t\t".join(self.losses.keys()) + ": {:4.4f}").format(
            *[x for x in self.losses.values()])
        return "Epoch:{:3d}\t[time={:3.2f}s]\t\t".format(epoch, epoch_time) + losses_str

    def write_generated_sample_image(self, epoch):
        sample = torch.randn(256, self.z_dim).cuda()
        decoded_images = self.model.module.decoder(sample)
        self.summary_writer.add_images("sample_generated", decoded_images, epoch)

    def write_reconstructed_image(self, epoch):
        x = self.DL.dataset.tensors[0][:50, :].cuda().view((-1, 1, 28, 28))
        mu, _ = self.model.module.encoder(x)
        x_hat = self.model.module.decoder(mu).detach().cpu()
        self.summary_writer.add_image("sample_reconstructed", make_grid(torch.cat((x.cpu(), x_hat)), nrow=x_hat.shape[0]), epoch)

    def write_graph(self):
        print('Writing graph summary to directory: {}'.format(RESULTS_DIR))
        temp_batch_size = 1
        self.summary_writer.add_graph(self.model, torch.zeros((temp_batch_size, 1, 28, 28)), False)

    def write_losses(self, epoch):
        for k in self.losses.keys():
            # self.summary_writer.add_scalar(tag=k, scalar_value=self.losses[k] / len(self.DL.dataset), global_step=epoch)
            self.summary_writer.add_scalar(tag=k, scalar_value=self.losses[k], global_step=epoch)
        self.summary_writer.close()

    def write_embeddings(self, epoch):
        self.summary_writer.add_embedding(
            mat=self.intermediate_variables["mu"],
            metadata=self.intermediate_variables["y"].cpu().numpy(),
            label_img=self.intermediate_variables["x"].detach().cpu().view((-1, 1, 28, 28)),
            global_step=epoch
        )

    def save_state_dict(self, epoch):
        torch.save(self.model.state_dict(), RESULTS_DIR + "model_dict_epoch" + str(epoch))

    def load_state_dict(self, filename):
        self.model.load_state_dict(torch.load(filename))

    @staticmethod
    def cluster_acc(Y_pred, Y):
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        ind = [[i, j] for i, j in zip(ind[0], ind[1])]
        return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w

    def write_state(self, epoch, losses=True, generated_sample=True, reconstructed_sample=True,
                    embeddings=True, heatmap=True, state_dict=True, weights=True):
        if losses:
            self.write_losses(epoch=epoch)
        if generated_sample:
            self.write_generated_sample_image(epoch=epoch)
        if reconstructed_sample:
            self.write_reconstructed_image(epoch)
        if embeddings:
            self.write_embeddings(epoch)
        if state_dict:
            self.save_state_dict(epoch)
        if weights:
            self.save_weights(epoch)
