import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST


def get_mnist(data_dir='./data/mnist/', batch_size=128):
    train = MNIST(root=data_dir, train=True, download=True)
    test = MNIST(root=data_dir, train=False, download=True)

    X = torch.cat([train.data.float().view(-1, 784) / 255., test.data.float().view(-1, 784) / 255.], 0)
    Y = torch.cat([train.targets, test.targets], 0)

    dataset = dict()
    dataset['X'] = X
    dataset['Y'] = Y

    dataloader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader, dataset
