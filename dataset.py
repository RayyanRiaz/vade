"""dataset.py"""

import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0



def get_mnist(data_dir='./data/mnist/', batch_size=128):
    from torchvision.datasets import MNIST
    train = MNIST(root=data_dir, train=True, download=True)
    test = MNIST(root=data_dir, train=False, download=True)

    X = torch.cat([train.data.float().view(-1, 784) / 255., test.data.float().view(-1, 784) / 255.], 0)
    Y = torch.cat([train.targets, test.targets], 0)

    dataset = dict()
    dataset['X'] = X
    dataset['Y'] = Y

    dataloader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader, dataset


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        img1 = self.data_tensor[index1]
        img2 = self.data_tensor[index2]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

    def __len__(self):
        return self.data_tensor.size(0)

class Fullloaded_dataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        #print("reached point 2")
        (self.data_tensor,self.labels) = data_tensor
        self.label = data_tensor
        self.transform = transform
        self.indices = range(len(self))
        
    def __getitem__(self, index1):

        img1 = self.data_tensor[index1]
        img1 = torch.from_numpy(img1).float()
        label = self.labels[index1]
        #label = torch.from_numpy(label).float()

        if self.transform is not None:
            img1 = self.transform(img1)
        

        return img1, label

    def __len__(self):
        return self.data_tensor.shape[0]

def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    #assert image_size == 64, 'currently only image size of 64 is supported'

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),])
    if name.lower() == 'mnist':
        DL,dset = get_mnist(data_dir='./data', batch_size=args.batch_size)
        return DL
    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
    elif name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
    elif name.lower() == 'dsprites':
        #print("reached point one")
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='latin1')
        #print(data.shape)
        imgs = np.expand_dims(data['imgs'], axis=1)
        labels = data['latents_values'][:,1]
        
        #data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':(imgs,labels)}
        dset = Fullloaded_dataset
    else:
        raise NotImplementedError


    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              #num_workers=num_workers,
                              #pin_memory=True,
                              drop_last=True)

    data_loader = train_loader
    return data_loader
