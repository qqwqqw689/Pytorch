import torch
import torchvision
import torchvision.transforms as transforms

# torchvision.transforms.Compose
# Composes several transforms together.
# transforms.ToTensor
# Convert a PIL Image or ndarray to tensor and scale the values accordingly.
# torchvision.transforms.Normalize(mean, std, inplace=False)
# Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels,
# this transform will normalize each channel of the input torch.*Tensor i.e.,
# output[channel] = (input[channel] - mean[channel]) / std[channel]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

# train : If True, creates dataset from training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# batch_size : how many samples per batch to load 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    # Tensor.numpy
    # Returns the tensor as a NumPy ndarray.
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
# Iterable-style datasets
# when called iter(dataset), could return a stream of data reading from a database,
# a remote server, or even logs generated in real time.
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
# torchvision.utils.make_grid
# Make a grid of images.
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.Size([4, 3, 32, 32])
        # Applies a 2D convolution over an input signal composed of several input planes.
        # in_channels, out_channels, kernel_size
        self.conv1 = nn.Conv2d(3, 6, 5)
        # torch.Size([4, 6, 28, 28])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
