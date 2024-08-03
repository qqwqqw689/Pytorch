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