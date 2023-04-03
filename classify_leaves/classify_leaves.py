import random
import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

train_images = train_data.iloc[:, 0].values 
# iloc() Purely integer-location based indexing for selection by position
# DataFrame.values Return a Numpy representation of the DataFrame.
# ['images/0.jpg' 'images/1.jpg' 'images/2.jpg' ... 'images/18350.jpg' 'images/18351.jpg' 'images/18352.jpg']

pred_images = test_data.iloc[:, 0].values
train_labels = pd.get_dummies(train_data.iloc[:, 1]).values.argmax(1)
# get_dummies() Convert categorical variable into dummy/indicator variables.
# numpy.argmax() Returns the indices of the maximum values along an axis.
# train_labels: [ 78  78  78 ...  40 125 144]

train_labels_header = pd.get_dummies(train_data.iloc[:, 1]).columns.values
# pandas.DataFrame.columns : the column labels of the DataFrame.
# ['abies_concolor' 'abies_nordmanniana' 'acer_campestre' 'acer_ginnala' ... 'ulmus_rubra' 'zelkova_serrata']

print(train_data.iloc[:, 1])
n_train = train_images.shape[0]

class Classify_leaves(torch.utils.data.Dataset):
    def __init__(self, root, images, labels, transform):
        super().__init__()
        self.root = root
        self.images = images
        if labels is None:
            self.labels = None
        else:
            self.labels = labels
        self.transform = transform
    
    def __getitem__(self, index):
        '''The __getitem__ function loads and returns a sample from the dataset at the given index'''
        image_path = os.path.join(self.root, self.images[index])
        image = Image.open(image_path)
        image = self.transform(image)
        if self.labels is None:
            return image
        label = torch.tensor(self.labels[index])
        return image, label
    
    def __len__(self):
        '''The __len__ function returns the number of samples in our dataset.'''
        return self.images.shape[0]
    
def load_data(images, labels, batch_size, train):
    aug = []
    if (train):
        aug = [transforms.RandomHorizontalFlip(),
               transforms.RandomVerticalFlip(),
               transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.5),
               transforms.ToTensor()]
        # Horizontally flip the given image randomly with a given probability.
        # Vertically flip the given image randomly with a given probability.
        # Change the brightness, contrast, saturation and hue of an image.
        # Convert a PIL Image or ndarray to tensor and scale the values accordingly.
    else:
        aug = [transforms.ToTensor()]

    transform = transforms.Compose(aug) 
    dataset = Classify_leaves('./data', images, labels, transform=transform)
    # dataset (Dataset) – dataset from which to load the data.
    # batch_size (int, optional) – how many samples per batch to load
    # shuffle (bool, optional) – set to True to have the data reshuffled at every epoch 
    # num_workers (int, optional) – how many subprocesses to use for data loading.
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, shuffle=train)

net = torchvision.models.resnet18()

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)
    # nn.Conv2d applies a 2D convolution over an input signal composed of several input planes.
    # nn.init.xavier_normal Fills the input Tensor with values according to a method

net.apply(init_weights)

def accuracy(y_hat, y):
    return (y_hat.argmax(1) == y).sum()

def train(net, train_iter, test_iter, num_epochs, lr, devices):
    net = nn.DataParallel(net, device_ids = devices).to(devices[0])
    # Implements data parallelism at the module level.
    # device_ids (list of python:int or torch.device) – CUDA devices
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    train_loss = []
    train_acc = []
    test_acc = []
    train_tot_list = []
    test_tot_list = []
    for epoch in range(num_epochs):
        train_loss_tot, train_acc_tot, train_tot = 0, 0, 0
        test_acc_tot, test_tot = 0, 0
        net.train()
        for X,y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step() 
            with torch.no_grad():
                train_loss_tot += l*X.shape[0]
                train_acc_tot += accuracy(y_hat, y)
                train_tot += X.shape[0]
            net.eval()
            with torch.no_grad():
                for X,y in test_iter:
                    X, y = X.to(devices[0]), y.to(devices[0])
                    train_acc_tot += accuracy(net(X), y)
                    test_tot += X.shape[0]
            
            train_loss.append(train_loss_tot / train_tot)
            train_acc.append(train_acc_tot / train_tot)
            test_acc.append(test_acc_tot / test_tot)
            train_tot_list.append(train_tot)
            test_tot_list.append(test_tot_list)
    train_loss_np = np.array(train_loss)
    train_acc_np = np.array(train_acc)
    test_acc_np = np.array(test_acc)
    train_tot__np = np.array(train_tot_list)
    test_tot_np = np.array(test_tot_list)
    plt.figure()
    plt.subplot(311)
    plt.plot(train_tot__np, train_loss_np)
    plt.subplot(312)
    plt.plot(train_tot__np, train_acc_np)
    plt.subplot(313)
    plt.plot(test_tot_np, test_acc_np)
    plt.suptitle('train_loss_np  train_acc_np  test_acc_np')
    plt.show()

train_slices = random.sample(list(range(n_train)), 15000)
test_slices = list(set(range(n_train)) - set(train_slices))
# set A = {10, 20, 30, 40, 80}
# set B = {100, 30, 80, 40, 60}
# set A - set B = {10, 20}

train_iter = load_data(train_images[train_slices], train_labels[train_slices], 512, train=True)
test_iter = load_data(train_images[test_slices], train_labels[test_slices], 512, train=False)

train(net, train_iter, test_iter, 10, 0.01, [torch.device('cuda:0')])


pred_iter = load_data(pred_images, None, 256, train=False)

def predict(net, pred_iter):
    net.to(torch.device('cuda:0'))
    net.eval()
    prediction = []
    for index, X in enumerate(pred_iter):
        # enumerate(): return: The count of the current iteration, The value of the item at the current iteration
        X = X.to('cuda:0')
        prediction.extend(train_labels_header[net(X).argmax(1).cpu()])
        # extend() method adds the specified list elements (or any iterable) to the end of the current list.
        # argmax() Returns the indices of the maximum values of a tensor across a dimension.
        # cpu() Returns a copy of this object in CPU memory.
    test_data['label'] = prediction
    test_data.to_csv('./data/submission.csv', index=False)

predict(net,pred_iter)