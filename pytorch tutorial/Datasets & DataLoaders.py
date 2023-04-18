# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
# torchvision.transforms.ToTensor : Convert a PIL Image or ndarray to tensor and scale the values accordingly.

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() 
    # randint : Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
    # size (tuple) – a tuple defining the shape of the output tensor.
    # (1,) represents the shape of a one-dimensional array with a single element
    # item() : Returns the value of this tensor as a standard Python number
    # This only works for tensors with one element.
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i) 
    # (int, int, index) :  
    # The subplot will take the index position on a grid with nrows rows and ncols columns. 
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    # numpy.squeeze(a, axis=None)
    # Remove axes of length one from a ( all or a subset of the dimensions of length 1 removed.)
plt.show()

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# batch_size (int, optional) – how many samples per batch to load (default: 1).
# shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).

'''
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        # DataFrame or TextFileReader
        # A comma-separated values (csv) file is returned as two-dimensional data structure with labeled axes.
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # The __len__ function returns the number of samples in our dataset.
        return len(self.img_labels)

    def __getitem__(self, idx):
        # The __getitem__ function loads and returns a sample from the dataset at the given index idx. 
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # iloc[] : Purely integer-location based indexing for selection by position.
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
'''