import torchvision.transforms
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    # date of type pandas.dataframe a container structure that stores the information found in the file ”data.csv”
    # mode of type String can be val or train
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = tv.transforms.Compose(
            [torchvision.transforms.ToPILImage(),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(train_mean, train_std)])

    def __len__(self):  # return len of currently loaded data
        return len(self.data)

    def __getitem__(self, index):

        img = imread(self.data.iloc[index, 0])
        img = gray2rgb(img)
        if self._transform:
            img = self._transform(img)
        crack = self.data.iloc[index, 1]
        inactive = self.data.iloc[index, 2]

        return img, torch.tensor([crack, inactive], dtype=torch.float)