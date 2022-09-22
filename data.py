import os
import random

from torchvision import transforms
import config
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
torch.set_printoptions(precision=4,sci_mode=False)



class FoodDataset(Dataset):
    def __init__(self, root, split = False,train=True, transform=None):

        mean_dict = {
            100:[-0.9608181, 0.31008407, -0.009351776, 0.0054555144, -0.23369068, -0.3488236],
            10:[-0.9608181, 0.31008407, -0.009351776, 0.054554835, -2.3369055, -3.4882338],
            1:[-0.9608181, 0.31008407, -0.009351776, 0.54554874, -23.369055, -34.882328],
            1000:[-0.9608181, 0.31008407, -0.009351776, 0.0005455491, -0.023369053, -0.034882355],
            5000:[-0.972108, 0.29212847, 0.0144133, -0.00012562927, -0.0036951215, -0.005630902]
        }
        std_dict={
            100: [0.029218799, 0.04372534, 0.02839355, 2.1205835, 3.1814947, 3.1679814],
            10:[0.029218799, 0.04372534, 0.02839355, 21.205835, 31.814943, 31.679813],
            1:[0.029218799, 0.04372534, 0.02839355, 212.05826, 318.1492, 316.7982],
            1000:[0.029218799, 0.04372534, 0.02839355, 0.21205823, 0.31814966, 0.3167982],
            5000:[0.024947584, 0.033967372, 0.021579275, 0.0327926, 0.04770578, 0.04992873]
        }

        self.gyrobase = 1000
        self.labels = [category for category in os.listdir(root) if os.path.isdir(os.path.join(root, category))]
        self.path_list = self.get_data_list(root)
        self.length = len(self.path_list)
        self.transform = transforms.Compose([
         # transforms.Lambda(lambda x : self.min_max_scaler(x)),
        transforms.Normalize(
            mean_dict.get(self.gyrobase),
            std_dict.get(self.gyrobase)
        )
        ])


    def min_max_scaler(self, data):
        maxs = [0.02758789, 1.3986816, 0.5061035, 7123.1074, 9545.349, 8531.921]

        mins =[-1.7145996, -1.420166, -0.8881836, -6555.7246, -10552.185, -7045.715]
        for d in range(6):
            data[:, d] =data[:,d].mul(100).add(-mins[d]).div((maxs[d] - mins[d]))
        return data

    def __len__(self):
        return len(self.path_list)


    def __getitem__(self, index):
        a = index // len(self.path_list)
        index= index  % len(self.path_list)
        path = self.path_list[index]
        label = path.split("/")[1]
        label =torch.tensor(self.labels.index(label))

        item = pd.read_csv(path)
        item = item / [1,1,1,1000,1000,1000]
        item = torch.tensor(item.values).to(torch.float32)
        # item = self.min_max_scaler(item)


        # 10   : 10
        #120_10  : 30

        #2 ： 25
        #5： 15
        shape_dict = {1:30,
                      2:25,
                      3:20,
                      5:15,
                      10:10}

        item = torch.reshape(item.T,(6,20,-1))

        if self.transform:
            item = self.transform(item)
        return item ,label

    def get_data_list(self, root):
        res = list()
        for category in self.labels:
            for file in os.listdir(os.path.join(root, category)):
                res.append(os.path.join(root, category, file))
        return res
    def get_label_dict(self):
        res ={}
        for i in range (len(self.labels)):
            res[i]=self.labels[i]
        return res

def split(dataset):
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])

def load_dataset(root):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    dataset = FoodDataset(root)
    return split(dataset)


# a = FoodDataset("content")
# b = a[300]
# print(b)
def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(6)
    std = torch.zeros(6)
    data_max = torch.zeros(6)
    data_min = torch.zeros(6)
    count=0
    for X, _ in train_loader:
        print(count,end="\t")
        for d in range(6):
            a = X[:, d, :, :].mean()
            print(a ,end="\t")
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
            data_max[d] = max( data_max[d], X[:, d, :, :].max())
            data_min[d] = min(data_min[d], X[:, d, :, :].min())
        print("")
        count+=1
    mean.div_(len(train_data))
    std.div_(len(train_data))
    print(list(mean.numpy()), list(std.numpy()))
    # print(list(data_max.numpy()), list (data_min.numpy()))
    return list(mean.numpy()), list(std.numpy())



if __name__ == "__main__":

    a = FoodDataset("edge_after_converted")
    a.transform=None
    getStat(a)
    print(a.labels)

    # train,test =  load_dataset("content")

