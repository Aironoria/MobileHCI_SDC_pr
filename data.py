import os

from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
torch.set_printoptions(precision=4,sci_mode=False)


class FoodDataset(Dataset):
    def __init__(self, root, split_num =5,train=True, transform=None):
        self.labels = [category for category in os.listdir(root) if os.path.isdir(os.path.join(root, category))]
        self.path_list = self.get_data_list(root)
        self.split_num=split_num
        self.transform = transforms.Compose([
        transforms.Normalize([-1.005378, 0.18341783, 0.017716132, 0.021392668, -0.08647295, -0.076252915] ,[0.03257781, 0.06589742, 0.035365306, 0.27180845, 0.41038275, 0.40504447])
    ])
    def __len__(self):
        return len(self.path_list) *self.split_num

    def __getitem__(self, index):
        a = index // len(self.path_list)
        index= index  % len(self.path_list)
        path = self.path_list[index]
        label = path.split("/")[1]
        label =torch.tensor(self.labels.index(label))
        acc = pd.read_csv(os.path.join(path,"acc.csv"))
        gyro =pd.read_csv(os.path.join(path,"gyro.csv"))/1000
        b =pd.concat([acc,gyro],axis=1)
        split_len = acc.shape[0] /self.split_num
        item = b.loc[ a *split_len: (a+1) *split_len -1]
        item= torch.tensor(item.values).to(torch.float32)
        item = torch.reshape(item.T,(6,15,-1))
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

def load_dataset(root):
    torch.manual_seed(0)
    dataset = FoodDataset(root)
    train_size = int( len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset,[train_size,test_size])

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
    count=0
    for X, _ in train_loader:
        print(count,end="\t")
        for d in range(6):
            a = X[:, d, :, :].mean()
            print(a ,end="\t")
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
        print("")
        count+=1
    mean.div_(len(train_data))
    std.div_(len(train_data))
    print(list(mean.numpy()), list(std.numpy()))
    return list(mean.numpy()), list(std.numpy())



if __name__ == "__main__":
    # getStat(FoodDataset("content", split_num=1))
    print(FoodDataset("content")[0][0].size())