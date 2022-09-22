import os

import torch.nn as nn
import torch.nn.functional as F
import config

#10 6*8
#120_10 26*36
#2 21 *20
#5 11 *12



class Net(nn.Module):

    def __init__(self):
        output_num = len([category for category in os.listdir("content") if os.path.isdir(os.path.join("content", category))])
        dict = {1: 26 * 36,
                2: 21 * 20,
                3:16*16,
                5: 11 * 12,
                10: 6 * 8}
        self.shape_1 = 30 * dict.get(config.split_num)
        # self.shape_1 = 30 * dict.get(2)
        # self.shape_1 = 30 *21 *20
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(6,10,kernel_size=3)
        self.conv2 = nn.Conv2d(10,30,kernel_size=3)
        self.fc1 =  nn.Linear(self.shape_1, 1000)
        self.fc2 = nn.Linear(1000,100)
        self.fc3 = nn.Linear(100,output_num)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x= x.view(-1, self.shape_1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        x= F.relu(x)
        x= self.fc3(x)
        return F.softmax(x,dim=1)

if __name__ == '__main__':
    import train