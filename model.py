import torch.nn as nn
import torch.nn.functional as F
import config

#10 6*8
#120_10 26*36
#2 21 *20
#5 11 *12



class Net(nn.Module):

    def __init__(self):
        dict = {1: 26 * 36,
                2: 21 * 20,
                5: 11 * 12,
                10: 6 * 8}
        self.shape_1 = 30 * dict.get(config.split_num)

        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(6,15,kernel_size=3)
        self.conv2 = nn.Conv2d(15,30,kernel_size=3)
        self.fc1 =  nn.Linear(self.shape_1, 200)
        self.fc2 = nn.Linear(200,4)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x= x.view(-1, self.shape_1)
        x=self.fc1(x)
        x=self.fc2(x)
        return F.softmax(x,dim=1)

if __name__ == '__main__':
    import train