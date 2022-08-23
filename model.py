import torch.nn as nn
import torch.nn.functional as F

shape_1 = 30*6*8

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(6,15,kernel_size=3)
        self.conv2 = nn.Conv2d(15,30,kernel_size=3)
        self.fc1 =  nn.Linear(shape_1, 200)
        self.fc2 = nn.Linear(200,4)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x= x.view(-1, shape_1)
        x=self.fc1(x)
        x=self.fc2(x)
        return F.softmax(x,dim=1)

if __name__ == '__main__':
    import train