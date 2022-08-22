from torch import optim
import torch.nn.functional as F
import data
import model
import torch
from torch.utils.data import Dataset, DataLoader
import test
from Utils import ConfusionMatrix



def train(net,epoch):
  net.train()
  correct = 0
  for batch_idx, (data, labels) in enumerate(train_loader):

    optimizer.zero_grad()
    outputs = net(data)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()
    _, predicted = torch.max(outputs.data, 1)
    correct += predicted.eq(labels.data.view_as(predicted)).sum().item()
    # if batch_idx % log_interval == 0:
    #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc:{:.6f}'.format(
    #     epoch, batch_idx * len(data), len(train_loader.dataset),
    #     100. * batch_idx / len(train_loader), loss.item(),correct/(log_interval * len(data))),end="\n")
  print("epoch {} ACC: {:.2f}%".format( epoch,correct *100 / len(train_loader.dataset)))



# if __name__ == '__main__':
n_epochs = 120
torch.set_printoptions(precision=4, sci_mode=False)
train_dataset, test_dataset = data.load_dataset("content")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

net = model.Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(n_epochs):
  train(net,epoch)
torch.save(net, 'model.pth')

test.test(net,train_loader)
test.test(net,test_loader)
