import os

import numpy as np
from torch import optim
import torch.nn.functional as F
import data
import model
import torch
from torch.utils.data import Dataset, DataLoader
import test
from Utils import ConfusionMatrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(train):
  data_loader = train_loader if train else test_loader
  title = "conf_train.jpg" if train else "conf_test.jpg"
  net.eval()
  class_indict = data_loader.dataset.dataset.get_label_dict()
  label = [label for _, label in class_indict.items()]
  confusion = ConfusionMatrix(num_classes=len(label), labels=label)
  with torch.no_grad():
    for data, labels in data_loader:
      outputs = net(data)
      # test_loss += F.nll_loss(output, target, size_average=False).item()
      _, predicted = torch.max(outputs.data, 1)
      confusion.update(predicted.numpy(), labels.numpy())
  confusion.plot(save_dir ,title)
  confusion.summary()

def eval(epoch):
  net.eval()
  correct =0
  loss_=0
  with torch.no_grad():
    for data, labels in test_loader:
      outputs = net(data)
      loss = F.cross_entropy(outputs, labels)
      loss_+=loss.item()
      _, predicted = torch.max(outputs.data, 1)
      correct += predicted.eq(labels.data.view_as(predicted)).sum().item()
  data_len = len(test_loader.dataset)
  loss_ = loss_ / len(test_loader)
  test_loss.append(loss_)
  correct = correct * 100 / data_len
  test_acc.append(correct)
  print("Test Loss: {:20.4f} ACC: {:20.2f}%".format(loss_, correct))


def train(epoch):
  net.train()
  correct = 0
  loss_=0
  for batch_idx, (data, labels) in enumerate(train_loader):
    optimizer.zero_grad()
    outputs = net(data)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()

    loss_+=loss.item()


    _, predicted = torch.max(outputs.data, 1)
    correct += predicted.eq(labels.data.view_as(predicted)).sum().item()
    # if batch_idx % 2 == 0:
    #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc:{:.6f}'.format(
    #     epoch, batch_idx * len(data), len(train_loader.dataset),
    #     100. * batch_idx / len(train_loader), loss.jpg.item(),correct/(log_interval * len(data))),end="\n")
  data_len = len(train_loader.dataset)
  loss_ = loss_ / len(train_loader)
  train_loss.append(loss_)
  correct =correct *100 / data_len
  train_acc.append(correct)
  print("epoch {:4} Train Loss: {:20.4f} ACC: {:20.2f}%".format( epoch,loss_,correct),end="\t")


save_dir = "result/2"
os.makedirs(save_dir)
n_epochs = 3
torch.set_printoptions(precision=4, sci_mode=False)
train_dataset, test_dataset = data.load_dataset("content")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

net = model.Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_loss=[]
train_acc =[]
test_loss=[]
test_acc=[]

for epoch in range(n_epochs):
  train(epoch)
  eval(epoch)
torch.save(net, 'model.pth')
plt.plot(np.arange(len(train_loss)), train_loss,label="train loss.jpg")
plt.plot(np.arange(len(test_loss)), test_loss, label="valid loss.jpg")

plt.title('loss.jpg')
plt.legend() #显示图例
plt.savefig(os.path.join( save_dir, "loss.jpg"))
plt.show()
plt.clf()
plt.plot(np.arange(len(train_acc)), train_acc, label="train acc.jpg")

plt.plot(np.arange(len(test_acc)), test_acc, label="valid acc.jpg")
plt.legend() #显示图例
plt.xlabel('epoches')
#plt.ylabel("epoch")
plt.title('acc.jpg')
plt.savefig(os.path.join( save_dir, "acc.jpg"))
plt.show()
plt.clf()
plot_confusion_matrix(True)
plot_confusion_matrix(False)
