import os

import numpy as np
from torch import optim
import torch.nn.functional as F

import Utils
import config
import data
import model
import torch
from torch.utils.data import Dataset, DataLoader
import test
from Utils import ConfusionMatrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(train,save,save_dir=""):
  data_loader = train_loader if train else test_loader
  title = "conf_train.jpg" if train else "conf_test.jpg"
  net.eval()
  class_indict = data_loader.dataset.get_label_dict()
  label = [label for _, label in class_indict.items()]
  confusion = ConfusionMatrix(num_classes=len(label), labels=label)
  with torch.no_grad():
    for data, labels in data_loader:
      outputs = net(data)
      # test_loss += F.nll_loss(output, target, size_average=False).item()
      _, predicted = torch.max(outputs.data, 1)
      confusion.update(predicted.numpy(), labels.numpy())
  confusion.plot(save_dir ,title,save)
  # confusion.summary()

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
  # print("Test Loss: {:20.4f} ACC: {:20.2f}%".format(loss_, correct))


def train_one_epoch(epoch):
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

  # print("epoch {:4} Train Loss: {:20.4f} ACC: {:20.2f}%".format( epoch,loss_,correct),end="\t")


def get_save_dir(prefix):
  root="res"
  size = len(train_dataset) +len(test_dataset)
  if not config.use_gyro:
    suffix = "_fft"
  else:
    suffix = ""
  res= os.path.join( root,prefix +"_"+ str(size)  +"_" + str(config.n_epochs) +suffix )
  if not os.path.exists(res):
    os.makedirs(res)
  return res


for i in [0]:
  if i ==0:
    config.use_gyro =True
  else:
    config.use_gyro =False

  if not config.use_gyro:
    suffix= "fft"
  else:
    suffix=""

  root = "data_26_augmented_80%"
  train_dataset, test_dataset = data.load_dataset(root)
  print("Train dataset {} , Test Dataset {}, Total {} ".format(len(train_dataset), len(test_dataset),len(train_dataset)+len(test_dataset)))
  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

  net = model.Net(root+"/train")
  # net = torch.load(root+ suffix +".pt")
  optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  train_loss = []
  train_acc = []
  test_loss = []
  test_acc = []

  for epoch in range(250):
    train_one_epoch(epoch)
    eval(epoch)
    if epoch% 25 ==0:
      print("epoch {:4} Train Loss: {:20.4f} ACC: {:20.2f}%  Test Loss: {:20.4f} ACC: {:20.2f}%"
            .format(epoch, train_loss[-1], train_acc[-1],test_loss[-1],test_acc[-1]))
      plot_confusion_matrix(train=True,save=False)
    # if abs(train_loss[-2] - train_loss[-1]) < 0.0005:
    #   break

  pic_save_dir = get_save_dir(root)

  torch.save(net, root +suffix +'.pt')
  plot_confusion_matrix(train=True,save=True,save_dir=pic_save_dir)
  plot_confusion_matrix(train=False, save=True,save_dir=pic_save_dir)
  Utils.plot_loss(pic_save_dir, train_loss, train_acc, test_loss, test_acc)


  # net = torch.load("model.pt")
  # plot_confusion_matrix(False,False)
  # print(train_dataset.dataset.labels)



