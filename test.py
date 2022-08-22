import data
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from Utils import ConfusionMatrix



def test(net,data_loader):
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
    confusion.plot()
    confusion.summary()

if __name__ == '__main__':
    train_dataset, test_dataset = data.load_dataset("content")
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)

    net = torch.load('model.pth')
    test(net,train_loader)
    test(net,test_loader)