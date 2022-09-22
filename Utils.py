import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.ticker import MultipleLocator
from prettytable import PrettyTable
from torch.utils.mobile_optimizer import optimize_for_mobile

class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        self.labels = labels  # 类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+120_10

    def get_acc(self):
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        return  sum_TP / n  # 总体准确率


    def summary(self):  # 计算指标函数
        n = np.sum(self.matrix)
        # calculate accuracy
        print("the model accuracy is ", str(self.get_acc()))

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()  # 创建一个表格
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):  # 精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.  # 每一类准确度
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self,root,tittle,save):  # 绘制混淆矩阵
        matrix = self.matrix
        # print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc=' + str(self.get_acc()) + ')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(root, tittle),bbox_inches = 'tight')
        else:
            plt.show()
        plt.clf()




def plot_loss(save_dir,train_loss, train_acc , test_loss, test_acc):
    plt.plot(np.arange(len(train_loss)), train_loss, label="train loss.jpg")
    plt.plot(np.arange(len(test_loss)), test_loss, label="valid loss.jpg")

    plt.title('loss.jpg')
    plt.legend()  # 显示图例
    plt.savefig(os.path.join(save_dir, "loss.jpg"),bbox_inches = 'tight')
    # plt.show()
    plt.clf()
    plt.plot(np.arange(len(train_acc)), train_acc, label="train acc.jpg")

    plt.plot(np.arange(len(test_acc)), test_acc, label="valid acc.jpg")
    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    # plt.ylabel("epoch")
    plt.title('acc.jpg')
    plt.savefig(os.path.join(save_dir, "acc.jpg"),bbox_inches = 'tight')
    # plt.show()
    plt.clf()


def plot_data(acc,gyro,label,file):

    locator = 150
    index = range(1, len(acc['x']) + 1)

    plt.rcParams["figure.figsize"] = (20, 10)
    plt.subplot(2,1,1)

    plt.plot(index, acc['x'], label='x', linestyle='solid', marker=',')
    plt.plot(index, acc['y'], label='y', linestyle='solid', marker=',')
    plt.plot(index, acc['z'], label='z', linestyle='solid', marker=',')
    plt.gca().xaxis.set_major_locator(MultipleLocator(locator))
    plt.title(file)
    plt.xlabel("Sample #")
    plt.ylabel("Acceleration (G)")
    plt.legend()


    plt.subplot(2,1,2)
    plt.plot(index, gyro['pitch'], label='pitch', linestyle='solid', marker=',')
    plt.plot(index, gyro['roll'], label='roll', linestyle='solid', marker=',')
    plt.plot(index, gyro['yaw'], label='yaw', linestyle='solid', marker=',')
    plt.gca().xaxis.set_major_locator(MultipleLocator(locator))
    plt.xlabel("Sample #")
    plt.ylabel("Gyroscope (deg/sec)")
    plt.legend()
    plt.show()
    # path = os.path.join("pic",label)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # plt.savefig("pic/"+label+"/"+file)
    plt.clf()




def pth_to_pt():
    model = torch.load("model.pth")
    model.eval()
    input = torch.rand(1,6,15,16)
    torch.jit.trace(model,input).save("model.pt")

def pt_to_ptl():
    model = torch.load("model2.pt")
    model.eval()
    scripted_module = torch.jit.script(model)
    optimize_for_mobile(scripted_module)._save_for_lite_interpreter("model2.ptl")

def plot_dir(dir):
    for file in os.listdir( dir):
        acc = pd.read_csv(dir+'/'+file+'/ACC.csv')
        if(file == "wall"):
            acc['y'] -=4
        gyro = pd.read_csv(dir+'/'+file+'/GYRO.csv')
        plot_data(acc, gyro,dir, file)

def convert_to_edgeimpulse(root,save_dir):
    name = "cjy"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    ids = [ i for i in range(1200)]
    for gesture in os.listdir(root):
        dir = os.path.join(root,gesture)
        for filename in os.listdir(dir):
            acc = pd.read_csv(dir + "/" + filename + "/ACC.csv")
            gyro = pd.read_csv(dir + '/' + filename + '/GYRO.csv')
            data = pd.concat([acc,gyro],axis=1)
            # data['timestamp']= ids
            # data = pd.DataFrame(data,columns=data,index=ids)
            data.to_csv(save_dir+ "/"+ gesture + "."+filename+".csv",index_label="timestamp")

def edgeimpulse_to_csv(root,save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for file in os.listdir(root):
        gesture = file.split(".")[0]
        save_file_parent = os.path.join(save_dir,gesture)
        if not os.path.exists(save_file_parent):
            os.mkdir(save_file_parent)
        save_file = os.path.join(save_file_parent,str(len(os.listdir(save_file_parent))) +".csv")
        with open(os.path.join(root,file),encoding="utf-8") as f:
            df = json.load(f)['payload']['values']
            df = pd.DataFrame(df)
            if(len(df) < 400):
                print(file + " only has " + str(len(df)) + " lines")
            df = df.loc[0:399]
            df.to_csv(save_file,index= False)


if __name__ == '__main__':
    # pth_to_pt()

    # dir = "TripleClick"
    # for file in os.listdir("content/" + dir):
    #     acc = pd.read_csv('content/'+dir+'/'+file+'/ACC.csv')
    #     gyro = pd.read_csv('content/'+dir+'/'+file+'/GYRO.csv')
    #     plot_data(acc, gyro,dir, file)
    # pt_to_ptl()
    # plot_dir("Collection")

    # convert_to_edgeimpulse("content1","edge1" )

    edgeimpulse_to_csv("edge_after","edge_after_converted")
