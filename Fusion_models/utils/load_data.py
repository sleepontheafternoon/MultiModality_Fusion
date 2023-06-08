import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from scipy import io
import torch

def trans(x1,x2,y):

    x1 = torch.from_numpy(x1).float()
    x1 = torch.unsqueeze(x1,dim=1)

    x2 = torch.from_numpy(x2).float()
    x2 = torch.unsqueeze(x2, dim=1)

    y = torch.from_numpy(y).long()
    y = torch.squeeze(y,dim=-1)

    return x1,x2,y




class Fusion_dataset(Dataset):

    def __init__(self,data1_path=r'F:\代码和数据集-5.24\提取的红外和WiFi特征\fea_wifi_3000x512.mat',
                 data2_path=r"F:\代码和数据集-5.24\提取的红外和WiFi特征\if10rgbflow_3000x4096.mat",
                 label_path=r"F:\代码和数据集-5.24\提取的红外和WiFi特征\label1_3000x1.mat",
                 ratio=0.7,mode="train"):

        super(Fusion_dataset, self).__init__()
        self.mode = mode

        # 获得数据
        data1 = io.loadmat(data1_path)
        data2 = io.loadmat(data2_path)
        label = io.loadmat(label_path)

        # 得到numpy数组
        data1 = data1["fea_wifi"]
        data2 = data2["if10rgbflow"]
        label = label["label1"]

        # 分别得到每个类别的下标索引
        # ls1 ls2 ls3
        ls1 = []
        ls2 = []
        ls3 = []

        # 得到相应的数据
        #  在ratio为0.7的情况下
        # 获取每个类别前210个向量
        num = int(ratio*300)
        if self.mode == "train":
            # i代表哪一个类别
            for i in range(10):
                index = i*300
                need_index = list(range(index,index+num))
                ls1.extend(need_index)
                ls2.extend(need_index)
                ls3.extend(need_index)
        # 获取每个类别后90个向量
        else:
            for i in range(10):
                index = i * 300
                need_index = list(range(index + num,index+300))
                ls1.extend(need_index)
                ls2.extend(need_index)
                ls3.extend(need_index)

        data1 = data1[ls1,:]
        data2 = data2[ls2,:]
        data3 = label[ls3,:]

        self.x1,self.x2,self.y = trans(data1,data2,data3)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, item):
        return self.x1[item],self.x2[item],self.y[item]


if __name__ == "__main__":
    train_set = Fusion_dataset()
    train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
    for i,v in enumerate(train_loader):
        print(i)
        print(v[0].shape)
        print(v[1].shape)
        print(v[2].shape)
        break








#
# for i in range(10):
#     for j in range(300):
#         index = i*300 + j
#         if result[index][0] != i:
#             print("Error")
#             break
# print("Over!!!")
