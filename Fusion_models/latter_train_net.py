import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import os
import time
# 第一个是resnet
from utils.resnet import generate_Res_model
# 加了注意力机制进行early fusion
from utils.decision_model import Decison_fusion
from utils.load_data import Fusion_dataset
from torch.nn.functional import cross_entropy
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")



local_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())

parser = argparse.ArgumentParser()

parser.add_argument("--date",default=local_time.split(" ")[0],type=str)

# 训练次数
parser.add_argument("--epoch",default=80,type=int)

# 保存频次，这里是每隔10次就保存一下
parser.add_argument("--save_times",default=20,type=int)


# 学习率
parser.add_argument("--lr",default=0.002,type=float)

# 数据集的存储位置 分别是两个数据集的训练数据位置
parser.add_argument("--data1_dir",default=r'Fusion_models\Features\fea_wifi_3000x512.mat',type=str)

parser.add_argument("--data2_dir",default=r"Fusion_models\Features\if10rgbflow_3000x4096.mat",type=str)

parser.add_argument("--label_dir",default=r"Fusion_models\Features\label1_3000x1.mat",type=str)
# 批次大小
#adiac选32 beef选4
parser.add_argument("--batch_size",default=64,type=int)

# 随机种子
parser.add_argument("--seed",default=123,type=int)

# 权重文件所在的文件夹
parser.add_argument("--save_path",default="checkpoint",type=str)

# 训练过程的情况
parser.add_argument("--log",default="log",type=str)

# 模型名称SE_Resnet
parser.add_argument("--model_name",default="Resnet",type=str)

# txt文件的记录编号 0
parser.add_argument("--log_index",default=0,type=int)

# 分别代表加法 堆叠 注意力机制(本身就是卷积神经网络，所以不作卷积)
parser.add_argument("--model_type",default=0,type=int)

# 要进行分类的类别数
# adiac 37 beef 5
parser.add_argument("--classes",default=10,type=int)

args = parser.parse_args()




def dir_is_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train_model(model,log_index=0,fusion_type=0,model_type=0):
    torch.manual_seed(args.seed)

    # 训练集
    train_set = Fusion_dataset(args.data1_dir,args.data2_dir,args.label_dir)
    # 为了增强泛化性，随机打乱数据
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True)

    # 测试集
    test_set = Fusion_dataset(args.data1_dir,args.data2_dir,args.label_dir,mode="test")
    # 这样就能单独计算某一类的p,r,f1了
    test_loader = DataLoader(test_set,batch_size=args.batch_size)

    model = torch.nn.DataParallel(model).cuda()

    #选择优化器
    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    # min_loss作为最优的模型的评价指标，loss最低的就是最好的模型
    min_loss = 100000
    # 记录最好的模型是哪一个epoch
    best_epoch = 0

    # 模型文件存放位置
    root = os.path.abspath(os.path.dirname(__file__))

    save_dir = os.path.join(args.save_path,args.model_name + "_" + args.date)
    dir_is_exist(save_dir)

    # txt记录文件的存放位置
    log_dir = os.path.join(root,args.log,args.model_name + "_" + args.date)
    dir_is_exist(log_dir)

    # 记录训练的情况
    log_name = "log_train_"+ str(log_index) +".txt"

    log_path = os.path.join(log_dir,log_name)

    # 记录测试的情况
    log_name_ = "log_test_" + str(log_index) + ".txt"

    log_path_ = os.path.join(log_dir, log_name_)

    # 记录loss的txt文件
    with open(log_path,"w") as f:
        f.write("Epoch,Loss,Acc\n")

    # 记录loss的txt文件
    with open(log_path_, "w") as f:
        f.write("Epoch,Loss,Acc,macro-P,macro-R,macro-F\n")

    print("Start!!!")
    for epoch in range(args.epoch):

        model.train()

        epoch_loss = 0

        # 训练开始时间
        epoch_start = time.time()
        correct = 0

        for i,data in enumerate(train_loader):

            optimizer.zero_grad()

            # 输入参数，标签
            x1,x2,label = data

            x1 = x1.cuda()
            x2 = x2.cuda()
            label = label.cuda()

            # 进行分类
            y = model(x1,x2)

            loss = cross_entropy(y,label)

            loss.backward()

            optimizer.step()

            pred = y.argmax(dim=1)
            cor = pred.eq(label).sum().item()
            correct += cor
            epoch_loss += loss.item()

        accuracy = 100. * correct / len(train_set)
        print(accuracy)
        # 记录每一次的loss
        with open(log_path,"a") as f:
            f.write("{},{:.3f},{:.3f}\n".format(epoch,epoch_loss,accuracy))


        test_loss = 0

        # 测试模型
        with torch.no_grad():

            model.eval()
            correct = 0
            predicts = []
            labels = []
            for i,data in enumerate(test_loader):

                x1, x2, target = data
                x1 = x1.cuda()
                x2 = x2.cuda()
                target = target.cuda()

                output = model(x1,x2)
                test_loss += cross_entropy(output, target).item()

                pred = output.argmax(dim=1)

                predicts.append(pred)
                labels.append(target)


                cor = pred.eq(target).sum().item()
                correct += cor

            all_predict = torch.cat(predicts,dim=-1)
            all_label = torch.cat(labels,dim=-1)

            all_label = all_label.detach().cpu().numpy()
            all_predict = all_predict.detach().cpu().numpy()
            precision, recall, f1, _ = precision_recall_fscore_support(all_label, all_predict, average='macro')

            accuracy = 100. * correct / len(test_set)
            # 记录每一次的loss
            with open(log_path_, "a") as f:
                f.write("{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(epoch, test_loss, accuracy,precision, recall, f1))


        epoch_end = time.time()

        spend_time = epoch_end - epoch_start

        print("Still need {:.2f} h to train".format((spend_time/3600)*(args.epoch-epoch)))

        # 保存最好模型
        if test_loss < min_loss:
            min_loss = test_loss
            file_path = os.path.join(save_dir,"best_epoch_model.pth")
            best_epoch = epoch
            torch.save(
                {
                    "Resnet":model.state_dict(),
                },
                file_path
            )

        # 每隔一定次数就保存模型,保存最后几个模型
        if epoch % args.save_times == 0 or (epoch + 1) % int(args.epoch - 1) == 0 \
                or (epoch + 1) % int(args.epoch - 2) == 0 \
                or (epoch + 1) % int(args.epoch - 3) == 0:
            file_path = os.path.join(save_dir, str(epoch)+"_epoch_model.pth")
            torch.save(
                {
                    "Resnet": model.state_dict(),
                },
                file_path
            )


        print(f"The best epoch is {best_epoch}")


def select_model(model_type=0):
    # 分别代表预测分数取平均值0、最大值1、相乘2、相加3、注意力机制4
    assert model_type in [0,1,2,3,4]
    # 先512形状 再4096形状
    model1 = generate_Res_model(10,n_input_channels=1,n_classes=args.classes)
    model2 = generate_Res_model(18,n_input_channels=1,n_classes=args.classes)

    model = Decison_fusion(model1,model2,fusion_type=model_type)
    args.model_name = "Decision_Resnet"+str(model_type)
    return model

if __name__ == "__main__":
    # net = generate_model(50,n_input_channels=1,n_classes=args.classes)
    for i in range(5):
        args.model_type = i
        print("The fusion type is ",args.model_type)
        net = select_model(args.model_type)
        train_model(net,log_index=args.log_index,model_type=args.model_type)
