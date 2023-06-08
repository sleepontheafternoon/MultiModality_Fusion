import torch
import torch.nn as nn
# from Fusion_models.utils.resnet import generate_Res_model


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels*2, out_channels=in_channels*2, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.ln = nn.Linear(20,10)


    def forward(self, x):

        # 计算输入张量的大小
        x_mean = torch.mean(x,dim=1,keepdim=True)
        x_max,_ = torch.max(x,dim=1,keepdim=True)
        x_sub = torch.cat([x_mean,x_max],dim=1)
        x_sub = self.conv(x_sub)
        x_sub = self.sigmoid(x_sub)
        y = x_sub*x
        y = y.view(y.shape[0],-1)
        # 通过线性层，变为我们想要的输出
        return self.ln(y)



class Decison_fusion(nn.Module):

    # 预测分数取平均值0、最大值1、相乘2、相加3、注意力机制4、知识蒸馏5
    def __init__(self,model1,model2,fusion_type=0):
        super(Decison_fusion, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.fusion_type = fusion_type
        if fusion_type != 4 and fusion_type != 5:
            self.layer = nn.Linear(10,10)
        elif fusion_type == 4:
            self.layer = SpatialAttention(in_channels=1)
        elif fusion_type == 5:
            self.layer = nn.Linear(20,10)
        self.soft = nn.Softmax(dim=-1)

    def forward(self,x1,x2):
        y1 = self.model1(x1)
        y2 = self.model2(x2)
        if self.fusion_type == 0:
            y = (y1+y2)/2
        elif self.fusion_type == 1:
            y1 = torch.unsqueeze(y1,dim=1)
            y2 = torch.unsqueeze(y2,dim=1)
            y = torch.cat([y1,y2],dim=1)
            y,_ = torch.max(y,dim=1)
        elif self.fusion_type == 2:
            y = y1*y2
        elif self.fusion_type == 3:
            y = y1+y2
        elif self.fusion_type == 4:
            y1 = torch.unsqueeze(y1, dim=1)
            y2 = torch.unsqueeze(y2, dim=1)
            y = torch.cat([y1,y2],dim=1)
        elif self.fusion_type == 5:
            y = torch.cat([y1,y2],dim=-1)
        y = self.layer(y)
        y = self.soft(y)
        return y

if __name__ == "__main__":
    pass
    # model1 = generate_Res_model(18,n_input_channels=1,n_classes=10)
    # model2 = generate_Res_model(10,n_input_channels=1,n_classes=10)
    # model = Decison_fusion(model1,model2,fusion_type=4)
    # t1 = torch.randn((64,1,4096))
    # t2 = torch.randn((64,1,512))
    # print(model(t1,t2).shape)



