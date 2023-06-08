import torch
import torch.nn as nn
# from Fusion_models.utils.resnet import generate_Res_model



class Mixer_fusion(nn.Module):

    def __init__(self,model):
        super(Mixer_fusion, self).__init__()
        self.model = model

        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(4096,2048)

    def forward(self,x1,x2):
        x1_ = self.fc1(x1)
        x2_ = self.fc2(x2)
        x1_ = torch.cat([x1,x1_],dim=-1)
        x2_ = torch.cat([x2, x2_], dim=-1)
        x = torch.cat([x1_,x2_],dim=-1)
        y = self.model(x)
        return y




if __name__ == "__main__":
    pass
    # model1 = generate_Res_model(34,n_input_channels=1,n_classes=10)
    # # model2 = generate_Res_model(10,n_input_channels=1,n_classes=10)
    # model = Mixer_fusion(model1)
    # t1 = torch.randn((64,1,4096))
    # t2 = torch.randn((64,1,512))
    # print(model(t1,t2).shape)



