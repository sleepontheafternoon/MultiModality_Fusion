### Description
**This project is study use only.**<br>
This project contains four types methods.Including early fusion, later fusion, mixed machine fusion, and model-level fusion.Here is providing some well-trained models.You can load the pth file to get a well-trained model without training.The model backbone is Resnet.You can change the depth of resnet. The arguments including 10,18,34 and 50 and so on.<br>
### 描述
**本项目仅供学习使用**<br>
本项目实现了四种模态融合的方法，包括早期融合，晚期融合，混合机融合以及模型级融合。这里提供训练好的模型权重文件，你可以不用训练，直接导入pth文件就能得到一个不错的模型。本项目模型的backbone是resnet，使用的卷积方法为一维卷积。你可以改变resent的深度，包括10，18，34和50等可选深度，也可以改变卷积方法完成诸如图片模态的融合任务。数据描述及其方法详见description.txt文件，可以借助该[博文](https://blog.csdn.net/weixin_43840280/article/details/118070317)辅助理解融合方法。
### 文件结构
1. Features文件夹包括相应的模态特征以及标签
2. Fusion_models文件夹包括训练文件以及模态融合方法
    - checkpoint文件夹包括模型的权重文件
    - log记录了每个文件的训练和测试情况
    - utils是需要使用到的模型及其相应改变
    - 该文件目录下的.py文件为训练文件
### 评价指标
本项目实现任务为多分类，使用macro-F1,macro-P,macro-R以及Acc

