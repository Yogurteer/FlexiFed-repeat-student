# repeat-Flexifed
2023-7月，FlexiFed联邦学习实验复现源码-更新中  
论文原文：[3543507.3583347](https://dl.acm.org/doi/10.1145/3543507.3583347)  
复现实验作者：周臻鹏  
HUST-iot2001  
[查看运行方式](#jump)
## 实验说明
本仓库复现了论文"FlexiFed: Personalized 
Federated Learning
for Edge Clients with Heterogeneous
Model Architectures"中Table1部分的实验结果
## 配置
### 硬件环境
- GPU RTX 4090 10G
- CPU 12核
- 40G内存
### 软件环境
- ubuntu==9.4.1 
- jupyter lab
- torch=1.11.0+cu113
- torchvision=0.12.0+cu113
- python==3.8
- h5py=3.9.0
- librosa=0.10.0.post2
- matplotlib=3.5.2
## 实验对象
### 数据集
- 图像-CIFAR-10
- 图像CINIC-10
- 音频-Speech_Commands
- 文本-Ag News
### 模型
- VGG Family
- ResNet Family
- CharCNN Family
- VDCNN Family
### FlexiFed框架
- Basic-Common
- Clustered-Common
- Max-Common
## 实验运行方式<span id="jump"></span>
参考仓库目录结构如下图,按步骤运行:
- 根据requirements.txt下载所需模块
- 下载所需数据集到本地
- 运行fed-main.py,通过运行目录名判断当前训练的数据集,模型和聚合方法
- 运行完成之后,得到table.txt用于绘图
- 将运行目录作为输入运行view_results/drawer.py,得到acc曲线图像
- 可以在三种策略均训练完成之后运行drawer_compare.py比较不同方法之间的效果差异
### 目录结构
FlexiFed-repeat-student</br>
│  README.md</br>
│  目录结构.txt</br>
│          
├─FlexiFed-main</br>
│  │  fed-main.py</br>
│  │  requirements.txt</br>
│  │  utils.py</br>
│  │  任务说明.txt</br>
│  │  
│  ├─datasets</br>
│  │      deldataset.py</br>
│  │      
│  ├─model_family</br>
│  │      CharCNN.py</br>
│  │      ResNet.py</br>
│  │      VDCNN.py</br>
│  │      vgg.py</br>
│  │      
│  └─view_results</br>
│          drawer.py</br>
│          drawer_compare.py</br>
│          view_model.py</br>
│          
└─数据&报告</br>
&emsp;FlexiFed 实验复现结果数据汇总.xlsx</br>
&emsp;FlexiFed 实验复现结果曲线图汇总.pdf</br>
&emsp;实验结果曲线图汇总-8.1.pdf</br>
&emsp;生产实习汇报小结-7.3-7.13-周臻鹏.pdf</br>
&emsp;生产实习汇报小结-第二、三周-周臻鹏.pdf</br>
&emsp;生产实习汇报小结-第四周&总结-周臻鹏.docx</br>
&emsp;生产实习汇报小结-第四周&总结-周臻鹏.pdf</br>
&emsp;第三周实验复现结果曲线图汇总-共12项.pdf</br>