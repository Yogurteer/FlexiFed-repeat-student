from itertools import groupby

from numpy import *
import matplotlib.pyplot as plt  # 导入画图工具库


"""
在下面编辑导入和导出目录,其他信息自动生成
"""
basic = "/root/autodl-tmp/0725zzp03rtx3080/results_all/cifar10-resnet/0802-basic-cifar10-resnet-t-epoch80"
clustered = "/root/autodl-tmp/0725zzp03rtx3080/results_all/cifar10-resnet/0802-clustered-cifar10-resnet-1-epoch100"
max = "/root/autodl-tmp/0725zzp03rtx3080/results_all/cifar10-resnet/0802-max-cifar10-resnet-1-epoch100"
xlength = 80
in1 = "{}/{}".format(basic, "table.txt")
with open(in1) as f:
    data1 = f.readline()
data1 = eval(data1)[6][0:80]
in2 = "{}/{}".format(clustered, "table.txt")
with open(in2) as f:
    data2 = f.readline()
data2 = eval(data2)[6][0:80]
in3 = "{}/{}".format(max, "table.txt")
with open(in3) as f:
    data3 = f.readline()
data3 = eval(data3)[7][0:80]
print(len(data3))

outputdir = "/root/autodl-tmp/0725zzp03rtx3080/results_all/cifar10-resnet/{}".format("figure.png")

sdir = basic.split('/')
totaldata = sdir[-1].split('-')
nepoch = [''.join(list(g)) for k, g in groupby(totaldata[-1], key=lambda x: x.isdigit())]
epochs = int(nepoch[1])
print("date:{}\nstrategy:{}\ndataset:{}\nmodelname:{}\ntimes:{}\nepochs:{}".format(
    totaldata[0],totaldata[1],totaldata[2],totaldata[3],totaldata[4],epochs
))
dataname = totaldata[2]
strategy = totaldata[1]
model = totaldata[3]
plt.rcParams['font.size'] = 10  # 设置字体的大小为10
plt.rcParams['axes.unicode_minus'] = False  # 显示正、负的问题
colors = ['yellow', 'green', 'blue', 'sandybrown', 'm', 'y', 'k', 'w']
labels = {'vgg': ["VGG-11", "VGG-13", "VGG-16", "VGG-19"],
          'resnet': ["ResNet20", "ResNet32", "ResNet44", "ResNet56"],
          'charcnn': ["CharCNN3", "CharCNN4", "CharCNN5", "CharCNN6"],
          'vdcnn': ["VDCNN9", "VDCNN17", "VDCNN29", "VDCNN49"]}
fsize = 12
i = 0
x = range(80)
xname = "Communication Rounds"
# 第一个子图
# ax1 = plt.subplot(221)
# ax1.plot(x, data1, color=colors[i], label="Basic-Common")
# ax1.set_title('{} {} basic'.format(model, dataname), fontsize=fsize,
#               color='black')  # 为子图添加标题，fontproperties是设置标题的字体，fontsize是设置标题字体的大小，color是设置标题字体的颜色
# ax1.set_xlabel(xname)  # 为x轴添加标签
# ax1.set_ylabel('Accuracy')  # 为y轴添加标签
# ax1.legend(loc='lower right')  # 设置图表图例在左上角
# ax1.grid(True)  # 绘制网格
# i += 1
# # 第二个子图
# ax1 = plt.subplot(222)
# ax1.plot(x, data2, color=colors[i], label="Clustered-Common")
# ax1.set_title('{} {} clustered'.format(model, dataname), fontsize=fsize,
#               color='black')  # 为子图添加标题，fontproperties是设置标题的字体，fontsize是设置标题字体的大小，color是设置标题字体的颜色
# ax1.set_xlabel(xname)  # 为x轴添加标签
# ax1.set_ylabel('Accuracy')  # 为y轴添加标签
# ax1.legend(loc='lower right')  # 设置图表图例在左上角
# ax1.grid(True)  # 绘制网格
# i += 1
# # 第三个子图
# ax1 = plt.subplot(223)
# ax1.plot(x, data3, color=colors[i], label="Max-Common")
# ax1.set_title('{} {} Max'.format(model, dataname), fontsize=fsize,
#               color='black')  # 为子图添加标题，fontproperties是设置标题的字体，fontsize是设置标题字体的大小，color是设置标题字体的颜色
# ax1.set_xlabel(xname)  # 为x轴添加标签
# ax1.set_ylabel('Accuracy')  # 为y轴添加标签
# ax1.legend(loc='lower right')  # 设置图表图例在左上角
# ax1.grid(True)  # 绘制网格
# i += 1
plt.title('{} {} '.format(model, dataname), fontsize=fsize,
              color='black')
plt.plot(x, data1, color='skyblue', label='Basic-Common')
plt.plot(x, data2, color='forestgreen', label='Clustered-Common')
plt.plot(x, data3, color='gold', label='Max-Common')
plt.legend(loc='lower right')  # 显示图例
plt.xlabel(xname)
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()  # 自动调整各子图间距
plt.savefig(outputdir)
plt.show()
print("图片存放在:{}".format(outputdir))


