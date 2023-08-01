from itertools import groupby

from numpy import *
import matplotlib.pyplot as plt  # 导入画图工具库

"""
在下面编辑导入和导出目录,其他信息自动生成
"""
dad_dir = "/root/autodl-tmp/0725zzp00rtx3080/0731-max-new_speech-vgg-1-epoch501"
sdir = dad_dir.split('/')
totaldata = sdir[-1].split('-')
nepoch = [''.join(list(g)) for k, g in groupby(totaldata[-1], key=lambda x: x.isdigit())]
epochs = int(nepoch[1])
print("date:{}\nstrategy:{}\ndataset:{}\nmodelname:{}\ntimes:{}\nepochs:{}".format(
    totaldata[0],totaldata[1],totaldata[2],totaldata[3],totaldata[4],epochs
))
inputdir = "{}/{}".format(dad_dir, "table.txt")
outputdir = "{}/{}".format(dad_dir, "figure.png")
# strategys = ["Basic-Common", "Clustered-Common", "Max-Common"]
# models = ["VGG", "ResNet"]
# datanames = ["CIFAR-10", "CINIC-10", "Speech Commands"]


dataname = totaldata[2]
strategy = totaldata[1]
model = totaldata[3]
xname = "epochs"

with open(inputdir) as f:
    data = f.readline()
data = eval(data)
xlength = len(data[0])
# 求取最终acc结果
for i in range(0, 8, 2):
    a = data[i][epochs-100:epochs]
    b = data[i+1][epochs-100:epochs]
    c = (mean(a)+mean(b))*100/2
    n = int((i+2)/2)
    # print("model version{} final average mean_acc:{}".format(n, round(c, 1)))
    d = max(data[i])*100
    print("model version{} final average max_acc:{}".format(n, round(d,1)))
x1 = range(xlength)
"""
final acc 计算方法：对最后100次acc求均值
"""
plt.rcParams['font.size'] = 10  # 设置字体的大小为10
plt.rcParams['axes.unicode_minus'] = False  # 显示正、负的问题
colors = ['aquamarine', 'dodgerblue', 'pink', 'sandybrown', 'm', 'y', 'k', 'w']
labels = ["VGG-11", "VGG-13", "VGG-16", "VGG-19"]
fsize = 8
i = 0
# 第一个子图
ax1 = plt.subplot(221)
ax1.plot(x1, data[0], color=colors[i], label=labels[i])
ax1.set_title('{}11 {} {}'.format(model, strategy, dataname), fontsize=fsize,
              color='black')  # 为子图添加标题，fontproperties是设置标题的字体，fontsize是设置标题字体的大小，color是设置标题字体的颜色
ax1.set_xlabel(xname)  # 为x轴添加标签
ax1.set_ylabel('Accuracy')  # 为y轴添加标签
ax1.legend(loc='lower right')  # 设置图表图例在左上角
ax1.grid(True)  # 绘制网格
i += 1
# 第二个子图
ax1 = plt.subplot(222)
ax1.plot(x1, data[2], color=colors[i], label=labels[i])
ax1.set_title('{}13 {} {}'.format(model, strategy, dataname), fontsize=fsize,
              color='black')  # 为子图添加标题，fontproperties是设置标题的字体，fontsize是设置标题字体的大小，color是设置标题字体的颜色
ax1.set_xlabel(xname)  # 为x轴添加标签
ax1.set_ylabel('Accuracy')  # 为y轴添加标签
ax1.legend(loc='lower right')  # 设置图表图例在左上角
ax1.grid(True)  # 绘制网格
i += 1
# 第三个子图
ax1 = plt.subplot(223)
ax1.plot(x1, data[4], color=colors[i], label=labels[i])
ax1.set_title('{}16 {} {}'.format(model, strategy, dataname), fontsize=fsize,
              color='black')  # 为子图添加标题，fontproperties是设置标题的字体，fontsize是设置标题字体的大小，color是设置标题字体的颜色
ax1.set_xlabel(xname)  # 为x轴添加标签
ax1.set_ylabel('Accuracy')  # 为y轴添加标签
ax1.legend(loc='lower right')  # 设置图表图例在左上角
ax1.grid(True)  # 绘制网格
i += 1
# 第四个子图
ax1 = plt.subplot(224)
ax1.plot(x1, data[6], color=colors[i], label=labels[i])
ax1.set_title('{}19 {} {}'.format(model, strategy, dataname), fontsize=fsize,
              color='black')  # 为子图添加标题，fontproperties是设置标题的字体，fontsize是设置标题字体的大小，color是设置标题字体的颜色
ax1.set_xlabel(xname)  # 为x轴添加标签
ax1.set_ylabel('Accuracy')  # 为y轴添加标签
ax1.legend(loc='lower right')  # 设置图表图例在左上角
ax1.grid(True)  # 绘制网格
i += 1
plt.tight_layout()  # 自动调整各子图间距
plt.savefig(outputdir)
plt.show()
print("图片存放在:{}".format(outputdir))

# plt.hist(y, range=(-1, 1), bins=200, density=True)  # 绘制直方图关键操作
# plt.grid(alpha=1, linestyle='-.')  # 网格线，更好看
# plt.xlabel('num')
# plt.ylabel('fre')
# plt.title('hist of audio 7-31')
# plt.show()
