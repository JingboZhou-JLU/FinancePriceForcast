# 导入库
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import pandas as pd
from PIL import Image
from PIL import ImageTk
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from torch import nn
from torch.autograd import Variable

# 修改图片大小
def get_img(filename, width, height):
    im = Image.open(filename).resize((width, height))
    im = ImageTk.PhotoImage(im)
    return im


# 获得数据路径
def click_button_3():
    global file_path
    file_path = askopenfilename(title='Please choose a file',initialdir='/', filetypes=[('Python source file', '*.csv')])
    if file_path== "":
        tk.messagebox.showinfo(title="数据导入", message="未导入")
    else:
        tk.messagebox.showinfo(title="数据导入", message="导入完成")
    return

def click_button_4():
    global file_path, _values
    data = {"date": [], "value": []}
    _data = pd.read_csv(file_path)

    date, _values = list(_data.iloc[:, 0]), list(_data.iloc[:, 1])
    for day, value in zip(date, _values):
        data['date'].append(day)
        data['value'].append(value)
    x = np.array(range(0, len(data['date'])))
    y = np.array(data['value'])
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(x, y)
    plt.xlabel("day")
    plt.ylabel("value")
    plt.show()
    return


def click_button_5():
    global _window
    _window.destroy()
    root = tk.Tk()
    root.title("用户手册")
    img_open = Image.open("pict/help_people.png")
    img_png = ImageTk.PhotoImage(img_open)
    label_img = tk.Label(root, image=img_png)
    label_img.pack()
    root.mainloop()
    return


# 获得超参
def get():
    global  entry1,entry2,entry3,entry4,day_window,day_after,hidden_layers,num_epochs
    day_window = int(entry1.get())
    day_after =  int(entry2.get())
    hidden_layers = int(entry3.get())
    num_epochs = int(entry4.get())
    return


# 归一化
def Normal_(List):
    max_value = np.max(List)
    min_value = np.min(List)
    scalar = max_value - min_value
    List = list(map(lambda x: x / scalar, List))
    return List, scalar  # 返回归一化 和 标量


def split_dataset(dataset,day_window, day_after):
    features = []
    labels = []
    for i in range(0, len(dataset)-day_after-day_window+1):
        features.append(copy.deepcopy(dataset[i:i+day_window]))
        labels.append(copy.deepcopy(dataset[i+day_window:i+day_window+day_after]))
    return features, labels


# 3.定义模型
class RNN(nn.Module):
    def __init__(self,day_window,day_after,hidden_layers):
        super(RNN, self).__init__()  # 面向对象中的继承
        self.lstm = nn.LSTM(day_window, hidden_layers, 2)  # day_window输入特征长度
        self.out = nn.Linear(hidden_layers, day_after)  # day_after 输出标签长度

    def forward(self, x):
        x1, _ = self.lstm(x)
        a, b, c = x1.shape
        out = self.out(x1.view(-1, c))  # 因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
        out1 = out.view(a, b, -1)  # 因为是循环神经网络，最后的时候要把二维的out调整成三维数据，下一次循环使用
        return out1

def train():
    global _values, day_window, day_after, num_epochs, hidden_layers
    global choose_window
    global first_x,first_y, second_x, second_y

    features, labels = split_dataset(_values, day_window, day_after)

    normal_features = []
    normal_labels = []
    scalar = []
    for i in range(len(features)):
        f_list, sc = Normal_(copy.deepcopy(features[i]))
        l_list = list(copy.deepcopy(labels[i] / sc))
        normal_features.append(f_list)
        normal_labels.append(l_list)
        scalar.append(sc)

    """
        features: 特征   type:tensor
        labels:   标签   type:tensor
        """
    normal_features, normal_labels = torch.tensor(normal_features), torch.tensor(normal_labels)
    normal_features = normal_features.reshape(-1, 1, day_window)
    normal_labels = normal_labels.reshape(-1, 1, day_after)

    rnn = RNN(day_window, day_after, hidden_layers)
    # 4.定义损失函数与优化算法
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)

    use_gpu = torch.cuda.is_available()
    if (use_gpu):
        rnn = rnn.cuda()
        loss_func = loss_func.cuda()
        normal_features, normal_labels = normal_features.cuda(), normal_labels.cuda()

    # 6.训练
    for i in range(num_epochs):
        var_x = Variable(normal_features).type(torch.FloatTensor)
        var_y = Variable(normal_labels).type(torch.FloatTensor)
        var_x = var_x.cuda()
        var_y = var_y.cuda()
        out = rnn(var_x)
        loss = loss_func(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if ((i + 1) % 50) == 0:
        #     print('Epoch:{}, Loss:{:.20f}'.format(i + 1, loss.item()))

    pred_x = Variable(normal_features).type(torch.FloatTensor)
    pred_x = var_x.cuda()
    pred_y = rnn(pred_x)

    for i in range(len(scalar)):
        pred_y[i] = pred_y[i] * scalar[i]

    pred_y = pred_y.reshape(len(features), day_after)
    pred_y = pred_y.cpu()
    pred_y = pred_y.detach().numpy()
    pred_y = pred_y.tolist()

    plot_pred = []
    for y_list in pred_y:
        sum = 0
        average = 0
        for num in y_list:
            sum = sum + num
        average = sum / day_after
        plot_pred.append(average)

    plot_labels = []
    for y_list in labels:
        sum = 0
        average = 0
        for num in y_list:
            sum = sum + num
        average = sum / day_after
        plot_labels.append(average)


    first_x = plot_pred
    first_y = plot_labels

    ##################### 预测过程
    f_list, sc = Normal_(copy.deepcopy(_values[-day_window:]))
    normal_features = []
    normal_features.append(f_list)
    normal_features = torch.tensor(normal_features)
    normal_features = normal_features.reshape(-1, 1, day_window)
    normal_features
    pred_x = Variable(normal_features).type(torch.FloatTensor)
    pred_x = pred_x.cuda()
    pred_y = rnn(pred_x)

    pred_y = pred_y.reshape(day_after)
    pred_y = pred_y.cpu()
    pred_y = pred_y.detach().numpy()
    pred_y = pred_y.tolist()

    pred_o = []
    for i in pred_y:
        pred_o.append(i * sc / 2)

    x = [str(i) for i in range(0, day_after)]
    second_x = x
    second_y = pred_o
    #####################
    tk.messagebox.showinfo(title="训练", message="训练完成!")
    choose_window.destroy()
    return


def click_button_6():
    global entry1,entry2,entry3,entry4
    global choose_window
    choose_window = tk.Tk()
    choose_window.title('选择参数')
    choose_window.geometry('250x250')
    choose_window.resizable(False, False)

    tk.Label(choose_window, text='滑动窗口大小').grid(row=0, column=1)
    tk.Label(choose_window, text='预测窗口大小').grid(row=1, column=1)
    tk.Label(choose_window, text='模型层数   ').grid(row=2, column=1)
    tk.Label(choose_window, text='学习轮数   ').grid(row=3, column=1)
    entry1 = tk.Entry(choose_window)
    entry1.grid(row=0, column=2, padx=10, pady=5)
    entry2 = tk.Entry(choose_window)
    entry2.grid(row=1, column=2, padx=10, pady=5)
    entry3 = tk.Entry(choose_window)
    entry3.grid(row=2, column=2, padx=10, pady=5)
    entry4 = tk.Entry(choose_window)
    entry4.grid(row=3, column=2, padx=10, pady=5)

    tk.Button(choose_window, text="确认", width=10, command=get).grid(row=8, column=2, sticky='w', padx=10, pady=5)
    tk.Button(choose_window, text="开始训练", width=10, command=train).grid(row=10, column=2, sticky='w', padx=10, pady=5)

    choose_window.mainloop()
    return


def click_button_7():
    global first_x,first_y, second_x, second_y,day_after
    plt.figure(figsize=(15, 6), dpi=80)

    plt.plot(first_x, 'r', label='prediction')
    plt.plot(first_y, 'b', label='real')
    plt.legend(loc='best')
    plt.savefig('pict/trend.png')
    plt.show()

    plt.figure(figsize=(15, 6), dpi=80)
    x = [str(i) for i in range(0, day_after)]
    plt.plot(second_x, second_y, 'r')
    plt.savefig("pict/predict.png")
    plt.show()

    return

# 主界面
def click_button_1():
    global root_window,_window
    root_window.destroy()  # 需要删除上个界面
    _window = tk.Tk()
    _window.title('基于LSTM神经网络的金融产品价格预测平台')
    _window.geometry('1000x600+180+100')
    _window.resizable(False, False)

    canvas_root = tk.Canvas(_window, width=1000, height=600)
    im_root = get_img('pict/background1.png', 1000, 600)
    canvas_root.create_image(500, 300, image=im_root)
    canvas_root.pack()


    im = get_img('pict/data.png', 280, 150)
    button = tk.Button(_window, image=im, compound=tk.CENTER, width=280, height=150, command=click_button_3)
    button.place(x=90, y=100)


    im_2 = get_img('pict/dataanalyse.png', 280, 150)
    button_2 = tk.Button(_window, image=im_2, compound=tk.CENTER, width=280, height=150, command=click_button_4)
    button_2.place(x=620, y=100)



    im_3 = get_img('pict/help.jpg', 100, 55)
    button_3 = tk.Button(_window, image=im_3, compound=tk.CENTER, width=100, height=55, command=click_button_5)
    button_3.place(x=0, y=0)

    im_4 = get_img('pict/ai.jpg', 280, 150)
    button_4 = tk.Button(_window, image=im_4, compound=tk.CENTER, width=280, height=150, command=click_button_6)
    button_4.place(x=90, y=400)

    im_5 = get_img('pict/Decisions.jpg', 280, 150)
    button_5 = tk.Button(_window, image=im_5, compound=tk.CENTER, width=280, height=150, command=click_button_7)
    button_5.place(x=620, y=400)

    _window.mainloop()




# 全局变量
_window = ""
choose_window = ""
file_path = ""
entry1 = ""
entry2 = ""
entry3 = ""
entry1 = ""

#超参
day_window = 0
day_after = 0
hidden_layers = 0
num_epochs = 0
_values = 0


# 图片横纵坐标
first_x = 0
first_y = 0
second_x = 0
second_y = 0



root_window = tk.Tk()
root_window.title('基于LSTM神经网络的金融产品价格预测平台')
root_window.geometry('1000x600+180+100')
root_window.resizable(False, False)



canvas_root = tk.Canvas(root_window, width=1000, height=600)
im_root = get_img('pict/background.png', 1000, 600)
canvas_root.create_image(500, 300, image=im_root)
canvas_root.pack()




# 点击按钮时执行的函数
im = get_img('pict/start.png', 280, 100)
button = tk.Button(root_window, image= im, compound = tk.CENTER, width=280, height=100, command=click_button_1)
button.place(x=365,y=400)


root_window.mainloop()





