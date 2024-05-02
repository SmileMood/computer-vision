import numpy as np

from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import visdom
import onnx

#########################################
#              设置网络参数
#########################################

viz = visdom.Visdom()

#下载数据集一级数据预处理操作
data_train = MNIST('./dataset',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./dataset',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=512, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)


#设置网络，损失函数以及优化器
net = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

#用于vis可视化
cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}
cur_epoch_win = None
cur_epoch_win_opts= {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Epoch Number',
    'ylabel': 'y',
    'width': 1200,
    'height': 600,
    'legend':['Loss','Accuracy']
}

#########################################
#              训练一轮
#########################################
def train(epoch):
    global cur_batch_win
    net.train()
    batch_list = []
    loss_list = []
    for i, (images, labels) in enumerate(data_train_loader):
        #梯度归零
        optimizer.zero_grad()
        #前向传播
        output = net(images)
        #计算损失函数
        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        # 更新可视化
        if viz.check_connection():
            cur_batch_win = viz.line(loss_list, batch_list,
                                     win=cur_batch_win, name='current_batch_loss',
                                     update=(None if cur_batch_win is None else 'replace'),
                                     opts=cur_batch_win_opts)
        #向后传播
        loss.backward()
        #更新网络参数
        optimizer.step()

    return loss_list

#########################################
#              测试
#########################################
def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        #计算当前批次的损失值
        avg_loss += criterion(output, labels).sum()
        #取出每个样本在输出output中预测概率最大的类别作为预测结果pred
        pred = output.detach().max(1)[1]
        #将模型预测正确的样本数累加到total_correct中
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))

    return  float(total_correct) / len(data_test)



#########################################
#              主函数
#            实现训练与测试
#########################################

def main():
    #定义全局变量
    global cur_epoch_win
    epoch_list = []
    loss_sum_list = []
    # 训练15轮
    for e in range(1, 16):

        #获取训练每轮得到的patch numeber损失列表
        loss = train(e)
        acc = test()

        loss_sum_list.append([np.mean(loss),acc])
        epoch_list.append(e)

        #可视化每轮的loss以及acc
        if viz.check_connection():
            cur_epoch_win = viz.line(torch.Tensor(loss_sum_list), torch.Tensor(epoch_list),
                                     win=cur_epoch_win,
                                     update=(None if cur_epoch_win is None else 'replace'),
                                     opts=cur_epoch_win_opts)

        # 创建虚拟输入，主要用于检查模型合法性
        dummy_input = torch.randn(1, 1, 32, 32, requires_grad=True)
        # 导出模型为ONNX格式
        torch.onnx.export(net, dummy_input, "lenet.onnx")
        # 加载导出的ONNX模型并检查模型合法性
        onnx_model = onnx.load("lenet.onnx")
        onnx.checker.check_model(onnx_model)




if __name__ == '__main__':
    main()
