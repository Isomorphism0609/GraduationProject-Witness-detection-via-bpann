from typing import List, Any

import numpy as np
import torch as tor
import torch
# from torch.autograd import Variable
import pylab as plt
import torch.nn.functional as F
from sklearn import manifold
import matplotlib
import time
import csv
from numerical_generation_526 import generate_Ws, generate_non_Ws, generate_non_hermites
import torch.utils.data as Data
from functools import reduce
from data_preparation import generate_W_matrices, generate_W_labels

torch.manual_seed(1)  # 初始化种子


def normalize(data):
	method = 'gaussuniform'
	if method == 'uniform':
		return (data - data.min(0)) / (data.max(0) - data.min(0))
	elif method == 'gauss':
		mu = np.mean(data, axis=0)
		std = np.std(data, axis=0)
		data1 = (data - mu) / np.sqrt(np.square(std) + 1e-5)
		return data1
	elif method == 'uniformgauss':
		data = (data - data.min(0)) / (data.max(0) - data.min(0))
		mu = np.mean(data, axis=0)
		std = np.std(data, axis=0)
		return (data - mu) / np.sqrt(np.square(std) + 1e-5)
	elif method == 'gaussuniform':
		mu = np.mean(data, axis=0)
		std = np.std(data, axis=0)
		data = (data - mu) / np.sqrt(np.square(std) + 1e-5)
		return (data - data.min(0)) / (data.max(0) - data.min(0) + 1e-5)


def get_x_y():
	x = np.loadtxt('数据\data2000.csv', delimiter=',')  # EW和nonEW都有
	y = np.loadtxt('数据\label2000.csv', delimiter=',').flatten()
	x = normalize(x)
	x = tor.from_numpy(x).type(torch.FloatTensor)
	y = tor.from_numpy(y).type(tor.LongTensor)
	return x, y

def get_val_and_test_data():
	valx = np.loadtxt(r'数据\validdata200.csv', delimiter=',')
	valy = np.loadtxt(r'数据\validlabel200.csv', delimiter=',').flatten()
	valx = normalize(valx)
	valx = tor.from_numpy(valx).type(torch.FloatTensor)
	valy = tor.from_numpy(valy).type(tor.LongTensor)
	testx = np.loadtxt(r'数据\testdata200.csv', delimiter=',')
	testy = np.loadtxt(r'数据\testlabel200.csv', delimiter=',').flatten()
	testx = normalize(testx)
	testx = tor.from_numpy(testx).type(torch.FloatTensor)
	testy = tor.from_numpy(testy).type(tor.LongTensor)
	return valx, valy, testx, testy


# ============== 超参数 ================
N = len(get_x_y()[1])
BATCH_SIZE = 64
ITERATIONS = int(N / BATCH_SIZE)


def train_loader():
	x, y = get_x_y()
	dataset = Data.TensorDataset(x, y)
	train_ds = Data.DataLoader(
		dataset=dataset,  # torch TensorDataset format
		batch_size=BATCH_SIZE,  # mini batch size
		shuffle=True,  # 要不要打乱数据 (打乱比较好)
		num_workers=2,  # 多线程来读数据
	)

	return train_ds

def val_loader():
	valx, valy = get_val_and_test_data()[0:2]
	dataset = Data.TensorDataset(valx, valy)
	val_loader = torch.utils.data.DataLoader(
		dataset=dataset,
		batch_size=32, shuffle=True)
	return val_loader

def _test_loader():
	testx, testy = get_val_and_test_data()[2:]
	dataset = Data.TensorDataset(testx, testy)
	test_loader = torch.utils.data.DataLoader(
		dataset=dataset,
		batch_size=32, shuffle=True)
	return test_loader


class Net(torch.nn.Module):  # 继承 torch 的 Module
	def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
		super(Net, self).__init__()  # 继承 __init__ 功能
		self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)  # layer1
		self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # layer2
		self.out = torch.nn.Linear(n_hidden2, n_output)  # 输出层线性输出

	def forward(self, x):
		# 正向传播输入值, 神经网络分析出输出值
		x = tor.tanh(self.hidden1(x))  # 激励函数(隐藏层的线性值)
		x = F.leaky_relu_(self.hidden2(x))  # 激励函数(隐藏层的线性值)
		x = tor.tanh(self.out(x))  # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
		return x


def _test_it(model, _test_loader):
	loss_func = torch.nn.CrossEntropyLoss()

	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in _test_loader:
			output = model(data)
			print(output, loss_func(output, target))
			print(loss_func(output, target).item())
			test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
			pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(_test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(_test_loader.dataset),
		100. * correct / len(_test_loader.dataset)))
	return correct / len(_test_loader.dataset)  # accuracy

	# 计算预测acc
def cal_acc(net, xb, yb):  #yb为真实值
	predy = torch.max(net(xb), 1)[1].data.numpy().squeeze()
	yb = yb.numpy()
	return np.sum(yb==predy)/len(yb)


def train_epoch(train_loader, net, loss_func, optimizer, epoch):
	train_loss = 0
	train_acc = 0
	num_correct = 0
	for i, (xb, yb) in enumerate(train_loader):
		optimizer.zero_grad()  # 清空上一步的残余更新参数值
		out = net(xb)  # 喂给 net 训练数据 x, 输出分析值
		loss = loss_func(out, yb)  # 计算两者的误差
		loss.backward()  # 误差反向传播, 计算参数更新值
		optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
		predy = out.argmax(dim=1)
		num_correct += torch.eq(predy, yb).sum().float().item()
		train_loss += loss.item()
		# print(f'epoch:{epoch}, step:{i}, size={len(yb)}, correct={num_correct}')
	train_loss /= len(train_loader)
	train_acc = num_correct / len(train_loader.dataset)
	# 返回本batch上的train loss 与 train acc \in R
	return train_loss, train_acc


def train(net, train_loader, valid_loader):
	#### 数据
	x, y = get_x_y()
	print(f'数据读取完毕, x.shape={x.shape}, y.shape={y.shape}')
	optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
	loss_func = torch.nn.CrossEntropyLoss()
	train_losses, train_acces, val_losses, val_acces = list(), list(), list(), list()
	EPOCHES = 20
	for epoch in range(EPOCHES):  # 训练
		train_loss, train_acc \
			= train_epoch(train_loader, net, loss_func, optimizer, epoch)
		net.eval()  # 评估模型
		with torch.no_grad():
			# pred_y = torch.max(out, 1)[1].data.numpy().squeeze()  # list
			# (torch.max(net(xb), yb) for xb, yb in train_loader)

			valid_loss = sum(loss_func(net(xb), yb).item() for xb, yb in valid_loader)/len(valid_loader)
			valid_acc = sum(cal_acc(net, xb, yb) for xb, yb in valid_loader)/len(valid_loader)
		print(f'epoch:{epoch}  train loss = {train_loss}, train acc = {train_acc}\n'
		      f'           val loss = {valid_loss}, val acc = {valid_acc}')

		train_losses.append(train_loss)
		train_acces.append(train_acc)
		val_losses.append(valid_loss)
		val_acces.append(valid_acc)
	return train_losses, train_acces, val_losses, val_acces

def _test(net, test_loader):
	loss_func = torch.nn.CrossEntropyLoss()
	correct = 0
	losses = 0
	net.eval()
	with torch.no_grad():
		for xb, yb in test_loader:
			out = net(xb)
			loss = loss_func(out, yb)
			losses += (loss.item())
			predy = out.argmax(dim=1)  # 找到概率最大的下标
			correct += tor.eq(predy, yb).sum().item()

	test_loss = losses / len(test_loader.dataset)
	test_acc = correct / len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))
	return test_loss, test_acc


def main():
	#### 数据
	x, y = get_x_y()
	train_ld = train_loader()
	valid_ld = val_loader()
	test_ld = _test_loader()

	#### 网络
	net = Net(n_feature=16 * 16,
	          n_hidden1=64, n_hidden2=64,
	          n_output=2)  # 几个类别就几个 output
	print(f'net 的结构: net = {net}')  # net 的结构
	print('开始训练……')
	tloss, tacc, vloss, vacc = \
		train(net, train_ld, valid_ld)
	testloss, testval = _test(net, test_ld)
	with open(r'生成文件\accloss060718.csv', 'w', newline='') as f:
		write = csv.writer(f)
		print(tloss)
		write.writerows([tloss, tacc, vloss, vacc])
	with open(r'生成文件\test result.csv', 'w', newline='') as f:
		write = csv.writer(f)
		write.writerow([testloss, testval])


if __name__ == '__main__':
	for i in range(1):
		main()
