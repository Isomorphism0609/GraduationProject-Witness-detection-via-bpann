# Generate Entanglement Witness by SPI Algorithm
import time
import csv
import numpy as np
from numpy import sqrt, inner, kron, eye, outer, trace as tr
from functools import reduce
from numpy.linalg import norm
from random import choice
from random import choices


def partial_trace(mat: np.ndarray, param):
	n = int(sqrt(mat.shape[0]))  # partial trace矩阵的阶数, 是mat的阶数的1/2
	basis = np.eye(n)  # 标准正交基 e1,e2,...
	if param == 1:
		return np.sum(
			[mat[i * n + k, j * n + l] * np.outer(basis[k], basis[l]) * np.inner(basis[i], basis[j])
			 for i in range(n) for j in range(n) for k in range(n) for l in range(n)],
			axis=0)
	elif param == 2:
		return np.sum(
			[mat[i * n + k, j * n + l] * np.outer(basis[i], basis[j]) * np.inner(basis[l], basis[k])
			 for i in range(n) for j in range(n) for k in range(n) for l in range(n)],
			axis=0)
	else:
		raise ValueError

def calculate_b(a1p, psi):
	n = len(a1p)
	basis = np.eye(n)
	b = np.zeros(a1p.shape)
	for i in range(n):
		for j in range(n):
			b += psi[i*n+j] * inner(a1p, basis[i]) * basis[j]
	return b

def SPI(L, N, dim=2):
	"""

	:param L: np-matrix ,shape=(dim**2, dim**2)
	:param N: int, 本题取2
	:param dim: int, 本题取4
	:return:
	"""
	n = dim  # 基向量 a_1 的维数, 本题中为4
	basis = np.eye(n)  # 标准正交基 e1,e2,e3,e4
	a_list = choices(basis, k=N)  # 4 选 2, start vectors  (双体系统)
	iters = 45  # 迭代次数
	for i in range(iters):
		a_otimes = reduce(np.kron, a_list)  # a1 \otimes a2
		psi = L.dot(a_otimes)
		if N != 1:
			Lp = partial_trace(np.outer(psi, psi), param = N)
			a_list[:-1] = SPI(Lp+np.eye(len(Lp)), N-1, dim)  # 修改 a_1, ..., a_{N-1}
			b = calculate_b(a_list[0], psi)
			psi = b
		a_N = psi / sqrt(np.inner(psi, psi))
		a_list[-1] = a_N  # 修改 a_N


	return a_list


def get_witness(n=4):
	# W = gI-L
	M = 2*np.random.rand(n**2, n**2)-1   # [-1,1]上的随机矩阵
	L = (eye(n**2) + M.dot(M.T))/tr(eye(n**2) + M.dot(M.T))
	a_list_optim = SPI(L,N=2,dim=n)
	a1oa2 = reduce(kron, a_list_optim)  # a1 \otimes a2
	gmax = inner(a1oa2, L.dot(a1oa2))
	return gmax * eye(n**2) - L

	# 生成witnesses
def generate_Ws(num):  # num: 生成数量
	n = 4  # 4**2=16
	Ws = []
	for i in range(num):
		Ws.append(get_witness(n))
		print(i)
	return np.array(Ws)


	# 生成非witness
def generate_non_Ws(num):
	n = 4
	non_W = []
	while True:
		if len(non_W) == num:
			return np.array(non_W)
		M = 2*np.random.rand(n**2, n**2)-1  # 随机矩阵 [-1,1]
		H = M.dot(M.T)  # 厄米矩阵 H
		for i in range(100):
			a, b = np.random.rand(n), np.random.rand(n)
			a /= norm(a)  # 归一化
			b /= norm(b)  # 归一化
			psi = kron(a,b)
			rho = outer(psi, psi)
			if tr(H.dot(rho)<0):  # 如果W是非EW
				non_W.append(H)
				print(len(non_W))
				break

	# 生成非厄米矩阵
def generate_non_hermites(num):
	n = 4
	non_herm = []
	M = 2*np.random.rand(num, n**2, n**2)-1  # 随机矩阵 [-1,1]
	return M

	# while True:
	# 		# print(len(non_herm))
	# 		if len(non_herm) == num:
	# 			return np.array(non_herm)
	# 		M = 2*np.random.rand(n**2, n**2)-1  # 随机矩阵 [-1,1]
	# 		if not np.all(M == M.T):  # 若M!=M.T, 即M不为对称,是非厄米矩阵
	# 			non_herm.append(M)

def save_data(arr, file):
	arr = np.array([x.flatten() for x in arr])  # 转换成256维向量
	with open(file='数据\\'+file+'.csv', mode='w', encoding='utf-8', newline='') as f:
		write = csv.writer(f)
		write.writerows(arr)
		# count = 0
		# for row in arr:
		# 	write.writerow(row)
		#
		# 	print(count)
		# 	count+=1

def generate_val_and_test_data():
	semisize = 100  # Xsize个样本
	x1 = generate_Ws(semisize)
	x2 = generate_non_Ws(semisize)
	data = np.vstack((x1,x2))
	y1 = np.ones(semisize, dtype=int)
	y2 = np.zeros(semisize, dtype=int)
	y = np.hstack((y1,y2))
	# save_data(data, 'testdata200')
	save_data(y, 'validlabel200')



if __name__ == '__main__':
	# Xsize = 1000  # Xsize个样本
	# x1 = generate_Ws(Xsize)
	# x2 = generate_non_Ws(Xsize)
	# data = np.vstack((x1,x2))
	# y1 = np.ones(Xsize, dtype=int)
	# y2 = np.zeros(Xsize, dtype=int)
	# y = np.hstack((y1,y2))
	# save_data(data, 'data2000')
	# save_data(y, 'label2000')
	generate_val_and_test_data()

