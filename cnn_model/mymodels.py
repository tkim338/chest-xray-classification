import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
	#IMPROVED
		self.hidden = nn.Linear(178, 30)
		self.hidden2 = nn.Linear(30,30)
		self.out = nn.Linear(30, 5)

	def forward(self, x):
		x = F.relu(self.hidden(x))
		i = 0
		while i<7:
			x = F.relu(self.hidden2(x))
			i = i+1
		x = self.out(x)
		return x
	#IMPROVED

class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
	#IMPROVED
		self.cnn_layer = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride =1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2,stride=2),
			# nn.Linear(8*87,300),
			# nn.Linear(300,120),
			nn.Conv2d(64, 6, 5, stride =1),
			# nn.BatchNorm2d(6),
			# nn.ReLU(inplace=True),
			# nn.Conv2d(6,16,5,stride=1),
			nn.BatchNorm2d(6),
			nn.ReLU(inplace=True)
		)
		self.linear = nn.Sequential(
			nn.Linear(165396, 32),
			nn.Linear(32,2)
		)

	def forward(self, x):
		# print(x.shape)
		x = x.float()
		x = self.cnn_layer(x)
		# print(x.shape)
		x = x.view(x.size(0), -1)
		x = self.linear(x)
		return x
	#IMPROVED


class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers = 2,batch_first=True, dropout =0.5)
		self.fc = nn.Linear(in_features=32, out_features=32)
		self.fc1 = nn.Linear(32,5)

	def forward(self, x):
		x, _ = self.rnn(x)
		x = F.relu(self.fc(x[:, -1, :]))
		x = self.fc1(x)
		return x


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		self.act = nn.Tanh()
		self.soft1 = nn.Softmax(dim=1)
		self.soft2 = nn.Softmax(dim=2)
		self.fc64 = nn.Linear(in_features=128, out_features=64)
		self.fc128 = nn.Linear(in_features=dim_input, out_features=128)
		self.mid = nn.Linear(in_features=64, out_features=64)
		self.fc32 = nn.Linear(in_features=dim_input, out_features=32)
		self.vrnn = nn.GRU(input_size = 32, hidden_size=16, num_layers = 2, batch_first=True, dropout=0.5)
		self.fcout = nn.Linear(in_features = 16, out_features=16)
		self.out = nn.Linear(in_features = 16, out_features=2)

	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		seqs, lengths = input_tuple
		# print(seqs.shape)
		# print(lengths)
		# x  = self.act(self.fc32(seqs))
		# print(seqs)
		# print(seqs.shape)
		# print(lengths)
		# print(seqs)
		# print(seqs.shape)
		x=self.fc32(seqs)
		# print(x)
		# print(x.shape)
		x = self.act(x)
		# print(x)
		# print(x.shape)
		# x= F.relu(self.fc64(x))
		# i = 0
		# while i<5:
		# 	x = F.relu(self.mid(x))
		# 	i = i+1
		# x = F.relu(self.fc32(x))
		# print(x.shape)
		# print(len(x))
		# print(x)
		if len(lengths)>1:
			x,h = self.vrnn(pack_padded_sequence(x,lengths, batch_first=True))
			# print("X MI CHAVO")
			# print(x)
			# print(x[0].shape)
			# print("H CUATE")
			# print(h)
			x,_ = pad_packed_sequence(x,batch_first=False, total_length=lengths[0])
			# print(x[:,-1,:])
			# print(x[:,-1,:].shape)
			# print(x)
			# print(x.shape)
			x = x.reshape((x.shape[1],x.shape[0],x.shape[2]))
			vals = x[:,-1,:]
			lengths= lengths-1
			# print(vals.shape)
			# print(x.shape)

			for i in range(seqs.shape[0]-1):
				vals[i,:] = x[i,lengths[i],:]
			vals = F.relu(self.fcout(vals))

			seqs = self.soft1(self.out(vals))
		else:
			x,_ = self.vrnn(x)
			x = F.relu(self.fcout(x))
			seqs = self.soft2(self.out(x))
		# print(x.shape)
		# print(x[0].shape)

		return seqs
