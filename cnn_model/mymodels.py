import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

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
