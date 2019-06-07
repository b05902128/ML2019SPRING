from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.autograd import Variable
import torchvision.transforms as trns
import sys



class MyNet(nn.Module):
	def __init__(self):
		super(MyNet, self).__init__() # call parent __init__ function
		#1*48*48
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				1,
				32,
				kernel_size = 3,
				stride = 1,
				padding = 1,
			),
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(32),
			nn.MaxPool2d(kernel_size = 2),
		)#64*24*24
		self.conv2 = nn.Sequential(
			nn.Conv2d(
				32,
				48,
				kernel_size = 3,
				stride = 1,
				padding = 1,
			),
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(48),
			nn.MaxPool2d(kernel_size = 2),
		)#128*12*12

		self.conv4 = nn.Sequential(
			nn.Conv2d(
				48,
				72,
				kernel_size = 3,
				stride= 1,
				padding = 1,
			),
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(72),
			nn.MaxPool2d(kernel_size = 2),
		)#512*6*6
		self.conv6 = nn.Sequential(
			nn.Conv2d(
				72,
				72,
				kernel_size = 3,
				stride = 1,
				padding = 1,
			),#10*10
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(72),
			nn.MaxPool2d(kernel_size = 2),
		)#512*3*3
		self.fc = nn.Sequential(
			nn.Linear(72*3*3, 7),
		)
		self.output = nn.Softmax(dim=1)


	def forward(self, x):
		# You can modify your model connection whatever you like
		out = self.conv1(x.view(-1,1,48,48))
		out = self.conv2(out)
		out = self.conv4(out)
		out = self.conv6(out)
		out = self.fc(out.view(-1,72*3*3))
		out = self.output(out)
		return out        

if __name__ == '__main__':
	inputfile = sys.argv[1]
	outputfile = sys.argv[2]
	
	device = torch.device('cuda')
	model1 = MyNet()
	model1.to(device)
	model1.load_state_dict(torch.load("half-batch32.th")['model'])
	model1.eval()

	df2 = pd.read_csv(inputfile,encoding = "ISO-8859-1")
	value = []
	index = []
	for i,row in enumerate(df2.iterrows()):
		index.append(row[1][0])
		img_cuda =  torch.tensor(np.array(row[1][1].split(),dtype = float)).to(device, dtype=torch.float)
		output = model1(img_cuda)
		predict = torch.max(output, 1)[1].cpu().numpy()

		value.append(predict)
	value = np.array(value)
	ans_df = pd.DataFrame({'id':index,'label':value.squeeze()})
	ans_df.to_csv(outputfile,index = False)