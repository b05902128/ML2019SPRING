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
		self.conv_drop = nn.Dropout(p = 0.2)
		self.drop = nn.Dropout(p = 0.5)
		self.conv1 = nn.Sequential(
			#1*48*48
			nn.Conv2d(
				1,
				64,
				kernel_size = 3,
				stride = 1,
				padding = 1,
			),
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(kernel_size = 2),
			# nn.Dropout2d(0.4),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(
				64,
				128,
				kernel_size = 3,
				stride = 1,
				padding = 1,
			),#42*42
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(128),
			nn.MaxPool2d(kernel_size = 2),
			# nn.Dropout2d(0.4),

			# self.conv_drop(),
		)#20*20
		self.conv3 = nn.Sequential(
			nn.Conv2d(
				128,
				256,
				kernel_size = 3,
				stride= 1,
				padding = 1,
			),
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(256),
			# nn.Dropout2d(0.4),

			# self.conv_drop(),
		)#9*9
		self.conv4 = nn.Sequential(
			nn.Conv2d(
				256,
				512,
				kernel_size = 3,
				stride= 1,
				padding = 1,
			),
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(kernel_size = 2),
			# nn.Dropout2d(0.4),

			# self.conv_drop(),
		)#9*9
		self.conv5 = nn.Sequential(
			nn.Conv2d(
				512,
				512,
				kernel_size = 3,
				stride = 1,
				padding = 1,
			),#10*10
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(512),
			# nn.Dropout2d(0.4),

			# self.conv_drop(),

		)#5*5
		self.conv6 = nn.Sequential(
			nn.Conv2d(
				512,
				512,
				kernel_size = 3,
				stride = 1,
				padding = 1,
			),#10*10
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(kernel_size = 2),
			# nn.Dropout2d(0.4),

			# self.conv_drop(),

		)#5*5
		self.fc = nn.Sequential(
			nn.Linear(512*3*3, 1024),
			nn.LeakyReLU(0.2),
			nn.BatchNorm1d(1024),
			nn.Dropout(0.5),

			# self.drop(),
			nn.Linear(1024, 512),
			nn.LeakyReLU(0.2),
			nn.BatchNorm1d(512),
			nn.Dropout(0.5),
			nn.Linear(512, 7),
		)
		self.output = nn.Softmax(dim=1)


	def forward(self, x):
		# You can modify your model connection whatever you like
		out = self.conv1(x.view(-1,1,48,48))
		# out = self.conv1(out)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		out = self.conv6(out)
		out = self.fc(out.view(-1,512*3*3))
		out = self.output(out)
		return out      
     

if __name__ == '__main__':
	inputfile = sys.argv[1]
	outputfile = sys.argv[2]
	
	device = torch.device('cuda')
	model1 = MyNet()
	model1.to(device)
	model1.load_state_dict(torch.load("aug_best_kernal3-epoch400.th")['model'])
	model1.eval()
	model2 = MyNet()
	model2.to(device)
	model2.load_state_dict(torch.load("aug_best_kernal3-epoch600.th")['model'])
	model2.eval()
	model3 = MyNet()
	model3.to(device)
	model3.load_state_dict(torch.load("aug_400-kernal3.th")['model'])
	model3.eval()
	df2 = pd.read_csv(inputfile,encoding = "ISO-8859-1")
	value = []
	index = []
	for i,row in enumerate(df2.iterrows()):
		index.append(row[1][0])
		img_cuda =  torch.tensor(np.array(row[1][1].split(),dtype = float)).to(device, dtype=torch.float)
		output1 = model1(img_cuda)
		predict1 = torch.max(output1, 1)[1].cpu().numpy()
		output2 = model2(img_cuda)
		predict2 = torch.max(output2, 1)[1].cpu().numpy()
		output3 = model3(img_cuda)
		predict3 = torch.max(output3, 1)[1].cpu().numpy()
		if predict3 == predict1:
			predict = predict3
		else:
			predict = predict2
		value.append(predict)
	value = np.array(value)
	ans_df = pd.DataFrame({'id':index,'label':value.squeeze()})
	ans_df.to_csv(outputfile,index = False)