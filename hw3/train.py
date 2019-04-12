from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.autograd import Variable
# from sklearn.model_selection import train_test_split
import torchvision.transforms as trns
from PIL import Image
import sys

class MyDataset(Dataset):
	def __init__(self, X,Y,transform):
		self.transform = transform
		self.data_X = np.array(X)
		self.data_y = np.array(Y)
		print(np.shape(self.data_X))
	def __len__(self):
		return np.shape(self.data_X)[0]
	
	def __getitem__(self, idx):
		label = self.data_y[idx]
		img = self.data_X[idx]
		if self.transform is not None:
			img = img.reshape((48,48))
			img = self.transform(Image.fromarray(np.uint8(img)))
			img = np.array(img)
			img = img.reshape((48*48,))
		return torch.tensor(img), torch.tensor(label)
class MyNet(nn.Module):
	def __init__(self):
		super(MyNet, self).__init__() # call parent __init__ function
		#1*48*48
		self.conv1 = nn.Sequential(
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
		)#64*24*24
		self.conv2 = nn.Sequential(
			nn.Conv2d(
				64,
				128,
				kernel_size = 3,
				stride = 1,
				padding = 1,
			),
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(128),
			nn.MaxPool2d(kernel_size = 2),
		)#128*12*12
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
		)#256*12*12
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
		)#512*6*6
		self.conv5 = nn.Sequential(
			nn.Conv2d(
				512,
				512,
				kernel_size = 3,
				stride = 1,
				padding = 1,
			),
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(512),

		)#512*6*6
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
		)#512*3*3
		self.fc = nn.Sequential(
			nn.Linear(512*3*3, 1024),
			nn.LeakyReLU(0.2),
			nn.BatchNorm1d(1024),
			nn.Dropout(0.5),

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

	df = pd.read_csv(inputfile,encoding = "ISO-8859-1")
	value = []
	index = []
	for row in df.iterrows():
		index.append(row[1][0])
		value.append(row[1][1].split())
	index = np.array(index,dtype = float)
	value = np.array(value,dtype = float)
	train_transform = trns.Compose([
					  trns.RandomAffine(30, translate=(0.2,0.2), scale=(0.8,1.2), shear=10, resample=False, fillcolor=0),
					  trns.RandomHorizontalFlip(),
				  ])


	# X_train, X_test, y_train, y_test = train_test_split(value, index, test_size=0.1, random_state=1234)
	# Traindataset = MyDataset(X_train,y_train,train_transform)
	Traindataset = MyDataset(value,index,train_transform)

	Traindataloader = DataLoader(Traindataset, batch_size=100, shuffle=True, num_workers=4)
	# Testdataset = MyDataset(X_test,y_test,None)
	# Testdataloader = DataLoader(Testdataset, batch_size=100, shuffle=True, num_workers=4)

	device = torch.device('cuda')
	model = MyNet()
	model.to(device)
	optimizer = Adam(model.parameters(), lr=0.0001)
	loss_fn = nn.CrossEntropyLoss()

	best = 0
	Accuracy = []
	Loss = []
	for epoch in range(200):
		train_loss = []
		train_acc = []
		# model.train()
		for _, (img, target) in enumerate(Traindataloader):
			img_cuda = img.to(device, dtype=torch.float)
			target_cuda = target.to(device, dtype=torch.long)
			optimizer.zero_grad()
			output = model(img_cuda)
			loss = loss_fn(output, target_cuda)
			loss.backward()
			optimizer.step()
			predict = torch.max(output, 1)[1]
			acc = np.mean((target_cuda == predict).cpu().numpy())
			train_acc.append(acc)
			train_loss.append(loss.item())
		print("TrainEpoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))

		# train_loss = []
		# train_acc = []
		# model.eval()
		# for _, (img, target) in enumerate(Testdataloader):
			# img_cuda = img.to(device, dtype=torch.float)
			# target_cuda = target.to(device, dtype=torch.long)
			# output = model(img_cuda)
			# predict = torch.max(output, 1)[1]
			# loss = loss_fn(output, target_cuda)
			# acc = np.mean((target_cuda == predict).cpu().numpy())
			
			# train_acc.append(acc)
			# train_loss.append(loss.item())
		# if np.mean(train_acc) > best:
		# 	with open('acc.txt','w') as f:
		# 		f.write(str(epoch+1)+'\t'+str(train_acc)+'\n')
		# 	torch.save({'model':model.state_dict()},"aug_best_kernal3-epoch600.th")
			# best = np.mean(train_acc)
		# 	print ('Model Saved!')
		# Loss.append(np.mean(train_loss))
		# Accuracy.append(np.mean(train_acc))
		# print("TestEpoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
	# print(dataset.__len__())
	torch.save({'model':model.state_dict()},"output.th")
	# model.eval()
	# df2 = pd.read_csv("test.csv",encoding = "ISO-8859-1")
	# value = []
	# index = []
	# for i,row in enumerate(df2.iterrows()):
	# 	index.append(row[1][0])
	# 	img_cuda =  torch.tensor(np.array(row[1][1].split(),dtype = float)).to(device, dtype=torch.float)
	# 	output = model(img_cuda)
	# 	predict = torch.max(output, 1)[1].cpu().numpy()
	# 	value.append(predict)
	# value = np.array(value)
	# ans_df = pd.DataFrame({'id':index,'label':value.squeeze()})
	# ans_df.to_csv('aug_600-kernal3.csv',index = False)

