from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.autograd import Variable
import torchvision.transforms as trns
from PIL import Image
import matplotlib.pyplot as plt
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
def compute_saliency_maps(x, y, model):
	model.eval()
	x.requires_grad_()
	y_pred = model(x)
	loss_func = torch.nn.CrossEntropyLoss()
	loss = loss_func(y_pred, y.cuda())
	loss.backward()
	saliency = x.grad.abs().squeeze().data
	return saliency
def show_saliency_maps(x, y, model):
	x_org = x.detach().cpu().numpy()
	# Compute saliency maps for images in X
	saliency = compute_saliency_maps(x, y, model)
	# Convert the saliency map from Torch Tensor to numpy array and show images
	# and saliency maps together.
	saliency = saliency.detach().cpu().numpy()
	num_pics = x_org.shape[0]
	for i in range(num_pics):
		# You need to save as the correct fig names
		plt.imsave(outputpath+'fig1_origin_'+ str(i)+".jpg", x_org[i].reshape((48,48)), cmap=plt.cm.gray)
		plt.imsave(outputpath+'fig1_'+ str(i)+".jpg", saliency[i].reshape((48,48)), cmap=plt.cm.jet)
if __name__ == '__main__':
	inputpath  = sys.argv[1]
	outputpath = sys.argv[2]
	device = torch.device('cuda')
	model = MyNet()
	model.to(device)
	model.load_state_dict(torch.load("aug_best_kernal3-epoch600.th")['model'])
	df = pd.read_csv(inputpath,encoding = "ISO-8859-1")
	value = []
	index = []
	count = 0
	for row in df.iterrows():
		index.append(row[1][0])
		value.append(row[1][1].split())
		count+=1
		if count == 400:
			break
	x_train = np.array(value,dtype = float)
	x_label = np.array(index,dtype = float)
	x = np.empty((7,48*48))
	y = np.empty((7))

	for i,j in enumerate([65,299,101,7,20,51,37]):
		x[i] = x_train[j]
		y[i] = x_label[j]

	Traindataset = MyDataset(x,y,None)
	Traindataloader = DataLoader(Traindataset, batch_size=7, shuffle=False, num_workers=4)
	optimizer = Adam(model.parameters(), lr=0.0001)
	for _, (img, target) in enumerate(Traindataloader):
		img_cuda = img.to(device, dtype=torch.float)
		target_cuda = target.to(device, dtype=torch.long)
		show_saliency_maps(img_cuda, target_cuda, model)


# using the first ten images for example
	