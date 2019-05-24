import pandas as pd
import numpy as np
import sys
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.autograd import Variable
import torchvision.transforms as trns
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
class autoencoder(nn.Module):
	def __init__(self):
		super(autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 32, 3, stride=1, padding=1),  # b, 16, 10, 10
			nn.ReLU(),
			nn.BatchNorm2d(32),
			nn.MaxPool2d(kernel_size = 2),
			nn.Conv2d(32, 64, 3, stride=1, padding=1),  # b, 8, 3, 3
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(kernel_size = 2),
			nn.Conv2d(64 , 128, 3, stride=1, padding=1),  # b, 8, 3, 3
			nn.ReLU(),
			nn.BatchNorm2d(128),
			nn.MaxPool2d(kernel_size = 2),
		)
		self.decoder = nn.Sequential(
			nn.Conv2d(128 , 128, 3, stride=1, padding=1),  # b, 16, 5, 5
			nn.ReLU(),
			nn.BatchNorm2d(128),

			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(128 , 64, 3, stride=1, padding=1),  # b, 16, 5, 5
			nn.ReLU(),
			nn.BatchNorm2d(64),

			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(64 ,32, 3, stride=1, padding=1),  # b, 8, 15, 15
			nn.ReLU(),
			nn.BatchNorm2d(32),
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(32, 3, 3, stride=1, padding=1),  # b, 1, 28, 28
			nn.Tanh()
		)
		self.fc1 = nn.Sequential(
			nn.Linear(128*4*4, 256),
			nn.ReLU(),
			nn.BatchNorm1d(256),
			nn.Linear(256, 96),
			nn.ReLU(),
			nn.BatchNorm1d(96),
			
		)
		self.fc2 = nn.Sequential(
			nn.Linear(96, 256),
			nn.ReLU(),
			nn.BatchNorm1d(256),
			nn.Linear(256 , 128*4*4),
			nn.ReLU(),
			nn.BatchNorm1d(128*4*4),
		)

	def forward(self, x):
		x = x.view(-1,3,32,32)
		x = self.encoder(x)
		x = x.view(-1,128*4*4)
		code = self.fc1(x)
		x = self.fc2(code) 
		x = x.view(-1,128,4,4)
		x = self.decoder(x)
		return x.view(-1,3,32,32),code
if __name__ == '__main__':
	imagepath  = sys.argv[1]
	testpath = sys.argv[2]
	outputpath = sys.argv[3]

	device = torch.device('cuda')
	model = autoencoder()
	model.to(device)
	model.load_state_dict(torch.load("autoencoder-linearto96-2dense-bn-3channel.th")['model'])
	model.eval()
	print("eval")

	allcode = np.zeros((40000,96))
	for i in range(40000):
		im = Image.open(imagepath+str(i+1).zfill(6)+".jpg")
		im = np.array(im).transpose((2,0,1))
		im = (im - 127.5)/127.5
		im = torch.tensor(im)
		img_cuda = im.to(device, dtype=torch.float)
		output,code = model(img_cuda)
		code = code.cpu().detach().numpy().flatten()
		allcode[i] = code
		if i%1000==0:
			print(i)
	print("pca")
	pca = PCA(n_components='mle', copy=False, whiten=True, svd_solver='full')

	allcode = pca.fit_transform(allcode)
	print(allcode.shape)
	print("kmean")
	kmeans = KMeans(n_clusters=2, random_state=9487,n_jobs=4).fit(allcode)



	df = pd.read_csv(testpath,encoding = "ISO-8859-1")
	print(df.iloc[1][1])
	print(df.iloc[1][2])
	l = []
	for i in range(1000000):
		if kmeans.labels_[df.iloc[i][1]-1] == kmeans.labels_[df.iloc[i][2]-1]:
			a = 1
		else:
			a = 0
		l.append(a)
	value = np.array(l)
	ans_df = pd.DataFrame({'id':range(1000000),'label':value})
	ans_df.to_csv(outputpath,index = False)
