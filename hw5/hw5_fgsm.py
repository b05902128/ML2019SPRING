import numpy as np
import torch 
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable 
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169
import sys

inputfile = sys.argv[1]
outputfile = sys.argv[2]

model = resnet50(pretrained = True)
device = torch.device('cuda')
model.to(device)
model.eval()
loss_fn = nn.CrossEntropyLoss()
epsilon = 0.06


label = np.load("./label.npy")

mean = np.array([0.485, 0.456, 0.406],dtype = float)
std  = np.array([0.229, 0.224, 0.225],dtype = float)
count = 0
norm = []
for i in range(200):

	origin = Image.open(inputfile+"/"+str(i).zfill(3)+".png")
	origin = np.array(origin) 
	image = origin / 255
	image = (image - mean) / std

	image = np.transpose(image, (2, 0, 1))
	image = torch.tensor(image)
	image = image.unsqueeze(0)
	image = image.to(device, dtype=torch.float)
	image.requires_grad = True
	# set gradients to zero
	zero_gradients(image)
	output = model(image)
	ans1 = np.argmax(output.detach().cpu().numpy())




	target = label[i]
	target = torch.tensor(target)
	target = target.unsqueeze(0)
	target = target.to(device, dtype=torch.long)

	loss = -loss_fn(output, target)
	loss.backward() 

	# add epsilon to image
	image = image - epsilon * image.grad.sign_()
	image = image.squeeze()
	image = image.detach().cpu().numpy()



	#form new image
	image2 = np.transpose(image, (1, 2, 0))
	image2 = image2 * std + mean
	image2 = image2 * 255
	image2 = np.clip(image2,0,255)

	sub = np.abs(np.array(image2,dtype = int) - origin)
	max = np.max(sub)
	norm.append(max)




	image2 = np.array(image2,dtype = np.uint8)
	im = Image.fromarray(image2).convert('RGB')
	im.save(outputfile+"/"+str(i).zfill(3)+".png")




	#test new image
	image = image2 / 255
	image = (image - mean) / std
	image = np.transpose(image, (2, 0, 1))
	image = torch.tensor(image)
	image = image.unsqueeze(0)
	image = image.to(device, dtype=torch.float)
	output = model(image)
	ans2 = np.argmax(output.detach().cpu().numpy())



	if ans1 == ans2:
		count+=1
		print(i)
print(count)
norm = np.array(norm,dtype = float)
print(np.sum(norm)/200)
# print(np.argmax(output.detach().cpu().numpy()))
