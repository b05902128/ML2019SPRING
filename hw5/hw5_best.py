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
epsilon = 0.015
mean = np.array([0.485, 0.456, 0.406],dtype = float)
std  = np.array([0.229, 0.224, 0.225],dtype = float)
count = 0
norm = np.zeros((200,))
attacked = np.zeros((200,))


for i in range(200):
	temp = Image.open(inputfile+"/"+str(i).zfill(3)+".png")
	temp.save(outputfile+"/"+str(i).zfill(3)+".png")
label = np.load("./label.npy")


for j in range(15):
	print("round"+str(j+1))
	for i in range(200):
		if attacked[i] == 1:
			continue
		origin = Image.open(outputfile+"/"+str(i).zfill(3)+".png")
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
		# print(np.argmax(output.detach().cpu().numpy()))


		target = label[i]
		target = torch.tensor(target)
		target = target.unsqueeze(0)
		target = target.to(device, dtype=torch.long)

		loss =  - loss_fn(output, target)
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



		#test new image 
		image2 = np.array(image2,dtype = np.uint8)
		im = Image.fromarray(image2).convert('RGB')
		im.save(outputfile+"/"+str(i).zfill(3)+".png")





		image = image2 / 255
		image = (image - mean) / std
		image = np.transpose(image, (2, 0, 1))
		image = torch.tensor(image)
		image = image.unsqueeze(0)
		# print(image.shape)
		image = image.to(device, dtype=torch.float)
		output = model(image)
		ans2 = np.argmax(output.detach().cpu().numpy())
		# print(i,ans1,ans2)
		if ans1 == ans2:
			count+=1
			print(i)
		else:
			attacked[i] = 1
			norm[i] = j+1
# print(norm)
print(np.sum(attacked))
print(np.sum(norm)/200)
print(norm)
# print(np.argmax(output.detach().cpu().numpy()))

# do inverse_transform if you did some transformation