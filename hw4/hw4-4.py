import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys
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
    inputpath  = sys.argv[1]
    outputpath = sys.argv[2]
    # Fully connected layer is not needed
    device = torch.device('cuda')
    model = MyNet()
    model.to(device)
    model.load_state_dict(torch.load("aug_best_kernal3-epoch600.th")['model'])
    df = pd.read_csv(inputpath,encoding = "ISO-8859-1")
    value = []
    index = []
    idx = 5
    count = 0
    for row in df.iterrows():
        index.append(row[1][0])
        value.append(row[1][1].split())
        count+=1
        if count == 20:
            break
    model.eval()

    x_train = np.array(value,dtype = int)
    x_label = np.array(index,dtype = float)
    img = torch.tensor(x_train[idx])
    img_cuda = img.to(device, dtype=torch.float)
    x1 = model.conv1(img_cuda.view(-1,1,48,48))
    x2 = model.conv2(x1)
    x3 = model.conv3(x2)
    x4 = model.conv4(x3)
    x5 = model.conv5(x4)
    x6 = model.conv6(x5)

    x1 = x1.detach().cpu().numpy()
    x2 = x2.detach().cpu().numpy()
    x3 = x3.detach().cpu().numpy()
    x4 = x4.detach().cpu().numpy()
    x5 = x5.detach().cpu().numpy()
    x6 = x6.detach().cpu().numpy()
    # plt.imsave("origin.jpg",x_train[1].reshape((48,48)),cmap = "Reds")
    for i in range(64):
        plt.subplot(8,8,i+1)
        plt.imshow(x1[0][i],cmap = "Reds")
        plt.axis('off')
    plt.savefig(outputpath+"fig4_1.jpg")
    plt.clf()

    for i in range(64):
        plt.subplot(8,8,i+1)
        plt.imshow(x2[0][i],cmap = "Reds")
        plt.axis('off')
    plt.savefig(outputpath+"fig4_2.jpg")
    plt.clf()

    for i in range(64):
        plt.subplot(8,8,i+1)
        plt.imshow(x3[0][i],cmap = "Reds")
        plt.axis('off')
    plt.savefig(outputpath+"fig4_3.jpg")
    plt.clf()

    for i in range(64):
        plt.subplot(8,8,i+1)
        plt.imshow(x4[0][i],cmap = "Reds")
        plt.axis('off')
    plt.savefig(outputpath+"fig4_4.jpg")
    plt.clf()

    for i in range(64):
        plt.subplot(8,8,i+1)
        plt.imshow(x5[0][i],cmap = "Reds")
        plt.axis('off')
    plt.savefig(outputpath+"fig4_5.jpg")
    plt.clf()
    
    for i in range(64):
        plt.subplot(8,8,i+1)
        plt.imshow(x6[0][i],cmap = "Reds")
        plt.axis('off')
    plt.savefig(outputpath+"fig4_6.jpg")

    
 

