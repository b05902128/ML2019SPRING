import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import slic
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


def predict(image):
    model.eval()
    image = torch.tensor(image[:,:,:,0])
    img_cuda = image.to(device, dtype=torch.float)
    return model(img_cuda).detach().cpu().numpy()
    # Input: image tensor
    # Returns a predict function which returns the probabilities of labels ((7,) numpy array)
    # ex: return model(data).numpy()
def segmentation(image):
    # Input: image numpy array
    # Returns a segmentation function which returns the segmentation labels array ((48,48) numpy array)
    # ex: return skimage.segmentation.slic()
    k = slic(image,n_segments=100,compactness = 0.000001)
    return k
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
    x_train = np.array(value,dtype = int)
    x_label = np.array(index,dtype = float)
    w,h = x_train.shape
    x_train_rgb = np.empty((w,h,3),dtype = int)
    for i in range(3):
        x_train_rgb[:,:,i] = value
    # Initiate explainer instance
    explainer = lime_image.LimeImageExplainer()
    np.random.seed(1088)
    count = 0
    for i in [22,299,33,7,20,26,36]:
    # Get the explaination of an image
        explaination = explainer.explain_instance(
                                image=x_train_rgb[i,:,:].reshape((48,48,3)), 
                                classifier_fn=predict,
                                segmentation_fn=segmentation
                            )
        # Get processed image
        image, mask = explaination.get_image_and_mask(
                                    label=x_label[i],
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=10,
                                    min_weight=0.0
                                )
        image = np.array(image,dtype = np.uint8)
        # save the image
        plt.imsave(outputpath+'fig3_'+str(count)+'.jpg', image)
        count+=1

# using the first ten images for example
    