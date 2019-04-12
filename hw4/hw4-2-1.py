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
class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
            # self.conv_output =  torch.tensor(grad_out,requires_grad=True).cuda()
        # Hook the selected layer
        self.model.conv2.register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (48,48,1)))
        # Process image and return variable
        img = torch.tensor(random_image,dtype = torch.float)
        img.requires_grad = True

        # Define optimizer for the image
        # set_trainable(img, False)
        optimizer = Adam([img], lr=0.1)
        for i in range(1, 1001):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = img
            x = x.to(device, dtype=torch.float)
            x = self.model.conv1(x.view(-1,1,48,48))
            x = self.model.conv2(x)


            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            # self.created_image = recreate_image(img)
        self.created_image = img.detach().cpu().numpy().reshape((48,48))
            # Save image
        return self.created_image

if __name__ == '__main__':
    cnn_layer = 2
    # Fully connected layer is not needed
    inputpath  = sys.argv[1]
    outputpath = sys.argv[2]
    device = torch.device('cuda')
    model = MyNet()
    model.to(device)
    model.load_state_dict(torch.load("aug_best_kernal3-epoch600.th")['model'])
    
    np.random.seed(7777)
    # Layer visualization with pytorch hooks
    for filter in range(128):
        print(filter+1)
        layer_vis = CNNLayerVisualization(model, cnn_layer, filter)
        img = layer_vis.visualise_layer_with_hooks()
        plt.subplot(8,16,filter+1)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(outputpath+"fig2_1.jpg")
