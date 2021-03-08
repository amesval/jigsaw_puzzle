import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
# https://neurohive.io/en/popular-networks/vgg16/
# http://datahacker.rs/building-inception-network/


# =============================================================================
# class Net(nn.Module):
#     
#     def __init__(self):
#         super(Net, self).__init__()
#             
#         self.conv1 = nn.Conv2d(3, 16, (3,3), padding=1)
#         self.act1 = nn.Tanh()
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(16, 8, (3,3), padding=1)
#         self.act2 = nn.Tanh()
#         self.pool2 = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(8 * 8 * 8, 32)
#         self.act3 = nn.Tanh()
#         self.fc2 = nn.Linear(32, 2)
#             
#     def forward(self, x):
#         out = self.pool1(self.act1(self.conv1(x)))
#         out = self.pool2(self.act2(self.conv2(out)))
#         out = out.view(-1, 8 * 8 * 8)
#         out = self.act3(self.fc1(out))
#         out = self.fc2(out)
#         return out
# 
# 
# if __name__ == '__main__':
#     
#     net = Net()
#     print(summary(net, (3,32, 32), 32, device='cpu'))
#     #print(net)
# =============================================================================


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, (3,3), padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 32, (3,3), padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 16, (3,3), padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(16, 8, (3,3), padding=1)
        self.act4 = nn.ReLU()
        self.pool4 = nn.ReLU()
        
    def forward(self, x):
       print(x.size()) 
       out = self.process_sub_image(x)
       return out
   
    def process_sub_image(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = self.pool3(self.act3(self.conv3(out)))
        out = self.pool4(self.act4(self.conv4(out)))
        return out

net = Net()
print(summary(net, (3, 32, 32), 32, device='cpu'))
print(net)