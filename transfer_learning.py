import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import cat

# =============================================================================
# alexnet = models.alexnet(pretrained=True)
# 
# model_blocks = [n for n, subm in alexnet.named_children()
#                 if len(list(subm.parameters())) > 0]
# 
# #alexnet.named_children() is a generator
# 
# for name, layers in alexnet.named_children():
#     print(name)
#     params = layers.parameters() #generator
#     print(params)
#     for i in params:
#         print("\n\n\nyuju")
#         print(i.size())
#         
# a = [i for i in alexnet.children()]    
# =============================================================================
        
class Net(nn.Module):
    
    def __init__(self, alexnet, num_outputs, num_tiles):
        super(Net, self).__init__()
        
        self.pretrained = nn.Sequential(*([alexnet.features, 
                                          alexnet.avgpool, 
                                          alexnet.classifier[0],
                                          alexnet.classifier[1]]))
        # self.fc1 = nn.Linear(4096, 4096)
        # self.act1 = nn.ReLU()
        # self.fc2 = nn.Linear(4096, 128)
        # self.act2 = nn.ReLU()
        # self.fc3 = nn.Linear(128*num_tiles, 100)
        # self.act3 = nn.ReLU()
        # self.fc4 = nn.Linear(100, num_outputs)
        # self.act4 = nn.Softmax()
        
        
    def forward(self, x):
        #B, num_tiles, C, H, W = x.size()
        #print(num_tiles, C, H, W)
        #print(B)
        #x = self.pretrained(x[:, 0, :, :, :])
        x = self.pretrained(x)
        
        # outputs = []
        # for i in range(num_tiles):
        #     outputs.append(self.process_per_tile(x[:, i, :, :, :]))
        
        # print("yolotl")
        
        # x = cat(outputs, 0)
        # x = x.view(B, -1)
        # x = self.act3(self.fc3(x))
        # x = self.act4(self.fc4(x))
        
        return x
    
    def process_per_tile(self, x):
        
        x = self.pretrained(x)
        #x = self.act1(self.fc1(x))
        #x = self.act2(self.fc2(x))
        
        return x
    
#https://discuss.pytorch.org/t/how-can-i-connect-a-new-neural-network-after-a-trained-neural-network-and-optimize-them-together/48293/4


net = Net(models.alexnet(), 24, 4)
#print(net)
print(summary(net, (3, 227, 227), 32, device='cpu'))