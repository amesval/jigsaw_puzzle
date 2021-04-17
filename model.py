import torchvision.models as models
import torch.nn as nn
from torchsummary import summary
from torch import cat
import torch


class Net(nn.Module):

    #def __init__(self, alexnet, num_outputs, num_tiles):
    def __init__(self, num_outputs, num_tiles):
        super(Net, self).__init__()

        self.num_tiles = num_tiles
        self.num_outputs = num_outputs

        # normalization module
        #self.normalization = nn.Sequential(Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.normalization = nn.Sequential(Normalization())
        #self.gaussian_noise = nn.Sequential(AddGaussianNoise())

        #https: // github.com / shivaverma / Jigsaw - Solver / blob / master / model.py
        # self.extract_features = nn.Sequential(
        #     nn.Conv2d(3, 64, (5,5), (2,2), padding=(0,0)),
        #     #nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(64, 256, (5, 5), (2, 2), padding=(0, 0)),
        #     #nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(256, 256, (3, 3), (2, 2), padding= (0,0)),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.3, inplace=False),
        #
        # )

        #self.extract_features = alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4,4)) # After this we need a reshape

        self.latent_space = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            #nn.Linear(4 * 4 * 256, 512),
            nn.Linear(4 * 4 * 32, 512),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.5, inplace=False),
        )


        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * num_tiles, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.3, inplace=False),
        #
        #     nn.Linear(512, 128),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(128, self.num_outputs),
        #     nn.LogSoftmax(dim=1)
        # )

        # self.extract_features = nn.Sequential(
        #     nn.Conv2d(3, 64, (11,11), (4,4), (2,2)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 256, (5,5), (1,1), (2,2)),
        #     nn.ReLU(inplace=True),
        #     #nn.Dropout(p=0.5, inplace=False)
        # )
        #
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * num_tiles, 512),
        #     nn.ReLU(inplace=True),
        #     #nn.Dropout(p=0.3, inplace=False),
        #
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(256, self.num_outputs),
        #     nn.LogSoftmax(dim=1)
        # )


        ##### ESTE FUNCIONA RELATIVAMENTE BIEN ### CREO QUE TAMBIEN EL DE ARRIBITa
        # self.extract_features = nn.Sequential(
        #     nn.Conv2d(1, 64, (11,11), (4,4), (2,2)),
        #     #nn.Conv2d(3, 64, (11, 11), (4, 4), (2, 2)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 256, (5,5), (1,1), (2,2)),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(512 * num_tiles, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.3, inplace=False),
        #     nn.Linear(512, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, self.num_outputs),
        #     nn.LogSoftmax(dim=1)
        # )
        ##############################################################33


        ############### 72.5% de accuracy en validacion :D #####################
        ### Moraleja: Estaba haciendo overfitting asi que hice el modelo menos complejo :)
        ##72.5% en 55 epocas , utilizando el dataset de train_triple, tamano de imagen 114
        ##imagenes en escala de grises y el dataset no esta en shuffle
        ##quizas seria bueno intentar con un learning rate scheduler
        ##entrenar con un dataloader fijo las primeras epocas y luego hacerle un shuffle al batch?
        ##TODO:seria bueno probar con un dataset mas sencillo que tiene objetos fijos, por ejemplo el PASCAL VOC, animal10, mnist, etc.

        # self.latent_space = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=False),
        #     #nn.Linear(4 * 4 * 256, 512),
        #     nn.Linear(4 * 4 * 64, 512),
        #     nn.ReLU(inplace=True),
        #     #nn.Dropout(p=0.5, inplace=False),
        # )

        self.extract_features = nn.Sequential(
            nn.Conv2d(1, 32, (11,11), (4,4), (2,2)),
            #nn.Conv2d(3, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, (5,5), (1,1), (2,2)), #el modelo bueno aqui tiene 64 filtros (32, 64, ..) ... esta configuracion no es buena.. regresarlo a 64
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512 * num_tiles, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False), #el modelo bueno aqui tiene p=0.3
            # nn.Linear(256, 128),
            # nn.ReLU(inplace=True),
            nn.Linear(256, self.num_outputs),
            nn.LogSoftmax(dim=1)
        )
        ###########################################


    def forward(self, x):
        #print(x.size())
        values = []
        #x = x / 255.0
        for i in range(self.num_tiles):
            a = self.process_per_tile(x[:, i, :, :, :])
            #print(a.size())
            #a = self.latent_space(a)
            #print(a.size())
            values.append(a)
        x = cat(values, 1)
        #print(x.size())
        x = self.classifier(x)
        #print(x.size())
        #print(x)
        return x

    def process_per_tile(self, x):
        #print(x.size()) # It seems that torchsummary always use a batch of 2 elements for checking solution.
        x = self.normalization(x)
        #x = self.gaussian_noise(x)
        x = self.extract_features(x)
        #print(x.size())
        x = self.avgpool(x)
        x = x.view((-1, x.size()[1] * x.size()[2] * x.size()[3]))
        x = self.latent_space(x)
        #print(x.size())
        return x


class Normalization(nn.Module):
    #def __init__(self, mean, std):
    def __init__(self):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        #self.mean = torch.tensor(mean, device='cuda').view(-1, 1, 1)
        #self.std = torch.tensor(std, device='cuda').view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        img = img / 255.0
        #return (img - self.mean) / self.std
        return img


# class AddGaussianNoise(nn.Module):
#     def __init__(self, mean=0., std=1.):
#         super(AddGaussianNoise, self).__init__()
#         self.std = std
#         self.mean = mean
#
#     def forward(self, tensor):
#         return tensor + torch.randn(tensor.size()).cuda() * self.std + self.mean


if  __name__ == '__main__':

    #alex = models.alexnet(pretrained=True)
    #net = Net(alex, 24, 4)

    net = Net(24, 4)

    #print(net)
    #summary(net, (4, 3, 227, 227), 32)
    # #https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte
    #https://discuss.pytorch.org/t/runtimeerror-input-type-torch-cuda-floattensor-and-weight-type-torch-floattensor-should-be-the-same/21782

    #summary(net, (4, 3, 227, 227), 32, device='cpu')

    summary(net.cuda(), (4, 3, 227, 227), 32, device = "cuda")

    #summary(alex.cuda(), (3, 227, 227), 32, device='cuda') #This is ok
