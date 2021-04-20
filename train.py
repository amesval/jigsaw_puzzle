import datetime
import torch
from torch.utils.data import DataLoader
from data_loader import PuzzleDataset
from model2 import Net
import torch.optim as optim
import torch.nn as nn
import math
import matplotlib.pyplot as plt

# https://discuss.pytorch.org/t/should-backward-function-be-in-the-loop-of-epoch-or-batch/65083/3
# https://towardsdatascience.com/pytorch-training-tricks-and-tips-a8808ebf746c
# https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944/2

# https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
# https://towardsdatascience.com/super-convergence-with-cyclical-learning-rates-in-tensorflow-c1932b858252
# https://www.datacamp.com/community/tutorials/cyclical-learning-neural-nets
# https://www.jeremyjordan.me/nn-learning-rate/
# https://arxiv.org/pdf/1506.01186.pdf

def training_loop(model, train_loader, val_loader, n_epochs, optimizer, loss_fn, device):

    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        train_acc = 0.0
        total = 0.0
        model.train()
        #optimizer.zero_grad() #for batch gradient descent (vanilla)
        for images, labels in train_loader:
            images = images.view(images.size()[0], 4, 1, 114, 114)
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)
            train_acc += (predictions == labels).sum()
            total += len(predictions)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        #optimizer.step() #for batch gradient descent (vanilla)

        train_acc = train_acc / total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        #check validation loss after every epoch
        val_loss = 0.0
        val_acc = 0.0
        total = 0.0
        with torch.no_grad():
            model.eval()
            for images, labels in val_loader:
                images = images.view(images.size()[0], 4, 1, 114, 114)
                images = images.to(device, dtype=torch.float)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                val_acc += (predictions == labels).sum()
                total += len(predictions)

        val_acc = val_acc / total
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        #if epoch == 1 or epoch % 5 == 0:
        print('{} Epoch {}/{}, Training loss {}, Validation loss {}, Train accuracy {}, validation accuracy {}'.format(
            datetime.datetime.now(),
            epoch,
            n_epochs,
            avg_train_loss,
            avg_val_loss,
            train_acc,
            val_acc
        ))

    return train_losses, val_losses

            
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #puzzle_data = PuzzleDataset('split_data_flickr_sub_images2/train.txt')
    puzzle_data = PuzzleDataset('split_data_pascal_images/train.txt') #PASCAL DATA
    train_loader = DataLoader(puzzle_data, batch_size=128, shuffle=True, num_workers=8)
    #puzzle_data = PuzzleDataset('split_data_flickr_sub_images/val.txt')
    puzzle_data = PuzzleDataset('split_data_pascal_images/val.txt') #PASCAL DATA
    val_loader = DataLoader(puzzle_data, batch_size=128, num_workers=4, shuffle=False)

    model = Net(24, 4)
    model = model.to(device)

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)
    #optimizer = optim.Adam(model.parameters())
    loss_fn = nn.NLLLoss()

    train_losses, val_losses = training_loop(model,
                  train_loader,
                  val_loader,
                  10,
                  optimizer,
                  loss_fn,
                  device)

    PATH = 'jigsaw_v2_pretrained_on_pascal.pt'
    torch.save(model.state_dict(), PATH)

    #Entrenar con el Pascal VOC con todos los objetos y todas las permutaciones.. (Es un dataset mas sencillo)

    #Entrenar con todas las permutaciones en una epoca??

    # Tambien podria entrenar el pascal VOC apartando una clase (avion) solamente para prueba y ver si los features generalizan
    # para clases que estna fuera del entrenamiento