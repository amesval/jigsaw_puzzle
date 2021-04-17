import datetime
import torch
from torch.utils.data import DataLoader
from data_loader import PuzzleDataset
from model import Net
import torch.optim as optim
import torch.nn as nn
import math
import matplotlib.pyplot as plt
#
# def find_lr(model, train_loader, loss_fn, optimizer, device, init_value=1e-8, final_value=10.0):
#
#     number_in_epoch = len(train_loader) - 1
#     update_step = (final_value / init_value) ** (1 / number_in_epoch)
#     lr = init_value
#     optimizer.param_groups[0]["lr"] = lr
#     best_loss = 0.0
#     batch_num = 0
#     losses = []
#     log_lrs = []
#     for inputs, labels in train_loader:
#         batch_num += 1
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = loss_fn(outputs, labels)
#
#         # Crash out if loss explodes
#
#         if batch_num > 1 and loss > 4 * best_loss:
#             return log_lrs[10:-5], losses[10:-5]
#
#         # Record the best loss
#
#         if loss < best_loss or batch_num == 1:
#             best_loss = loss
#
#         # Store the values
#
#         losses.append(loss)
#         log_lrs.append(math.log10(lr))
#
#         # Do the backward pass and optimize
#
#         loss.backward()
#         optimizer.step()
#
#         # Update the lr for the next step and store
#
#         lr *= update_step
#         optimizer.param_groups[0]["lr"] = lr
#     return log_lrs[10:-5], losses[10:-5]




def training_loop(model, train_loader, val_loader, n_epochs, optimizer, loss_fn, device):

    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        train_acc = 0.0
        total = 0.0
        model.train()

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
            #scheduler.step()

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

        if epoch == 1 or epoch % 5 == 0:
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

    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    puzzle_data = PuzzleDataset('v2puzzle_2_by_2_train_triple/')
    train_loader = DataLoader(puzzle_data, batch_size=64, shuffle=True, num_workers=4)

    puzzle_data = PuzzleDataset('v2puzzle_2_by_2_val/')
    val_loader = DataLoader(puzzle_data, batch_size=64, num_workers=4)

    model = Net(24, 4)
    model = model.to(device)

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.NLLLoss()

    #logs, losses = find_lr(model, train_loader, loss_fn, optimizer, device, init_value=1e-8, final_value=10.0)
    #plt.plot(logs, losses)

    train_losses, val_losses = training_loop(model,
                  train_loader,
                  val_loader,
                  100,
                  optimizer,
                  loss_fn,
                  device)


    # otra opcion es usar un dataset mas sencillo




    # PATH = 'jigsaw_v1_pretrained.pt'
    # torch.save(model.state_dict(), PATH)


    # https://stats.stackexchange.com/questions/245502/why-should-we-shuffle-data-while-training-a-neural-network

    # Cambiar la normalizacion al dataloader y Agregar colorjitter???
    #agregar el jitter a la red neuronal


    #TODO: Agregar funcion para formatear otros datasets (Animal 10)
    #TODO: Agregar funciones de prueba