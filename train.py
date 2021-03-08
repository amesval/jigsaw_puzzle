import datetime
import torch
from torch.utils.data import DataLoader
from data_loader import PuzzleDataset2
from transfer_learning import Net
import torch.optim as optim
import torch.nn as nn


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            
            outputs = model(imgs)
            
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))

            
if __name__ == '__main__':    
    
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #puzzle_data = PuzzleDataset2('/Users/amesval/Documents/puzzle_dataset_v1_4_4/')
    puzzle_data = PuzzleDataset3('/Users/amesval/Documents/puzzle_2_by_2/')
    train_loader = DataLoader(puzzle_data, batch_size=32)
    
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    
    training_loop(
        n_epochs = 100,
        optimizer = optimizer,
        model = model,
        loss_fn=loss_fn,
        train_loader = train_loader)

    PATH = './test1.pt'
    torch.save(model.state_dict(), PATH)
    
    
    
    #Me falta modificar la red apropiadamente,
    # hacer la red
    # congelar las capas que no me interesan con el named_children,
    #hacer el resize de las imagenes
    # hacer ciclycal learning rates
    # dividir dataset en entrenamiento y prueba
    # entrenar
    # normalizar entrada
    # plotear function de costo
    # hacer ciclycal learning rates
    # entrenar
    # probar
    #entrenar sin transfer learning
    
    
# =============================================================================
#     
#     m_state_dict = torch.load('./test1.pt')
#     new_m = Net()
#     new_m.load_state_dict(m_state_dict)
# =============================================================================
                

#TODO: 
#Testing loop
#Leslie method of Cyclical learning rates to find learning rate.
#Modify neural network
#Check summary method
#Review dataset puzzle2
