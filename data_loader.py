import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import glob
import pickle
from preprocessing import get_sub_images_permutation

class PuzzleDataset(Dataset):

    def __init__(self, path_dataset):

        with open(path_dataset, 'r') as f:
            files = f.read()
        files = files.split('\n')[:-1]

        self.len = len(files)
        self.index_to_file = {}
        self.labels = {}

        for index, path in enumerate(files):
            #print(path)
            path, label = path.split(',')
            #print(path, label)
            #break
            self.index_to_file[index] = path
            self.labels[index] = int(label)

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        #label = np.random.randint(24) #reemplaza al random choice
        label = self.labels[index]

        sub_images_path = os.path.join(self.index_to_file[index])
        #print(self.index_to_file[index])
        #print(sub_images_path, label)

        with open(sub_images_path, 'rb') as f:
            sub_images = pickle.load(f)

        sub_images = get_sub_images_permutation(sub_images, label)
        a = np.dot(np.asarray(sub_images), [0.2989, 0.5870, 0.1140]) #convert to grayscale
        #print(a.shape)

        a = torch.from_numpy(a)

        a.view(4, 1, 114, 114)
        #a = a / 255.0
        # mean = 0.0
        # std = 1.0
        # a = a + torch.randn(a.size()) * std + mean
        #a = a.view(4, 3, 114, 114)
        #a = a.view(4, 3, 75, 75)
        return a, label


def select_random_permutation(total_permutation_number=24, num_random_permutation=24):
    a = np.arange(total_permutation_number)
    return np.random.choice(a, num_random_permutation, replace=False)






if __name__ == '__main__':

    #PuzzleDataset that receive the split_data_flick_sub_images files
    path ='split_data_flickr_sub_images/train.txt'
    puzzle_data = PuzzleDataset(path)
    train_loader = DataLoader(puzzle_data, batch_size=10, shuffle=False)
    batches = iter(train_loader)

    for images, label in train_loader:
        #print(images.size())
        #print(label)
        break


    #Tambien quizas seria bueno generar N etiquetas por cada muestra..
    # Ahorita el script que carga datos desde un archivo usa N=1.. seria aumenta N por cada etiqueta
    # Estas N pueden ser seleccionadas al azar para cada imagen..


    # o puede seleccionar N permutaciones y usar esas N fijas para entrenar el modelo??? esto no
    # porque sino el modelo estaria limitado a esas N categorias en lugar de todas (N max es 24 para el puzzle 2 x 2)