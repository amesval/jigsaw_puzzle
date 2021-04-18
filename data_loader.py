import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import glob
import pickle
from preprocessing import get_sub_images_permutation

class PuzzleDataset(Dataset):

    def __init__(self, path_dataset):
        self.path_dataset = path_dataset
        self.image_folders = glob.glob(self.path_dataset + '*')
        self.len = len(self.image_folders)
        self.index_to_file = {}
        self.labels = select_random_permutation(24, 24)

        for index, path in enumerate(self.image_folders):
            self.index_to_file[index] = path

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        label = np.random.choice(self.labels)

        sub_images_path = os.path.join(self.index_to_file[index],
                                       'sub_images.pkl')

        with open(sub_images_path, 'rb') as f:
            sub_images = pickle.load(f)

        print(self.index_to_file[index])

        a = get_sub_images_permutation(sub_images, label)
        a = np.dot(np.asarray(sub_images), [0.2989, 0.5870, 0.1140]) #convert to grayscale
        a = torch.from_numpy(a)
        #print(a.size())

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

    puzzle_data = PuzzleDataset('flickr_sub_images/')
    train_loader = DataLoader(puzzle_data, batch_size=10, shuffle=True)
    batches = iter(train_loader)

    # Aunque shuffle = False, siempre esta generando nuevas etiquetas
    # porque el getitem tiene un shuffle...
    # entonces de todas formas aunque shuffle == False, siempre esta volviendo a generar otro trainloader cuando uno termina..

    # for sub_images, labels in train_loader:
    #     #print(sub_images)
    #     #print(labels)
    #     break
    # print()
    #
    # for sub_images, labels in train_loader:
    #     #print(sub_images)
    #     #print(labels)
    #     break