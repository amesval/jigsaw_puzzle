import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import glob
import pickle


class PuzzleDataset(Dataset):

    def __init__(self, path_dataset):
        self.path_dataset = path_dataset
        self.files = glob.glob(self.path_dataset + '*')
        self.len = len(self.files)
        self.index_to_file = {}
        self.labels = {}
        for index, path in enumerate(self.files):
            self.index_to_file[index] = path
            self.labels[index] = int(os.listdir(os.path.join(path, "sub_images"))[0])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        label = self.labels[index]
        sub_images_path = os.path.join(self.index_to_file[index],
                                       'sub_images',
                                       str(label),
                                       'random_sub_images.pkl')

        with open(sub_images_path, 'rb') as f:
            sub_images = pickle.load(f)
        #print(self.index_to_file[index], label)

        a = np.dot(np.asarray(sub_images), [0.2989, 0.5870, 0.1140]) #convert to grayscale
        a = torch.from_numpy(a)

        a.view(4, 1, 114, 114)
        #a = a / 255.0
        # mean = 0.0
        # std = 1.0
        # a = a + torch.randn(a.size()) * std + mean
        #a = a.view(4, 3, 114, 114)
        #a = a.view(4, 3, 75, 75)
        return a, label





if __name__ == '__main__':

    puzzle_data = PuzzleDataset('v2puzzle_2_by_2_train/')
    train_loader = DataLoader(puzzle_data, batch_size=10, shuffle=False) #siempre genera el dataloader en el mismo orden :o
    batches = iter(train_loader)

    # print(len(puzzle_data))
    # print(len(batches))

    # cont = 0
    # for images, labels in train_loader:
    #     if cont == 0:
    #         print(labels)
    #         break
    #     cont += 1
    # print(cont)
    # cont = 0
    # for images, labels in train_loader:
    #     if cont == 0:
    #         print(labels)
    #         print("james")
    #         break
    #     cont += 1
    # print(cont)
    #el trainloader consiste de muchos batches... parece que se genera cuando se invoca?..
    #parece que cuando se termina un train_loader, automaticamente genera otro al azar si shuffle=True (sino, seran identicos)

    #https: // discuss.pytorch.org / t / dataloader - super - slow / 38686