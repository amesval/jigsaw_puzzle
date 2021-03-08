# =============================================================================
# from __future__ import print_function, division
# import os
# import torch
# import pandas as pd
# from skimage import io, transform
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
# 
# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")
# 
# plt.ion()   # interactive mode
# =============================================================================


import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
#from PIL import Image
#from multiprocessing import cpu_count
import glob
import pickle


# class PuzzleDataset(Dataset):
    
#     def __init__(self, path_dataset):
        
#         self.path_dataset = path_dataset
#         self.files = glob.glob(self.path_dataset+'*')
#         self.len = len(self.files)
#         self.index_to_file = {}
#         for index, path in enumerate(self.files):
#             self.index_to_file[index] = path
    
#     def __len__(self):
#         return self.len
    
#     def __getitem__(self, index):
        
#         image_path = os.path.join(self.index_to_file[index], 'image', 'image.jpg')
#         puzzle_path = os.path.join(self.index_to_file[index], 'puzzle', 'puzzle.jpg')
        
#         image = np.asarray(Image.open(image_path))
#         puzzle = np.asarray(Image.open(puzzle_path))
        
#         return torch.from_numpy(puzzle), torch.from_numpy(image)
    
    
# class PuzzleDataset2(Dataset):
    
#     def __init__(self, path_dataset):
        
#         self.path_dataset = path_dataset
#         self.files = glob.glob(self.path_dataset+'*')
#         self.len = len(self.files)
#         self.index_to_file = {}
#         for index, path in enumerate(self.files):
#             self.index_to_file[index] = path
    
#     def __len__(self):
#         return self.len
    
#     def __getitem__(self, index):
        
#         image_path = os.path.join(self.index_to_file[index], 'image', 'image.jpg')
#         sub_images_path = os.path.join(self.index_to_file[index], 'sub_images', 'random_sub_images.pkl')
        
#         image = np.asarray(Image.open(image_path))
#         with open(sub_images_path, 'rb') as f:
#             sub_images = pickle.load(f)
        
#         return torch.from_numpy(sub_images), torch.from_numpy(image)
    
    
class PuzzleDataset3(Dataset):
    
    def __init__(self, path_dataset):
        
        self.path_dataset = path_dataset
        self.files = glob.glob(self.path_dataset+'*')
        self.len = len(self.files)
        self.index_to_file = {}
        for index, path in enumerate(self.files):
            self.index_to_file[index] = path
            
        self.num_permutations = len(glob.glob(self.files[0]+'/sub_images/*'))
        print(self.num_permutations)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        
        label = np.random.randint(self.num_permutations)
        
        sub_images_path = os.path.join(self.index_to_file[index],
                                       'sub_images', 
                                       str(label), 
                                       'random_sub_images.pkl')
        
        with open(sub_images_path, 'rb') as f:
            sub_images = pickle.load(f)
        print(label)
        return torch.from_numpy(np.asarray(sub_images)), label#torch.IntTensor(label)
            

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://www.youtube.com/watch?v=zN49HdDxHi8
# https://discuss.pytorch.org/t/how-does-enumerate-trainloader-0-work/14410/2

if __name__ == '__main__':
    
    puzzle_data = PuzzleDataset3('/Users/amesval/Documents/puzzle_2_by_2/')
    
    trainloader = DataLoader(puzzle_data, batch_size=32)
    
    batches = iter(trainloader)
    
    #does dataloader change when the epoch finish?