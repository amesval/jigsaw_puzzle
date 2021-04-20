import os
import glob
import shutil
import numpy as np

def split_data(folder_path, split_folder, proportion):

    if not os.path.exists(split_folder):
        os.mkdir(split_folder)

    paths = glob.glob(folder_path+'*')
    random_paths = np.random.choice(paths, len(paths), replace=False)

    train_samples = int(len(paths) * proportion[0])

    val_samples = int(len(paths) * proportion[1])

    test_samples = int(len(paths) * proportion[2])


    train_paths = random_paths[:train_samples]
    val_paths = random_paths[train_samples: train_samples + val_samples]
    test_paths = random_paths[train_samples + val_samples: train_samples + val_samples + test_samples]

    aux = '/sub_images.pkl'
    with open(os.path.join(split_folder, 'train.txt'), 'w') as f:
        for path in train_paths:
            #label = np.random.choice(np.arange(24))
            #f.write(path+ aux + f',{label}' + "\n")
            for i in range(24):
                f.write(path + aux + f',{i}' + "\n")

    with open(os.path.join(split_folder, 'val.txt'), 'w') as f:
        for path in val_paths:
            #label = np.random.choice(np.arange(24))
            #f.write(path+ aux + f',{label}' + "\n")
            for i in range(24):
                f.write(path + aux + f',{i}' + "\n")

    with open(os.path.join(split_folder, 'test.txt'), 'w') as f:
        for path in test_paths:
            #label = np.random.choice(np.arange(24))
            #f.write(path+ aux + f',{label}' + "\n")
            for i in range(24):
                f.write(path + aux + f',{i}' + "\n")

    print(f"The last {len(paths) - train_samples - val_samples - test_samples} files were discarded to match desired proportion {proportion}.")
    return train_paths, val_paths, test_paths




if __name__ == '__main__':



    # train_paths, val_paths, test_paths = split_data('flickr_sub_images/',
    #                                                 'split_data_flickr_sub_images2/',
    #                                                 (.60, .20, .20))

    train_paths, val_paths, test_paths = split_data('PASCAL_sub_images/',
                                                    'split_data_pascal_images/',
                                                    (.60, .20, .20))

    print(len(train_paths) + len(val_paths) + len(test_paths))
