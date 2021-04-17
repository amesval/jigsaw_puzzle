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

    with open(os.path.join(split_folder, 'train.txt'), 'w') as f:
        for path in train_paths:
            f.write(path+"\n")

    with open(os.path.join(split_folder, 'val.txt'), 'w') as f:
        for path in val_paths:
            f.write(path + "\n")

    with open(os.path.join(split_folder, 'test.txt'), 'w') as f:
        for path in test_paths:
            f.write(path + "\n")

    print(f"The last {len(paths) - train_samples - val_samples - test_samples} files were discarded to match desired proportion {proportion}.")
    return train_paths, val_paths, test_paths


#
# def make_subset(folder_images, subset_folder, max_num_images):
#     paths = glob.glob(folder_images + '*')
#     paths = paths[:max_num_images]
#
#     if not os.path.exists(subset_folder):
#         os.mkdir(subset_folder)
#
#     for path in paths:
#         shutil.copyfile(path, os.path.join(subset_folder, path.split('/')[-1] ))
#
# def training_set(puzzle_2_by_2_path, total_permutations, num_of_desire_permutation,data_path, range_of_users=(0, 300)):
#     user_paths = glob.glob(puzzle_2_by_2_path+'*')
#     num_original_images = len(user_paths)
#     with open(data_path, 'w') as f:
#         for i in range(range_of_users[0], range_of_users[1]):
#             selection = np.random.choice(np.arange(total_permutations), num_of_desire_permutation, replace=False)
#             for permutation in selection:
#                 path = os.path.join(puzzle_2_by_2_path, str(i), 'sub_images', str(permutation), 'random_sub_images.pkl')
#                 f.write(path+'\n')

def generate_label_data(path_data, output_file, number_of_permutations):

    with open(path_data, 'r') as f:
        paths = f.read()
    paths = paths.split('\n')[:-1]

    with open(output_file, 'w') as f:
        for path in paths:
            label = np.random.randint(number_of_permutations)
            line = f"{path}, {label}\n"
            f.write(line)


def rename_inside_folder(folder_path):
    paths = glob.glob(folder_path + '*')
    #print(paths)
    for path in paths:
        new_path = path.replace('/', '/aaa')
        os.rename(path, new_path)

# def rename_file(path):
#     with open(path, 'w+') as f:
# lo hare manualmente        f


if __name__ == '__main__':



    # train_paths, val_paths, test_paths = split_data('flickr30k_images/',
    #                                                 'split_data_flickr/',
    #                                                 (.60, .20, .20))
    #
    # print(len(train_paths) + len(val_paths) + len(test_paths))

    generate_label_data('split_data_flickr/train.txt', 'split_data_flickr/train_labels3.txt', 24)

    #generate_label_data('split_data_flickr/train.txt', 'split_data_flickr/train_labels.txt', 24)
    # generate_label_data('split_data_flickr/val.txt', 'split_data_flickr/val_labels.txt', 24)
    # generate_label_data('split_data_flickr/test.txt', 'split_data_flickr/test_labels.txt', 24)

    #rename_inside_folder('flickr30k_images_triple/')