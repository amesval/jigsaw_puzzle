import numpy as np
from PIL import Image
import os
import glob
import multiprocessing
from functools import partial
import pickle
from itertools import permutations
from math import sqrt
import tqdm

#https://shiva-verma.medium.com/solving-jigsaw-using-neural-nets-cc543a5f025c
#https://github.com/anilsathyan7/Jigsaw-Net/blob/master/Jigsaw_Net.ipynb

def get_sub_images(image, num_images_per_row, num_images_per_col):
    """Obtain num_images_per_row * num_images_per_col subimages from original image.
    
    image: np.ndarray that represent an image in RGB format (Rows, Cols, Channels).
    num_images_per_row: Number of images per row.
    num_images_per_col: Number of images per column.
    
    Image is crop from the last rows and cols to perfectly reconstruct from subimages.
    """
    
    height_size = image.shape[0]
    width_size = image.shape[1]
    #print(height_size, width_size)
    
    height_size_sub_image = height_size // num_images_per_col
    width_size_sub_image = width_size // num_images_per_row
    #print(height_size_sub_image, width_size_sub_image)
    
    reshape_height_size = height_size_sub_image * num_images_per_col
    reshape_width_size = width_size_sub_image * num_images_per_row
    
    image = image[:reshape_height_size, :reshape_width_size, :]
    #print("image reshape: ", image.shape)
    
    sub_images = []
    for i in range(0, num_images_per_col):
        for j in range(0, num_images_per_row):
            
            row_init_coord = i * height_size_sub_image
            row_end_coord = row_init_coord + height_size_sub_image
            
            col_init_coord = j * width_size_sub_image
            col_end_coord = col_init_coord + width_size_sub_image
   
# =============================================================================
#             pil_img = Image.fromarray(image[row_init_coord : row_end_coord,
#                         col_init_coord : col_end_coord, :])
#             pil_img.show()
# =============================================================================
            
            sub_images.append(image[row_init_coord : row_end_coord,
                        col_init_coord : col_end_coord, :])
            
    return image, sub_images


def randomize_sub_images_2_2(sub_images):
    
    indices = np.arange(len(sub_images))
    output = []
    for label, permutation in enumerate(permutations(indices)):
        selection = []
        for index in permutation:
            selection.append(sub_images[index])
        output.append((selection, permutation, label))
    
    return output


#### esta es la nueva ############
def one_random_sub_image_2_2(sub_images, label):

    indices = get_permutation_from_label(label, len(sub_images))
    selection = []
    for index in indices:
        selection.append(sub_images[index])
    return selection
#####################################





def resize_image(pil_image, pil_size,pil_resampling):
    
    #resampling: Image.<NEAREST, LANCZOS, BILINEAR, BICUBIC, BOX, HAMMING>
    #pil_size: tuple (Width, Height)
    
    return pil_image.resize(pil_size, pil_resampling)


### Need a function to plot sub_images
def make_puzzle(sub_images, rows=2, cols=2):
    
    puzzle = np.hstack(sub_images[:cols])
    for index in range(cols, len(sub_images), cols):
        pixel_arrays = sub_images[index: index+cols]
        arrays_block = np.hstack((pixel_arrays))
        puzzle = np.vstack((puzzle, arrays_block))
    
    return puzzle
            

def plot_puzzle(puzzle):    
    
    Image.fromarray(puzzle).show()
    
    
    
def get_permutation_from_label(label, total_sub_images):
    
    permutations_ = list(permutations(range(total_sub_images)))
    
    indices = permutations_[label]
    
    return indices


def read_sub_images(path):
    
    with open(path, 'rb') as file:
        sub_images = pickle.load(file)
    
    return sub_images


def reconstruct_image_from_puzzle(label, sub_images, read_from_pickle=False):
    
    if isinstance(sub_images, str) and not read_from_pickle:
        raise("Please set read_from_pickle to True")
    
    if read_from_pickle:
        sub_images = read_sub_images(sub_images)
    
    num_sub_images = len(sub_images)
    indices = get_permutation_from_label(label, num_sub_images)
    reconstruction = []
    for i in range(len(indices)):
        position = indices.index(i)
        reconstruction.append(sub_images[position])
    
    puzzle = make_puzzle(reconstruction,
                         rows = int(sqrt(num_sub_images)), 
                         cols = int(sqrt(num_sub_images)))
    
    plot_puzzle(puzzle)
    
    

#
# def puzzle_generation_step(path,
#                            index,
#                            out_dataset_dir,
#                            images_per_row,
#                            images_per_col,
#                            target_height,
#                            target_width,
#                            resampling):
#
#     # if not os.path.exists(out_dataset_dir):
#     #     os.mkdir(out_dataset_dir)
#
#     img = Image.open(path)
#     img = resize_image(img, (target_width, target_height), resampling)
#
#
#     image, sub_images = get_sub_images(np.asarray(img), images_per_row, images_per_col)
#     output = randomize_sub_images_2_2(sub_images)
#
#
#     save_path = os.path.join(out_dataset_dir, str(index))
#
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#
#     image_save_path = os.path.join(save_path, 'image')
#     if not os.path.exists(image_save_path):
#         os.mkdir(image_save_path)
#
#     sub_images_save_path = os.path.join(save_path, 'sub_images')
#     if not os.path.exists(sub_images_save_path):
#         os.mkdir(sub_images_save_path)
#
#
#     img.save(os.path.join(image_save_path, 'image.jpg'))
#
#     for element in output:
#         random_sub_images, _, label = element
#         path_ = os.path.join(sub_images_save_path, str(label))
#         if not os.path.exists(path_):
#             os.mkdir(path_)
#         filename = 'random_sub_images.pkl'
#         with open(os.path.join(path_, filename), 'wb') as file:
#             pickle.dump(random_sub_images, file)


def puzzle_generation_step2(path,
                           label,
                           out_dataset_dir,
                           images_per_row,
                           images_per_col,
                           target_height,
                           target_width,
                           resampling):
    # if not os.path.exists(out_dataset_dir):
    #     os.mkdir(out_dataset_dir)
    img = Image.open(path)
    img = resize_image(img, (target_width, target_height), resampling)

    image, sub_images = get_sub_images(np.asarray(img), images_per_row, images_per_col)
    sub_images = one_random_sub_image_2_2(sub_images, label)
    index = path.split('/')[-1][:-4]

    save_path = os.path.join(out_dataset_dir, str(index))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    image_save_path = os.path.join(save_path, 'image')
    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)

    sub_images_save_path = os.path.join(save_path, 'sub_images')
    if not os.path.exists(sub_images_save_path):
        os.mkdir(sub_images_save_path)

    img.save(os.path.join(image_save_path, 'image.jpg'))

    path_ = os.path.join(sub_images_save_path, str(label))
    if not os.path.exists(path_):
        os.mkdir(path_)

    filename = 'random_sub_images.pkl'
    with open(os.path.join(path_, filename), 'wb') as file:
        pickle.dump(sub_images, file)


def generate_dataset2(path, out_dataset_dir, images_per_row,
                     images_per_col, target_height, target_width, resampling):

    with open(path, 'r') as f:
        files = f.read()
    rows = files.split('\n')[:-1]
    info = []
    for row in rows:
        file, label = row.split(',')
        label = int(label.strip())
        info.append((file, label))

    cores = multiprocessing.cpu_count()
    pool_process = multiprocessing.Pool(cores - 2)

    if not os.path.exists(out_dataset_dir):
        os.mkdir(out_dataset_dir)

    partial_function = partial(puzzle_generation_step2, out_dataset_dir=out_dataset_dir,
                               images_per_row=images_per_row,
                               images_per_col=images_per_col,
                               target_height=target_height,
                               target_width=target_width,
                               resampling=resampling)
    pool_process.starmap(partial_function, info)
    # pool_process.starmap(partial_function, tqdm.tqdm(info, total=len(info)))
    pool_process.close()
    pool_process.join()



# def generate_dataset(path, out_dataset_dir,images_per_row,
#                      images_per_col, target_height, target_width, resampling):
#     files = glob.glob(path+'*')
#     info = [(file, index) for index, file in enumerate(files)]
#
#     cores = multiprocessing.cpu_count()
#     pool_process = multiprocessing.Pool(cores-2)
#
#     if not os.path.exists(out_dataset_dir):
#         os.mkdir(out_dataset_dir)
#
#     partial_function = partial(puzzle_generation_step,out_dataset_dir=out_dataset_dir,
#                                                       images_per_row=images_per_row,
#                                                       images_per_col=images_per_col,
#                                                       target_height=target_height,
#                                                       target_width=target_width,
#                                                       resampling=resampling)
#     pool_process.starmap(partial_function, info)
#     #pool_process.starmap(partial_function, tqdm.tqdm(info, total=len(info)))
#     pool_process.close()
#     pool_process.join()







if __name__ == '__main__':
    
    # a = Image.open('image.jpg')
    # image, sub_images = get_sub_images(np.asarray(a), 2, 2)
    # output = randomize_sub_images_2_2(sub_images)
    
    # test_sub_image, test_target, test_label = output[3]
    # print(test_target)
    # puzzle = make_puzzle(test_sub_image, rows=2, cols=2)
    # print(puzzle.shape)
    
    # plot_puzzle(puzzle)

    import time
    start = time.time()
    
    # generate_dataset2('split_data_flickr/train_labels_triple.txt',
    #                   'v2puzzle_2_by_2_train_triplev2/',
    #                   2,
    #                   2,
    #                   150,
    #                   150,
    #                   Image.BICUBIC)


    # generate_dataset2('custom_test_dataset/custom_data.txt',
    #                   'custom_test_generate/',
    #                   2,
    #                   2,
    #                   228,
    #                   228,
    #                   Image.BICUBIC)



    #
    # generate_dataset2('split_data_flickr/test_labels.txt',
    #                   'v2puzzle_2_by_2_test',
    #                   2,
    #                   2,
    #                   228,
    #                   228,
    #                   Image.BICUBIC)
    #
    generate_dataset2('split_data_flickr/val_labels.txt',
                      'v2puzzle_2_by_2_valv2',
                      2,
                      2,
                      150,
                      150,
                      Image.BICUBIC)


    end = time.time()
    print("total generation time: ", end - start)
    
    #si pongo el path como puzzle_2_by_2/ sin el path completo, entonces no funciona


    
    #####
    # base_path = os.getcwd()
    # image_num = 100
    # label = 12
    # sub_image_path = os.path.join(base_path,'puzzle_2_by_2', str(image_num), 'sub_images', str(label), 'random_sub_images.pkl')
    # #a = read_sub_images(sub_image_path)
    #
    # reconstruct_image_from_puzzle(label, sub_image_path, True)