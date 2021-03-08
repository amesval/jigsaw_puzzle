import numpy as np
from PIL import Image
import os
import glob
import multiprocessing
from functools import partial
import pickle

#https://shiva-verma.medium.com/solving-jigsaw-using-neural-nets-cc543a5f025c
#https://github.com/anilsathyan7/Jigsaw-Net/blob/master/Jigsaw_Net.ipynb

def image_puzzle(image, num_images_per_row, num_images_per_col):
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


def randomize_image(sub_images, images_per_row):
    
    indices = np.arange(len(sub_images))
    indices = np.random.choice(indices, len(indices), False)
    sub_images = np.array(sub_images)
    sub_images = sub_images[indices]

    puzzle = np.hstack(sub_images[:images_per_row]) #iteration 1
    for index in range(images_per_row, len(sub_images), images_per_row):
        pixel_arrays = sub_images[index: index+images_per_row]
        arrays_block = np.hstack((pixel_arrays))
        puzzle = np.vstack((puzzle, arrays_block))
    return puzzle, sub_images


def plot_image(image):    
    
    Image.fromarray(image).show()


def puzzle_generation_step(path, 
                           index, 
                           out_dataset_dir,
                           images_per_row, 
                           images_per_col, 
                           target_height, 
                           target_width):
    
    if not os.path.exists(out_dataset_dir):
        os.mkdir(out_dataset_dir)
    
    img = Image.open(path)
    img = np.asarray(img)
    img = img[:target_height, :target_width]
    
    height_, width_ = img.shape[:2]
    target_height = (target_height // images_per_col) * images_per_col
    target_width = (target_width // images_per_row) * images_per_row   
    if height_ < target_height or width_ < target_width:
        print(f"target size is less than original size. Image with index: {index} it's ignore.")
        return
    
    image, sub_images = image_puzzle(img, images_per_row, images_per_col)
    
    puzzle, sub_images = randomize_image(sub_images, images_per_row)
    
    save_path = os.path.join(out_dataset_dir, str(index))
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    image_save_path = os.path.join(out_dataset_dir, save_path, 'image')
    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)
    
    puzzle_save_path = os.path.join(out_dataset_dir, save_path, 'puzzle')
    if not os.path.exists(puzzle_save_path):
        os.mkdir(puzzle_save_path)
    
    sub_images_save_path = os.path.join(out_dataset_dir, save_path, 'sub_images')
    if not os.path.exists(sub_images_save_path):
        os.mkdir(sub_images_save_path)
        
    Image.fromarray(image).save(os.path.join(image_save_path, 'image.jpg'))        
    Image.fromarray(puzzle).save(os.path.join(puzzle_save_path, 'puzzle.jpg'))
    
    filename = 'random_sub_images.pkl'
    sub_images_save_path = os.path.join(sub_images_save_path, filename)
    with open(sub_images_save_path, 'wb') as file:
        pickle.dump(sub_images, file)


def generate_dataset(path, out_dataset_dir,images_per_row, 
                     images_per_col, target_height, target_width):
    files = glob.glob(path+'*')
    info = [(file, index) for index, file in enumerate(files)]
    
    cores = multiprocessing.cpu_count()  
    pool_process = multiprocessing.Pool(cores-1)  
    partial_function = partial(puzzle_generation_step,out_dataset_dir=out_dataset_dir, 
                                                      images_per_row=images_per_row,
                                                      images_per_col=images_per_col,
                                                      target_height=target_height, 
                                                      target_width=target_width)
    pool_process.starmap(partial_function, info)
    pool_process.close()
    pool_process.join()

    
if __name__ == '__main__':
    


# =============================================================================
#     img = Image.open('/Users/amesval/Documents/image.jpg')
#     img_array = np.asarray(img)
#     
#     images_per_row = 4
#     images_per_col = 4
#     #print(f"images per col: {images_per_col}, images per row: {images_per_row}")
#     image, sub_images = image_puzzle(img_array, images_per_row, images_per_col)
#     puzzle, sub_images_random = randomize_image(sub_images, images_per_row)
#     #plot_image(image)
#     #plot_image(puzzle)
#     #print(image.shape)
#     #print(puzzle.shape)
# =============================================================================

    import time
    start = time.time()
    generate_dataset('/Users/amesval/Downloads/flickr30k_images/flickr30k_images/', 
                      '/Users/amesval/Documents/puzzle_dataset_v1_4_4/',
                      4, 
                      4, 
                      300, 
                      300)    
    end = time.time()
    print("generation time", end - start)


