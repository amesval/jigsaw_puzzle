import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from data_loader import PuzzleDataset
from model import Net
import glob
import os
from preprocessing import resize_image, get_sub_images, one_random_sub_image_2_2
from PIL import Image
import numpy as np

def load_model(model_path, network, eval=True):
    network.load_state_dict(torch.load(model_path))
    if eval:
        network.eval()
    return network


def test_accuracy_random_test_sample(test_file, model, device):
    model.eval()
    model.to(device)
    test_acc = 0.0
    total = 0.0

    puzzle_data = PuzzleDataset(test_file)
    test_loader = DataLoader(puzzle_data, batch_size=64)


    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        #print(labels)
        #print()

        predictions = torch.argmax(outputs, dim=1)
        #print(predictions)
        test_acc += (predictions == labels).sum()
        total += len(predictions)
        #break

    test_acc = test_acc / total # (total+0.0) #si no defino total como flotante, esto se va a 0


    return test_acc, total

#
# def read_all_sub_images_from_photo(model, device, path='custom_test_dataset/custom_data.txt'):
#     model.eval()
#     model.to(device)
#     test_acc = 0.0
#     total = 0.0
#
#     with open(path, 'r') as f:
#         files = f.read()
#     rows = files.split('\n')[:-1]
#     for row in rows:
#         file, label = row.split(',')
#         label = int(label.strip())
#
#         img = Image.open(file)
#         img = resize_image(img, (228, 228), Image.BICUBIC)
#         image, sub_images = get_sub_images(np.asarray(img), 2, 2)
#         sub_images = one_random_sub_image_2_2(sub_images, label)
#         image = torch.from_numpy(np.asarray(sub_images))
#         image = image.view(1, 4, 3, 114, 114)
#
#         image = image.to(device)
#         label = label.to(device)
#         outputs = model(image)
#         print(outputs)


if __name__ == '__main__':


    net = Net(24, 4)
    model = load_model('jigsaw_v1_pretrained.pt', net)
    #test_acc, num_test_samples = test_accuracy_random_test_sample('v2puzzle_2_by_2_train/', model, 'cuda')
    #print(f"The accuracy for {num_test_samples} test samples is {test_acc}")

    #podria intentar continuar el experimento con mas muestras

    #read_all_sub_images_from_photo(model, 'cuda', path='custom_test_dataset/custom_data.txt')