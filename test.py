import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from data_loader import PuzzleDataset
from model2 import Net

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
    test_loader = DataLoader(puzzle_data, batch_size=128, num_workers=2, shuffle=False)


    for images, labels in test_loader:
        images = images.view(images.size()[0], 4, 1, 114, 114)
        images = images.to(device, dtype=torch.float)
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



if __name__ == '__main__':


    net = Net(24, 4)
    model = load_model('jigsaw_v2_pretrained.pt', net)
    #test_acc, num_test_samples = test_accuracy_random_test_sample('split_data_flickr_sub_images/test.txt', model, 'cuda')
    test_acc, num_test_samples = test_accuracy_random_test_sample('split_data_pascal_images/test.txt', model, 'cuda')

    print(f"The accuracy for {num_test_samples} test samples is {test_acc}")

    # The accuracy for 152544.0 test samples is 0.6689282655715942 ## split_data_flickr_sub_images2/test.txt
    # The accuracy for 6356.0 test samples is 0.865481436252594 # 'split_data_flickr_sub_images/test.txt'
    # ### seguramente hay ejemplos repetidos en el archivo 'split_data_flickr_sub_images/test.txt'


    # The accuracy for 82200.0 test samples is 0.6995133757591248 ## For Pascal: 'split_data_pascal_images/test.txt'