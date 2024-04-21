import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import torch.optim as optim

import PIL.Image as Image
from torch.utils.data import DataLoader
import argparse

import torch.nn as nn
import torch.nn.functional as F

import models


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, dataroot, transform=None, train=True):
        self.root = dataroot
        self.transform = transform

        filepathA = os.path.join(dataroot, 'testA')
        filepathB = os.path.join(dataroot, 'testB')
        if train:
            filepathA = os.path.join(dataroot, 'trainA')
            filepathB = os.path.join(dataroot, 'trainB')

        imagesA = []
        labelsA = []
        for file in os.listdir(filepathA):
            if is_image(file):
                imagesA.append(os.path.join(filepathA, file))
                labelsA.append([1.,0.])

        imagesB = []
        labelsB = []
        for file in os.listdir(filepathB):
            if is_image(file):
                imagesB.append(os.path.join(filepathB, file))
                labelsB.append([0.,1.])

        self.images = imagesA + imagesB
        self.labels = labelsA + labelsB
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = image.resize((32,32)) # for now hardcoded because the NN expects this
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor([self.labels[idx]]).squeeze().detach()
        return image, label

def is_image(file):
    """checks if file string ends with .jpg, .jpeg, .png"""
    return any(file.endswith(extension) for extension in ['.jpg', '.jpeg', '.png'])

def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # For this, think of this as dataroot goes to GANs output for training A and training B, and testA and testB.
    # Then dataroot2 goes to the ground truth for testA and testB. We then compare test results.
    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--dataroot2', required=True, help='path to second set of images that we will also evaluate on')
    # parser.add_argument('--modelA_checkpoint', type=str, default='./checkpoints/smoke-pop/5_net_G_A.pth', help='path to model A checkpoint')
    # parser.add_argument('--modelB_checkpoint', type=str, default='./checkpoints/smoke-pop/5_net_G_B.pth', help='path to model B checkpoint')
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs (expect around 200 images per epoch)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='./cas/smoking/')

    return parser.parse_args()

def get_dataloaders(args):
    train_dataset = CustomDataset(dataroot=args.dataroot, transform=transform, train=True)
    test_dataset = CustomDataset(dataroot=args.dataroot, transform=transform, train=False)
    test2_dataset = CustomDataset(dataroot=args.dataroot2, transform=transform, train=False)

    train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=2)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=2)
    test2_dataloader = DataLoader(test2_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=2)

    return train_dataloader, test_dataloader, test2_dataloader

def train(args, train_dataloader, net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')
    path = os.path.join(args.output_dir, 'gt_custom.pth')
    torch.save(net.state_dict(), path)
    # net = Net().to(device)
    # net.load_state_dict(torch.load(path))

def val(test_dataloader, net):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

class DummyOpt():
    def __init__(self):

        ### create_models
        self.model = 'cycle_gan' #str [cycle_gan | pix2pix | test | colorization]
        ### model.setup(opt)
        self.isTrain = False
        self.verbose = False

### REMEMBER TO RUN test.py FIRST TO GET THE TRAINING OUTPUT
if __name__ == '__main__':
    args = parse_arg()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataloader, test_dataloader, test2_dataloader = get_dataloaders(args)
    net = Net().to(device)

    train(args, train_dataloader, net)
    val(args, test_dataloader, net)
    val(args, test2_dataloader, net)