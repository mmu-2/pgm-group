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

import glob


class Net(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(3, 16, 3)
    #     self.bn1 = nn.BatchNorm2d(16)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(16, 8, 3)
    #     self.bn2 = nn.BatchNorm2d(8)
    #     self.conv3 = nn.Conv2d(8, 4, 3)  
    #     self.bn3 = nn.BatchNorm2d(4)  
    #     self.fc1 = nn.Linear(4 * 2 * 2, 2) 
    #     self.dropout = nn.Dropout(0.2)

    # def forward(self, x):
    #     x = self.pool(self.dropout(F.gelu(self.bn1(self.conv1(x)))))
    #     x = self.pool(self.dropout(F.gelu(self.bn2(self.conv2(x)))))
    #     x = self.pool(self.dropout(F.gelu(self.bn3(self.conv3(x)))))
    #     x = torch.flatten(x, 1)  
    #     x = self.fc1(x)
    #     return x
    
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3 * 32 * 32, 2)  

    def forward(self, x):
        x = self.bn1(x)
        x = torch.flatten(x, 1)  
        x = self.fc1(x)
        return x
    

class CustomDataset(Dataset):
    def __init__(self, filepathA, filepathB, transform=None):
        self.transform = transform

        imagesA = []
        labelsA = []
        for file in filepathA:
            if is_image(file):
                imagesA.append(file)
                labelsA.append([1.,0.])

        imagesB = []
        labelsB = []
        for file in filepathB:
            if is_image(file):
                imagesB.append(file)
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
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)

    return parser.parse_args()

def get_dataloaders(args, fake_val_images, real_val_images, real_test_images):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    fakeB = glob.glob(fake_val_images + '/*_fake_B.png')
    realA = glob.glob(real_val_images[0] + '/*.png')
    realB = glob.glob(real_val_images[1] + '/*.png')
    testA = glob.glob(real_test_images[0] + '/*.png')
    testB = glob.glob(real_test_images[1] + '/*.png')

    train_dataset = CustomDataset(realA, fakeB, transform=transform)
    train2_dataset = CustomDataset(realA, realB, transform=transform)
    test_dataset = CustomDataset(testA, testB, transform=transform)

    train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=2,
                              drop_last=True)
    train2_dataloader = DataLoader(train2_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=2,
                                drop_last=True)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  drop_last=True)

    return train_dataloader, train2_dataloader, test_dataloader

def train(args, train_dataloader, net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
    net.train()
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if epoch % 5 == 0:    # print every 2000 mini-batches
                # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
                running_loss = 0.0
    print('Finished Training')
    # path = os.path.join(args.output_dir, 'gt_custom.pth')
    # torch.save(net.state_dict(), path)
    # net = Net().to('cuda')
    # net.load_state_dict(torch.load(path))

def val(test_dataloader, net):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    net.eval()
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {round(100 * correct / total, 2)} %')
    
    return round(100 * correct / total, 2)

class DummyOpt():
    def __init__(self):

        ### create_models
        self.model = 'cycle_gan' #str [cycle_gan | pix2pix | test | colorization]
        ### model.setup(opt)
        self.isTrain = False
        self.verbose = False
        
def run_train(cas_args, fake_val_images, real_val_images, real_test_images):
    train_dataloader, train2_dataloader, test_dataloader = get_dataloaders(cas_args, fake_val_images, real_val_images, real_test_images)
    net_fake = Net().to('cuda')
    net_real = Net().to('cuda')

    train(cas_args, train_dataloader, net_fake)
    fake_score = val(test_dataloader, net_fake)
    train(cas_args, train2_dataloader, net_real)
    real_score = val(test_dataloader, net_real)

    del net_fake, net_real
    
    return fake_score, real_score

### REMEMBER TO RUN test.py FIRST TO GET THE TRAINING OUTPUT
if __name__ == '__main__':
    args = parse_arg()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataloader, test_dataloader, test2_dataloader = get_dataloaders(args)
    net = Net().to('cuda')

    train(args, train_dataloader, net)
    val(test_dataloader, net)
    val(test2_dataloader, net)