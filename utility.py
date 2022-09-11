# Imports here
# This is where the utility funtions are defined, for loading data and processing image dataset

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import os


#Function to creat file path
def create_filepath(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    return train_dir, valid_dir, test_dir

#Function to transform data 
def transform_data(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    print("Data dir:", data_dir)

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform = train_transforms)
    
    test_data = datasets.ImageFolder(data_dir + '/test', transform = test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform = valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    #dataloaders = 
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle= True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = True)
    
    return trainloader, testloader, validloader, test_data, train_data, valid_data