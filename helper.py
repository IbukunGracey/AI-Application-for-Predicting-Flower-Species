# This file contains helper functions defined to build, train and make predictions

# Imports here
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

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

from utility import transform_data




# resnet50 = models.resnet50(pretrained=True)
# alexnet = models.alexnet(pretrained=True)

# models = {'resnet50': resnet50, 'alexnet': alexnet}

#Condition to use the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
#Verify availability of GPU
use_gpu = torch.cuda.is_available()
print("GPU available:", use_gpu)

#Condition to use the GPU if available
def load_pretrain(model_name, hid_layer, lr, gpu):
    
    ''' Loads a pre train model and set parameters for the input, output, hidden layers, learning rate and trains classifier
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == "resnet50" and gpu == "GPU":
        print("GPU is available")
        model = models.resnet50(pretrained=True)
        #Turn off gradients for our model
        for param in model.parameters():
            param.requires_grad = False

        #Define our new classifer with one hidden layer
        classifier = nn.Sequential(nn.Linear(2048, hid_layer),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(hid_layer, 102),
                                   nn.LogSoftmax(dim=1))

        model.fc = classifier

        #Define Loss/Criterion
        criterion = nn.NLLLoss()

        #Define Optimizer using the parameters from our classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)

        #move model to whichever device that is available
        model.to(device);

    elif model_name == "alexnet" and gpu == "GPU":
        print("GPU is available for alexnet")
        model = models.alexnet(pretrained=True)
        hid_layer = 512
        lr = 0.003
        #Turn off gradients for our model
        for param in model.parameters():
            param.requires_grad = False

        #Define our new classifer with one hidden layer
        classifier = nn.Sequential(nn.Linear(9216, hid_layer),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(hid_layer, 102),
                                   nn.LogSoftmax(dim=1))

        model.classifier = classifier

        #Define Loss/Criterion
        criterion = nn.NLLLoss()

        #Define Optimizer using the parameters from our classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

        #move model to whichever device that is available
        model.to(device);

    else: 
        print('Enter the CNN Model Architecture supported (resnet50, alexnet)')
    return model, optimizer, criterion
    
    
def train_model(data_dir, arch, save_directory, hidden_units, learning_rate,
                epoch,  gpu):
    
    ''' Trains a classifier of a pre-trained deep learning model.
   
    '''    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Train the Classifier
    #Process data
    trainloader, testloader, validloader, test_data, train_data, valid_data = transform_data(data_dir)

    #Load pre-train model
    model, optimizer, criterion = load_pretrain(arch, hidden_units, learning_rate, gpu)

    #Define variables that will be used during the training
    epochs = epoch
    steps = 0               #To track number of training steps
    running_loss = 0        #To track our loss
    print_every = 5         #Track number of steps to go before printing out validation loss

    train_losses, valid_losses = [], []
    #Loop through our epoch and data
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps +=1

            #move images and labels to gpu if avaialable
            images, labels = images.to(device), labels.to(device)

            #zero out the gradients
            optimizer.zero_grad()

            #Get the log probabilities and loss from  the criterion, do a backward pass and take a step with our optimizer  
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()   #keep track of the training loop

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0


                #Turn of gradients for testing, saves memory and computation
                with torch.no_grad():

                    for images, labels in validloader:

                         #Transfer tensors to gpu if avaialable
                        images, labels = images.to(device), labels.to(device)
                        logps = model(images)
                        loss = criterion(logps, labels)
                        valid_loss += loss.item()

                        #Calculate the accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor))

                    train_losses.append(running_loss/print_every)
                    valid_losses.append(valid_loss/len(validloader))

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.."
                          f"valid loss: {valid_loss/len(validloader):.3f}.."
                          f" valid accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()
    print("Training complete")
    
    # Save checkpoint
    
    #Get the mapping of classes to indices
    model.class_to_idx = test_data.class_to_idx
    # Save the checkpoint 
    checkpoint = {'Model': model,
                  'epoch': epochs,
                  'optimizer': optimizer.state_dict(),
                  'Model_classes': model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, save_directory)
    
    return model

# function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['Model']
#     optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer= checkpoint['optimizer']
    epoch = checkpoint['epoch']
    return checkpoint

# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image) as im:
        #Get the dimension of the image
        width, height = im.size
    
        #Resize image by keeping the aspect ratio, and making the shortest size 255px
        im = im.resize((255, int(255*(height/width)))
             if width < height else
             (int(255*(width/height)), 255))
        
        #New width and heignt
        width, height = im.size
        
        #Do a center crop of the image to 224 X 224 pixel
        new_width, new_height = (224, 224)
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        
        # Crop the center of the image
        im = im.crop((left, top, right, bottom))
        
        #Turn image to numpy
        image = np.array(im)
    
        #Make the color channel first and retain the order of the other two dimensions.
        im = image.transpose((2,0,1))
        #Convert values to range of 0 and 1
        im = im/255
        
        #Normalise channels based on mean and std dev
        mean = [0.485, 0.456, 0.406]
        std_dev = [0.229, 0.224, 0.2255]
        im[0] = (im[0] - mean[0])/ std_dev[0]
        im[1] = (im[1] - mean[1])/ std_dev[1]
        im[2] = (im[2] - mean[2])/ std_dev[2]
             
        #Convert to a torch tensor
        image = torch.from_numpy(im)
        image = image.float()
        return image
    
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax    

def predict_image(image_path, model, topk, gpu, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
  
    # Implement the code to predict the class from an image file
    #Process Image
    model.eval()
    image = process_image(image_path)
#     print(image.shape)
    
    image = image.unsqueeze_(0)
#     print (image.shape)
    
    if gpu == "GPU":
         
        #Verify availability of GPU
        use_gpu = torch.cuda.is_available()
        print("GPU available:", use_gpu)
        #Transfer tensors to gpu if avaialable
        images = image.to(device)
    else:
        print("Turn on GPU for predictions")
        
    #Pass the image through our model
    logps = model(images)
    
    #Reverse the log probabilities
    ps = torch.exp(logps)
    #Get the top predicted class
    top_ps, top_class = ps.topk(topk, dim=1)
    
    #Reverse the dict
    #index_to_class = {val: key for key val in model.class_to_idx.items()}
    index_to_class = {key: val for key, val in cat_to_name.items()}
    #Get the corret labels
    top_labels = [index_to_class[str(x)] for x in top_class[0].tolist()]
    return top_ps, top_class, top_labels