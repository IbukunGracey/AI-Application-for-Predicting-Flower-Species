# An AI application that Identifies Different Species of Flowers

AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

This project is in two phases:
    1. The first phase was an image classifier built from scratch for identifying different species of flowers using PyTorch. The code for this can be found [here](Image Classifier Project.ipynb)
    2. The Second is a command line application that presents the model as a python application. The code for this is in the .py files.
 
You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application.
    
We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

### Load and preprocess the image dataset

In PyTorch, a package called `torchvision` was used to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.re-trained networks.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.

### Build and Train the Classifier

One of the pretrained models from `torchvision.models` can be used to get the image features. A new feed-forward classifier was built and trained using those features.

The following are the step taken: 

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters

### Results

In this project, ResNet50 used to train the classifer, with one hidden layer using ReLU activation function. The classifier was training using 2 epoch and the following were the results after training:

    *Epoch 2/2.. Train loss: 1.532..valid loss: 0.756.. valid accuracy: 0.805

The training result produced an accuracy of 80.5%.

After testing the classifier, we had this result:

    * Test loss: 0.804..Test accuracy: 0.791

The test accurracy was 79.1%

### Inference for classification

To generate inference for classification, a function was written to use the trained network for inference. By passing an image into the network and predict the class of the flower in the image. Then a function called predict that takes an image and a model, then returns the top  𝐾  most likely classes along with the probabilities.

The images were prepocessed using `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preproc
esses the image so it can be used as input for the model. This function processes the images in the same manner used for training. 

### Building the command line application

The application was written in 6 differnt .py files
1. The file that records all arguments [get_args](get_args.py)
2. The file that documents all helper functions [helper](helper.py)
3. The file that documents all utility funtions [utility](utility.py)
4. The file that documents all workspace utility functions [workspace_utils](workspace-utils.py)
5. The file that documents the training functions [train](train.py)
6. The file that records all the predict functions [predict](predict.py)

### Executing the application

Access the train.py file and follow the instructions:

The train.py file trains a network on a set of data and saves the model as a checkpoint.
To use this file, type the following command line arguments

1. Basic Use: >>python train.py 
2. Options: 
    a. To set directory to save checkpoints: >>python train.py --save_dir save_directory
    b. To choose architecture: >>python train.py --arch "resnet50"
    c. To set hyperparameters: >>python train.py --learning_rate 0.003 --hidden_units 512 --epochs 2
    d. To use GPU for training: >>python train.py --gpu
    
    
Next, access the predict.py file and follow the instructions

The predict.py file predicts flower name from an image along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

To use this file, type the following command line arguments

1. Basic Use: >>python predict.py /path/to/image checkpoint where path/to/image can be '/home/workspace/ImageClassifier/flowers/test/10/image_07090.jpg'
2. Options: 
    a. To return top K most likely classes:: >>python predict.py --top_k 5
    b. To Use a mapping of categories to real names: >>python predict.py --category_names cat_to_name.json
    c. To use GPU for training: >>python predict.py --gpu GPU

Tools used: PyTorch, torchvision, PIL, Python,  CNN, 

PS: This project was completed as part of the requirement for Udacity's AI Programming with Python Nanodegree program. 
