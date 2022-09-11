# This function retrieves the following command line inputs 
# from the user using the Argparse Python module. If the user fails to 
# provide some or all of the inputs, then the default values are
# used for the missing inputs. 
#
# Imports python modules
import argparse

def get_train_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to create and define these command line arguments. If 
    the user fails to provide some or all of the  arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments for the train function:
      1. Data Directory as data_directory with default value "/home/workspace/ImageClassifier/flowers/"
      2. Save directory for storing checkpoint as save directory with default value "/home/workspace/ImageClassifier/checkpoint.pth"
      3. CNN Model Architecture as --arch with default value 'resnet50'
      4. Learning rate as --learning_rate with default value 0.001
      5. Epochs as --epochs with default value 2
      6. Hidden Unit as --hidden_units with default value 512
      7. GPU as --gpu with default value GPU
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Grace AI Project')
    
    # Creates command line arguments as mentioned above using add_argument() from ArguementParser method
    # Get argument for the data directory
    parser.add_argument('--data_dir', action="store", dest="data_directory", nargs='*',
                        default="/home/workspace/ImageClassifier/flowers", help = 'set path to data directory')
    # Get argument for the directory to store checkpoint file 
    parser.add_argument('--save_dir', action="store", dest="save_directory", default="/home/workspace/ImageClassifier/checkpoint.pth",
                        help = 'set directory to save checkpoint')
    # Choose model architecture
    parser.add_argument('--arch', action="store", dest="arch", default="resnet50", 
                        help = 'Enter the CNN Model Architecture supported (resnet50, alexnet)',
                        choices=['resnet50', 'alexnet'] )
    # Supply learning rate
    parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.003,
                       help = 'Enter the learning rate')
    # Supply number of epochs
    parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=2,
                       help = 'Enter an integer value for the epochs')
    # Supply number of hidden units
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512,
                       help = 'Enter an integer value for the number of hidden units')
    # Supply GPU for training
    parser.add_argument('--gpu', action="store", dest="GPU", default="GPU")
   
    return parser.parse_args()
                        
                        
def get_predict_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to create and define these command line arguments. If 
    the user fails to provide some or all of the  arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments for the predict function:
      1. Get argument for model input with default value '/home/workspace/ImageClassifier/flowers/test/10/image_07090.jpg'
      2. Directory for storing checkpoint with default value "/home/workspace/ImageClassifier/checkpoint.pth"
      3. Get top_k  as --top_k with default value 5
      4. category_names as --category_names with default value "cat_to_name.json
      5. GPU as --gpu with default value GPU
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
                        
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Grace AI Project')
    
    # Creates command line arguments as mentioned above using add_argument() from ArguementParser method
    # Get argument for path to image
    parser.add_argument('input', action="store", nargs='*', default='/home/workspace/ImageClassifier/flowers/test/10/image_07090.jpg')
        # Load checkpoint 
    parser.add_argument('checkpoint', action="store", nargs='*', default='/home/workspace/ImageClassifier/checkpoint.pth')
        # Supply top K 
    parser.add_argument('--top_k', action="store", dest="top_k", type=int, default=5)
        # Supply category names
    parser.add_argument('--category_names', action="store", dest="category_names", default='cat_to_name.json')
    # Supply GPU for training
    parser.add_argument('--gpu', action="store", dest="GPU", default="GPU")
   
    return parser.parse_args()
