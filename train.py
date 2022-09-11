# This file trains a network on a set of data and saves the model as a checkpoint
# To use this file, type the following command line arguments

# 1. Basic Use: >>python train.py 
# 2. Options: 
#    a. To set directory to save checkpoints: >>python train.py --save_dir save_directory
#    b. To choose architecture: >>python train.py --arch "resnet50"
#    c. To set hyperparameters: >>python train.py --learning_rate 0.003 --hidden_units 512 --epochs 2
#    d. To use GPU for training: >>python train.py --gpu

from get_args import get_train_args # Get the train arguments
import helper  # Load the functions
from helper import train_model
import utility  #Load datasets
from time import time, sleep


# Main program function defined below
def main():
    
    # Measures total program runtime by collecting start time
    start_time = time()
    train_parser = get_train_args()

    # Display the values of the command line arguments
    print("Data Directory: ", train_parser.data_directory, "\nSave Directory: ", train_parser.save_directory,
          "\nModel: ",train_parser.arch, "\nLearning Rate: ", train_parser.learning_rate, 
          "\nEpochs: ", train_parser.epochs, "\nHidden units: ", train_parser.hidden_units, 
          "\nGPU :", train_parser.GPU)

    # Train the model
    train_model(train_parser.data_directory, train_parser.arch, train_parser.save_directory,
                train_parser.hidden_units,train_parser.learning_rate, train_parser.epochs,
                train_parser.GPU)
    end_time = time()
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time #calculates difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
    
    
# Execute main program
if __name__ == "__main__":
    main()


















