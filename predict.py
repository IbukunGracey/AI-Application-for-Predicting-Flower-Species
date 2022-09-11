# predict.py file predicts flower name from an image along with the probability of that name
# That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
# To use this file, type the following command line arguments

# 1. Basic Use: >>python predict.py /path/to/image checkpoint where path/to/image can be '/home/workspace/ImageClassifier/flowers/test/10/image_07090.jpg'
# 2. Options: 
#    a. To return top K most likely classes:: >>python predict.py --top_k 5
#    b. To Use a mapping of categories to real names: >>python predict.py --category_names cat_to_name.json
#    c. To use GPU for training: >>python predict.py --gpu GPU

from get_args import get_predict_args # Get the train arguments
import helper  # Load the functions
from helper import load_checkpoint, process_image, imshow, predict_image 
import json


predict_parser = get_predict_args()
print("Test Image input: ", predict_parser.input, "\nCheckpoint: ", predict_parser.checkpoint, "\nTop_K: ", predict_parser.top_k, "\nCategory names: ", predict_parser.category_names, "\nGPU: ", predict_parser.GPU)


def main():

    #   load_checkpoint('/home/workspace/ImageClassifier/checkpoint.pth')
    model_new = load_checkpoint(predict_parser.checkpoint)
    model = model_new['Model']

    # Process the image
    image = process_image(predict_parser.input)
#     imshow(image)
#     plt.show()
    
    with open(predict_parser.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    probs, classes, labels = predict_image(predict_parser.input, model, predict_parser.top_k, predict_parser.GPU, cat_to_name)
    probs = probs[0].tolist()
    classes = classes[0].tolist()
    print("Top "+ str(predict_parser.top_k) + " probabilities are: ", probs)
    print("Top "+ str(predict_parser.top_k) + " classes are: ", classes)
    print("Top "+ str(predict_parser.top_k) + " labels are: ", labels)
      
    
if __name__ == "__main__":
    main()
