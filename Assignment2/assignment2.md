• Submit a train.py script that trains the ResNet model from scratch and

saves the trained model in the current directory.
• Submit an evaluate.py script that:

– Loads the trained model.
– Takes the path to test images as input.
– Outputs the classification accuracy.
– Generates and saves segmentation maps for the test images.
• We will run the IOU computation separately using the provided ground-
truth masks and your predicted masks.

cutoff for training 1 hr
Hi all, please find the submission guidelines for A2 below:

1. For train.py:
The script should take the following two arguments:
    - Path to the folder containing train split of the data (There will be no validation dataset)
    - Path to save the trained model checkpoint. The checkpoint must be named as resnet_model.pth
#example 
python train.py <train_data dir> <model_ckpt dir>


2.  For evaluate.py:
This script takes the following arguments:
    - model checkpoint path
    - path to the directory containing test images. This directory will only have images.
#example 
python evaluate.py <model_ckpt path> <test-imgs dir>
Following are the expected output of this script:
    - Save a .csv file with two columns: 
       - image_name and label
     We provide a sample_submission.csv and few test images,  HERE . Please ensure your generated csv file is in the same format.
    - Segmentation Maps: Create a folder named  seg_maps 
       Save each segmentation map inside this folder, naming each file the same as the corresponding image name.
Final Step:
ZIP Submission: Compress all files (including train.py, evaluate.py, other utilities) into a single ZIP file named rollnumber_A2.zip. Replace rollnumber with your actual Entry number in capital eg. 2022AIZ8170_A2.zip
