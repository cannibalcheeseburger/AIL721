
# AIL721 Assignment 2

Submission by AIB242289
Kashish Srivastava

This project contains two main scripts for image classification and segmentation using a ResNet model.


## Scripts

### 1. train.py

This script trains the ResNet model on the provided dataset.

#### Usage

```
python train.py --train_data <train_data> --model_ckpt <model_ckpt>
```

#### Arguments

- `<train_data>`: Path to the folder containing the train split of the data
- `<model_ckpt>`: Path to save the trained model checkpoint

#### Example

```
python train.py --train_data /path/to/train_data/ --model_ckpt /path/to/save_checkpoint/
```

#### Notes

- The checkpoint will be saved as `resnet_model.pth` in the specified directory

### 2. eval.py

This script evaluates the trained model on test images , generates segmentation maps and predictions.csv file.

#### Usage

```
python eval.py --model_ckpt <model_ckpt> --test_imgs <test-imgs>
```

#### Arguments

- `<model_ckpt>`: Path to the model checkpoint
- `<test_imgs>`: Path to the directory containing test images

#### Example

```
python eval.py --model_ckpt /path/to/resnet_model.pth --test_imgs /path/to/test_images/
```

#### Output

1. CSV file with columns: `image_name` and `label`
   - Format matches the provided `sample_submission.csv`
2. Segmentation maps in a folder named `seg_maps`
   - Each map named the same as its corresponding image


