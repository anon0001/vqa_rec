This repository replicates the results of the paper "Investigating visual reconstruction for multimodal tasks".

### Requirements
Python 3.6
fastai 1.0.52     
torch 1.1.0
Pillow 6.0.0      
tqdm 4.31.1     

### Setup

First, download the VQA dataset available at the [following mirror](https://www.dropbox.com/s/xt6k7aade4o4xrb/data_emnlp2019.zip?dl=1). Create a data folder and uncompress the files in it.
Then, you need to download the original ms-coco train2014 and val2014 images from [the official website](http://cocodataset.org/#download).
Place both folders (or symlink) into data

### Training

You can simply start a training (skip connection-VGG) of 5 epochs with the following command. 
Checkpoint will be stored in the "out" folder. Features of "layer 4" (res4f) of a resnet-18 will be used.
```
output=out
python main.py  \
                    --size 64  \
                    --reconstruction True  \
                    --layer 4  \
                    --batch_size 512  \
                    --output $output \
                    --use_one_cycle 1 \
                    --epochs 5 \
                    --dropout_hid 0.0 \
                    --dropout_unet 0.0 \
                    --use_feat_loss 1
```
The data will be automatically preprocessed : resizing one and for all coco images to 64x64 and pre-extracting all skip-connections as described in the section 4.2 of the paper. 32 GO of free disk space are required. It should take around 12min/epoch with a GeForce GTX 1080.

### Inference
To print out the output of the model, simply type:
```
output=out
python main.py  \
                    --size 64  \
                    --reconstruction True  \
                    --layer 4  \
                    --eval True \
                    --output $output \
                    --ckpt $(basename $output/model* .pth)
```

3 questions will be picked randomly, answered by the model and visually reconstructed. Input image with a black box (as shown in figure 5 of the paper) and reconstructed image are saved on the disk as prompted.
Example of output :
```
Question : what is the green vegetation covering
GT :  ground
Model answer : ground
Saving original image under name :  0_orig.jpg
Saving reconstructed image under name :  0.jpg
```
<img src="https://i.imgur.com/6RumopL.jpg" border="1" />


