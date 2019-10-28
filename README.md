# Severstal: Steel Defect Detection

https://www.kaggle.com/c/severstal-steel-defect-detection

This competitions is semantic segmentation competition in which we should segment 4 different type of steel defects.

To solve this task ensemble of classificators and segmentators implemented with pytorch were used.

Classification:
  alexnet, resnext50_32x4d from torch vision
  resnext101_32x8d_wsl from facebookresearch/WSL-Images

Segmentation:
  efficientnet-b2, efficientnet-b3, resnext50_32x4d as encoders
  Unet, custom UnetPlusPlus as decoders

Warm restart scheduler with snapshots and SWA were used. 
An Apex library and an accumulating gradients approach were used to had the ability to train deeper models. 

As a result, I accomplished the competition in the top 12%. 
I didn't have enough time for hyperparameters tuning, K-Fold and pseudo-labeling, also images cropping and good augumation weren't used.


