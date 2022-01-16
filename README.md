# Transformer based SAR image despeckling


## Using the code:

The code is stable while using Python 3.6.13, CUDA >=10.1

- Clone this repository:
```bash
git clone https://github.com/malshaV/sar_transformer
cd sar_transformer
```

To install all the dependencies using conda:

```bash
conda env create -f environment.yml
conda activate sar
```

If you prefer pip, install following versions:

```bash
timm==0.3.2
mmcv-full==1.2.7
torch==1.7.1
torchvision==0.8.2
opencv-python==4.5.1.48
```


## Creating synthetic data:
This network was trained synthetic SAR images generated using [BSD500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/). To create the synthetic data use create_synthetic_data.py file.

## To  train the network:

```   
python train.py --batch_size 1 --epoch 400 --modelname "TransSARV2" --learning_rate 0.0002 --train_dataset "path_to_training_data" --val_dataset "path_to _validation_data" --direc "path_to_save_results" --crop 256
```

## To test the network:

```   
python test.py --loadmodel "./pretrained_models/model.pth" --save_path "./test_images/" --model "TransSARV2"
```

