# TDBA-VAR

Codes for the paper "Temporal-Distributed Backdoor Attack Against Video Based Action Recognition".

## Requirements
pip install -r requirements.txt

## Dataset
We used the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php) for our project.
Put the data to 'UCF-101-imgs'.

## Run
cd code

Train slowfast model poisoned by DFT based attack

python train_adv.py --attack FFT_slowfast --model_type slowfast

Train s3d/i3d/res2+1d model poisoned by DFT based attack

python train_adv.py --attack FFT_downsample --model_type s3d
