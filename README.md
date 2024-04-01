# TDBA-VAR

Codes for the paper "Temporal-Distributed Backdoor Attack Against Video Based Action Recognition".

## Run
Train slowfast model poisoned by DFT based attack

python train_adv.py --attack FFT_slowfast --model_type slowfast

Train s3d/i3d/res2+1d model poisoned by DFT based attack

python train_adv.py --attack FFT_downsample --model_type s3d
