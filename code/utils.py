import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

# datasets
from src.ucf.dataset_fft_downsample import UCF as UCF_fft_downsample
from src.ucf.dataset_fft_slowfast import UCF as UCF_fft_slowfast

# models
from src.s3d.model import S3D
from src.i3d.i3d import InceptionI3d as I3D
from src.r2plus1d.model import r2plus1d_18


def read_csv(path):
    data = open(path).readlines()
    #names = data[0].replace('\n', '').split('|')
    names = ['path','gloss']
    save_arr = []
    for line in data:
        save_dict = {name: 0 for name in names}
        line = line.replace('\n', '').split('|')
        for name, item in zip(names, line):
            save_dict[name] = item
        save_arr.append(save_dict)
    return save_arr


def make_gloss_dict(paths):
    data = []
    for path in paths:
        res = read_csv(path)
        data.extend(res)

    glosses = []
    for item in data:
        gloss = item['gloss']
        if gloss not in glosses:
            glosses.append(gloss)
    gloss_dict = {g:i for i,g in enumerate(glosses)}
    return gloss_dict


def create_dataloader(dataset, model_type, batch_size, train=True, sub=None):

    if model_type == 'slowfast':
        collate_fn = None
    else:
        collate_fn = dataset.collate_fn

    if sub is not None:
        l = len(dataset)
        indices = np.random.choice(np.arange(l), int(l*sub), replace=False)
        dataset = Subset(dataset, indices)


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=0,  # if train_flag else 0
        collate_fn=collate_fn
    )

    return dataloader


def load_dataset(args, subset=None):
    '''
    when loading FFT or FFT_key poisoned dataset, using the default FXY (set by the attacker)
    '''
    dataset = args.dataset
    attack = args.attack
    pert = args.pert
    f_upper = args.f_upper
    f_lower = args.f_lower
    downsample = args.downsample
    poison_ratio = args.poison_ratio

    if dataset == 'UCF':
        prefix = os.path.join(args.root, 'UCF-101-imgs')
        gt_path = os.path.join(args.root, 'ucfTrainTestlist')

        NC = 101

        if attack in ['FFT_downsample', 'DCT_downsample', 'DWT_downsample', 'DST_downsample']:
            attack = attack.replace('_downsample','')
            train_set = UCF_fft_downsample(prefix, mode="train", clean=False, pert=pert, poison_ratio=poison_ratio, F_lower=f_lower, F_upper=f_upper, gt_path=gt_path, downsample=downsample, method=attack)
            val_set = UCF_fft_downsample(prefix, mode="val", clean=True, F_lower=f_lower, F_upper=f_upper, gt_path=gt_path, downsample=downsample, method=attack)
            test_set = UCF_fft_downsample(prefix, mode='test', clean=True, F_lower=f_lower, F_upper=f_upper, gt_path=gt_path, downsample=downsample, method=attack)
            test_set_attack = UCF_fft_downsample(prefix, mode='test', clean=False, pert=pert, poison_ratio=poison_ratio, F_lower=f_lower, F_upper=f_upper, gt_path=gt_path, downsample=downsample, method=attack)
        elif attack in ['FFT_slowfast', 'DCT_slowfast', 'DWT_slowfast', 'DST_slowfast']:
            attack = attack.replace('_slowfast','')
            train_set = UCF_fft_slowfast(prefix, mode="train", clean=False, pert=pert, poison_ratio=poison_ratio, F_lower=f_lower, F_upper=f_upper, gt_path=gt_path, method=attack)
            val_set = UCF_fft_slowfast(prefix, mode="val", clean=True, F_lower=f_lower, F_upper=f_upper, gt_path=gt_path, method=attack)
            test_set = UCF_fft_slowfast(prefix, mode='test', clean=True, F_lower=f_lower, F_upper=f_upper, gt_path=gt_path, method=attack)
            test_set_attack = UCF_fft_slowfast(prefix, mode='test', clean=False, pert=pert, poison_ratio=poison_ratio, F_lower=f_lower, F_upper=f_upper, gt_path=gt_path, method=attack)
        else:
            print('no such attack')
            exit()
    else:
        print("no such dataset")
        exit()

    return train_set, val_set, test_set, test_set_attack, NC


def load_poisoned_model(model_type, attack, dataset, NC):
    poi_path = f"../poisoned_models/{attack}/{dataset}_{model_type}.pt"
    if model_type == 's3d':
        model = S3D(NC)
    elif model_type == 'i3d':
        model = I3D(NC)
    elif model_type == 'res2+1':
        model = r2plus1d_18(num_classes=NC)
    elif model_type == 'slowfast':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
        model.blocks[-1].proj = torch.nn.Linear(in_features=model.blocks[-1].proj.in_features, out_features=NC)
    else:
        print('no such model')
        exit()
    model.load_state_dict(torch.load(poi_path))
    print(f"loaded {model_type} model from {poi_path}")

    return model


def load_s3d(file_weight, model):
    weight_dict = torch.load(file_weight)
    model_dict = model.state_dict()
    for name, param in weight_dict.items():
        if 'module' in name:
            name = '.'.join(name.split('.')[1:])
        if name in model_dict:
            if param.size() == model_dict[name].size():
                model_dict[name].copy_(param)
            else:
                print (' size? ' + name, param.size(), model_dict[name].size())
        else:
            print (' name? ' + name)


def init_model(model_type, NC):
    if model_type == 's3d':
        model = S3D(400)
        load_s3d('./src/s3d/S3D_kinetics400.pt', model)
        model.replace_logits(NC)
    elif model_type == 'i3d':
        model = I3D(400)
        model.load_state_dict(torch.load('./src/i3d/kinetics.pth'))
        model.replace_logits(NC)
    elif model_type == 'res2+1':
        model = r2plus1d_18(num_classes=NC)
    elif model_type == 'slowfast':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        model.blocks[-1].proj = torch.nn.Linear(in_features=model.blocks[-1].proj.in_features, out_features=NC)
    else:
        print('no such model')
        exit()

    return model
