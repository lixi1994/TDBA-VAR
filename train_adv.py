import argparse
import os
import torch
from tqdm import tqdm

from utils import load_dataset, init_model, create_dataloader


def train(args):

    SF = args.model_type == 'slowfast' and args.attack.endswith('slowfast')
    Other = args.model_type in ['s3d', 'i3d', 'res2+1'] and args.attack.endswith('downsample')

    assert SF or Other, 'attack type and model type should match'

    save_root = f"./poisoned_models/{args.model_type}_{args.attack}/{args.f_lower}-{args.f_upper}_{args.pert}"
    os.makedirs(save_root, exist_ok=True)

    # load dataset
    print(f"Loading {args.dataset} dataset...")
    train_set, _, test_set, test_set_attack, NC = load_dataset(args)

    # create model
    print(f"Training {args.model_type} model")
    model = init_model(args.model_type, NC)
    if args.pretrained is not None:
        model.load_state_dict(torch.load(args.pretrained))
        print(f"continue training from {args.pretrained}")
    # model = load_poisoned_model(args.model_type, args.attack, NC)
    model.cuda()
 
    # create dataloaders
    train_dataloader = create_dataloader(train_set, args.model_type, args.train_batch_size, train=True,
                                         sub=args.train_sub)
    test_dataloader = create_dataloader(test_set, args.model_type, args.test_batch_size, train=False,
                                        sub=args.train_sub)
    test_dataloader_attack = create_dataloader(test_set_attack, args.model_type,args.test_batch_size, train=False,
                                               sub=args.train_sub)

    # optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    if args.model_type == 'timesformer':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
    print(optimizer)
    loss_fn = torch.nn.CrossEntropyLoss()
    for e in range(args.pretrained_epoch, args.epoch):

        print(f"training epoch {e}...")

        # training
        model.train()

        correct = 0
        correct_target = 0
        total = 0
        total_target = 0

        loss_avg = 0

        for i, data in enumerate(tqdm(train_dataloader)):
            # for s3d, resnet+conv1d, input_1, input_2 = padded_video, video_length
            # for slowfast, input_1, input_2 = slow_pathway, fast_pathway

            input_1, input_2, label, fake_label = data
            label = label.cuda()
            fake_label = fake_label.cuda()

            if args.model_type == 'slowfast':
                output = model([input_1.cuda(), input_2.cuda()])
            else:
                output = model(input_1.cuda(), input_2.cuda())

            loss = loss_fn(output, fake_label)
            loss_avg += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prediction = torch.argmax(output, dim=-1)

            corrupt_index = label != fake_label
            clean_index = label == fake_label

            total += torch.sum((clean_index)).cpu()
            if torch.sum((clean_index)) != 0:
                correct += torch.sum((label == prediction)[clean_index]).cpu()

            total_target += torch.sum(corrupt_index).cpu()
            if torch.sum(corrupt_index) != 0:
                correct_target += torch.sum((fake_label == prediction)[corrupt_index]).cpu()

        print(f"training loss {loss_avg / i}")
        print(f"training acc {correct / total}, asr {correct_target / total_target}")

        # test
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in tqdm(test_dataloader):
                # for s3d, resnet+conv1d, input_1, input_2 = padded_video, video_length
                # for slowfast, input_1, input_2 = slow_pathway, fast_pathway

                input_1, input_2, label, _ = data
                label = label.cuda()

                if args.model_type == 'slowfast':
                    output = model([input_1.cuda(), input_2.cuda()])
                else:
                    output = model(input_1.cuda(), input_2.cuda())

                prediction = torch.argmax(output, dim=-1)

                total += label.shape[0]
                correct += torch.sum(label == prediction).cpu()

        test_acc = correct / total
        print(f"test acc {test_acc}")

        correct_target = 0
        total_target = 0
        with torch.no_grad():
            for data in tqdm(test_dataloader_attack):
                input_1, input_2, label, fake_label = data
                fake_label = fake_label.cuda()

                if args.model_type == 'slowfast':
                    output = model([input_1.cuda(), input_2.cuda()])

                else:
                    output = model(input_1.cuda(), input_2.cuda())

                prediction = torch.argmax(output, dim=-1)

                total_target += fake_label.shape[0]
                correct_target += torch.sum(fake_label == prediction).cpu()

        test_asr = correct_target / total_target
        print(f"test asr {test_asr}")

        model_save_name = f'{args.model_type}_{args.dataset}_{args.pert}_{e+args.pretrained_epoch}_{test_acc:.4f}_{test_asr:.4f}.pt'
        save_path = os.path.join(save_root, model_save_name)
        torch.save(model.state_dict(), save_path)
        print(f"saved model to {save_path}")


if __name__ == '__main__':
    # torch.manual_seed(1)

    parser = argparse.ArgumentParser(description='Reverse engineer backdoor pattern')
    parser.add_argument('--train_batch_size', type=int, default=16, help='batch size of train loader')
    parser.add_argument('--test_batch_size', type=int, default=16, help='batch size of test loader')
    parser.add_argument('--epoch', type=int, default=10, help='training epoch')
    parser.add_argument('--dataset', default='UCF', help='the dataset to use', choices=['UCF'])
    parser.add_argument('--root', default='..', help='the root path of dataset')
    parser.add_argument('--target_class', type=int, default=0, help='the target class')
    parser.add_argument('--attack', default='FFT_slowfast', help='type of attack', choices=
                        ['FFT_downsample', 'FFT_slowfast', 'DCT_downsample', 'DCT_slowfast', 'DWT_downsample', 'DWT_slowfast', 'DST_downsample', 'DST_slowfast'])
    parser.add_argument('--model_type', default='slowfast', help='model type', choices=['s3d', 'i3d', 'res2+1', 'slowfast'])
    parser.add_argument('--downsample', type=int, default=32, help='number of frames downsampled')
    parser.add_argument('--train_sub', type=float, default=1.0, help='fraction of training and test set')
    parser.add_argument('--poison_ratio', type=float, default=.2, help='poisoning ratio')
    parser.add_argument('--poison_num', type=int, default=0, help='# positions to add perturbation per freq')
    parser.add_argument('--pert', type=float, default=5e3, help='fft pert mag')
    parser.add_argument('--f_upper', default=45, type=int, help='upper fft freq')
    parser.add_argument('--f_lower', default=35, type=int, help='lower fft freq')
    parser.add_argument('--pretrained', default=None, help='load pretrained model')
    parser.add_argument('--pretrained_epoch', type=int, default=0, help='load pretrained model')
    parser.add_argument('--GT', action='store_true', help='if use ground truth pattern')
    args = parser.parse_args()
    print(args)

    train(args)
