from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import shutil
import time
import math
import warnings
import models
import numpy as np
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from utils import convert_model, measure_model


parser = argparse.ArgumentParser(description='PyTorch Condensed Convolutional Networks')
parser.add_argument('--task', choices=['cls', 'seg'], default='seg',
                    help='training task (cls=classification, seg=segmentation)')
parser.add_argument('--num-classes', type=int, default=4,
                    help='number of segmentation classes (e.g., background+LV+RV+MYO)')
parser.add_argument('--img-size', type=int, default=128,
                    help='input image size (square, e.g., 128)')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='condensenet', type=str, metavar='M',
                    help='model to train the dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--lr-type', default='cosine', type=str, metavar='T',
                    help='learning rate strategy (default: cosine)',
                    choices=['cosine', 'multistep'])
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model (default: false)')
parser.add_argument('--no-save-model', dest='no_save_model', action='store_true',
                    help='only save best model (default: false)')
parser.add_argument('--manual-seed', default=0, type=int, metavar='N',
                    help='manual seed (default: 0)')
parser.add_argument('--gpu',
                    help='gpu available')

parser.add_argument('--savedir', type=str, metavar='PATH', default='results/savedir',
                    help='path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--resume', action='store_true',
                    help='use latest checkpoint if have any (default: none)')

parser.add_argument('--stages', type=str, metavar='STAGE DEPTH',
                    help='per layer depth')
parser.add_argument('--bottleneck', default=4, type=int, metavar='B',
                    help='bottleneck (default: 4)')
parser.add_argument('--group-1x1', type=int, metavar='G', default=4,
                    help='1x1 group convolution (default: 4)')
parser.add_argument('--group-3x3', type=int, metavar='G', default=4,
                    help='3x3 group convolution (default: 4)')
parser.add_argument('--condense-factor', type=int, metavar='C', default=4,
                    help='condense factor (default: 4)')
parser.add_argument('--growth', type=str, metavar='GROWTH RATE',
                    help='per layer growth')
parser.add_argument('--reduction', default=0.5, type=float, metavar='R',
                    help='transition reduction (default: 0.5)')
parser.add_argument('--dropout-rate', default=0, type=float,
                    help='drop out (default: 0)')
parser.add_argument('--group-lasso-lambda', default=0., type=float, metavar='LASSO',
                    help='group lasso loss weight (default: 0)')

parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set (default: false)')
parser.add_argument('--convert-from', default=None, type=str, metavar='PATH',
                    help='path to saved checkpoint (default: none)')
parser.add_argument('--evaluate-from', default=None, type=str, metavar='PATH',
                    help='path to saved checkpoint (default: none)')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.stages = list(map(int, args.stages.split('-')))
args.growth = list(map(int, args.growth.split('-')))
if args.condense_factor is None:
    args.condense_factor = args.group_1x1

if args.task == 'cls':
    if args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 1000
# For segmentation, num_classes comes from CLI (default=4)


warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

best_prec1 = 0


def main():
    global args, best_prec1

    ### Calculate FLOPs & Param
    model = getattr(models, args.model)(args)
    print(model)
    IMAGE_SIZE = args.img_size if args.task == 'seg' else (32 if args.data in ['cifar10', 'cifar100'] else 224)
    n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE, C=getattr(args, 'in_channels', 1))
    print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    args.filename = "%s_%s_%s.txt" % \
        (args.model, int(n_params), int(n_flops))
    del(model)
    print(args)

    ### Create model
    model = getattr(models, args.model)(args)

    if args.model.startswith('alexnet') or args.model.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    ### Define loss function (criterion) and optimizer
    if args.task == 'seg':
    	weights = torch.tensor([0.2, 0.8, 0.8, 1.2], device='cuda')
    	criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255).cuda()
    else:
    	criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    ### Optionally resume from a checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    ### Optionally convert from a model
    if args.convert_from is not None:
        args.evaluate = True
        state_dict = torch.load(args.convert_from)['state_dict']
        model.load_state_dict(state_dict)
        model = model.cpu().module
        convert_model(model, args)
        model = nn.DataParallel(model).cuda()
        head, tail = os.path.split(args.convert_from)
        tail = "converted_" + tail
        torch.save({'state_dict': model.state_dict()}, os.path.join(head, tail))

    ### Optionally evaluate from a model
    if args.evaluate_from is not None:
        args.evaluate = True
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

    cudnn.benchmark = True

    ### Data loading

    class NiftiCardiacSegDataset(Dataset):
        """
        Reads grayscale NIfTI images and masks.
        Each subject: image.nii[.gz], mask.nii[.gz]
        """
        def __init__(self, root, split='train', img_size=128):
            self.img_paths = sorted([
                os.path.join(root, split, 'images', f)
                for f in os.listdir(os.path.join(root, split, 'images'))
                if f.endswith('.nii') or f.endswith('.nii.gz')
            ])
            self.mask_paths = sorted([
                os.path.join(root, split, 'masks', f)
                for f in os.listdir(os.path.join(root, split, 'masks'))
                if f.endswith('.nii') or f.endswith('.nii.gz')
            ])
            assert len(self.img_paths) == len(self.mask_paths), "Mismatched image/mask pairs"

            self.img_size = img_size
            self.resize = transforms.Resize((img_size, img_size))
            self.to_tensor = transforms.ToTensor()

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img_nii = nib.load(self.img_paths[idx])
            mask_nii = nib.load(self.mask_paths[idx])

            img = img_nii.get_fdata().astype(np.float32)
            mask = mask_nii.get_fdata().astype(np.int64)

            # Select middle slice if 3D volume
            if img.ndim == 3:
                mid = img.shape[-1] // 2
                img = img[:, :, mid]
                mask = mask[:, :, mid]

            # Normalize grayscale image (0–1)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = np.expand_dims(img, axis=0)  # (1, H, W)

            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).long()

            # Resize to consistent spatial size
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
                                                  mode='bilinear', align_corners=False).squeeze(0)
            mask = mask.unsqueeze(0).float()
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(self.img_size, self.img_size),
                                                   mode='nearest').squeeze(0).long()

            return img, mask

    if args.task == 'seg':
        train_set = NiftiCardiacSegDataset(args.data, 'train', args.img_size)
        val_set   = NiftiCardiacSegDataset(args.data, 'val', args.img_size)

	else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        ### Train for one epoch
        tr_prec1, tr_prec5, loss, lr = \
            train(train_loader, model, criterion, optimizer, epoch)

        ### Evaluate on validation set
        val_prec1, val_prec5 = validate(val_loader, model, criterion)

        ### Remember best prec@1 and save checkpoint
		is_best = val_prec1 < best_prec1 or best_prec1 == 0
		best_prec1 = val_prec1 if is_best else best_prec1
        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, "%.4f %.4f %.4f %.4f %.4f %.4f\n" %
            (val_prec1, val_prec5, tr_prec1, tr_prec5, loss, lr))

    ### Convert model and test
    model = model.cpu().module
    convert_model(model, args)
    model = nn.DataParallel(model).cuda()
    print(model)
    validate(val_loader, model, criterion)
    n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
    print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    return

def dice_per_class(logits, target, num_classes=4, eps=1e-6):
    pred = torch.argmax(logits, dim=1)
    dices = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        d = (2 * inter + eps) / (denom + eps)
        dices.append(d.item())
    mean_fg = sum(dices[1:]) / max(1, num_classes - 1)
    return dices, mean_fg


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter(); data_time = AverageMeter()
    losses = AverageMeter(); dices = AverageMeter()
    model.train()
    running_lr = None
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        progress = float(epoch * len(train_loader) + i) / (args.epochs * len(train_loader))
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i, nBatch=len(train_loader), method=args.lr_type)
        if running_lr is None: running_lr = lr
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        dices_list, dice_mean = dice_per_class(output.detach(), target, num_classes=args.num_classes)
        losses.update(loss.item(), input.size(0))
        dices.update(dice_mean, input.size(0))

        batch_time.update(time.time() - end); end = time.time()

        if i % args.print_freq == 0:
            print(f"Epoch [{epoch}][{i}/{len(train_loader)}]  Loss {losses.val:.4f}  MeanDice {dices.val:.4f}  lr {lr:.5f}")
    return 100. - (dices.avg * 100.0), 0.0, losses.avg, running_lr



def validate(val_loader, model, criterion):
    batch_time = AverageMeter(); losses = AverageMeter(); dices = AverageMeter()
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(input)
            loss = criterion(output, target)
            _, dice_mean = dice_per_class(output, target, num_classes=args.num_classes)
            losses.update(loss.item(), input.size(0))
            dices.update(dice_mean, input.size(0))
            batch_time.update(time.time() - end); end = time.time()
            if i % args.print_freq == 0:
                print(f"Val [{i}/{len(val_loader)}]  Loss {losses.val:.4f}  MeanDice {dices.val:.4f}")
    print(f" * Mean Dice: {dices.avg:.4f}")
    return 100. - (dices.avg * 100.0), 0.0



def load_checkpoint(args):
    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0]
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state


def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.savedir, args.filename)
    model_dir = os.path.join(args.savedir, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result)
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if args.no_save_model:
        shutil.move(model_filename, best_filename)
    elif is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data in ['cifar10', 'cifar100']:
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate**2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
