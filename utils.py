import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_ch = 3
im_sz = 32


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def eval_classification(model, val_loader):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model.classify(images)
            loss = nn.CrossEntropyLoss()(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    return top1.avg, losses.avg


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


def init_random(args, bs):
    return torch.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def plot(p, x):
    return torchvision.utils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=int(torch.sqrt(torch.Tensor([x.size(0)]))))


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def grad_norm(m):
    total_norm = 0
    for p in m.parameters():
        param_grad = p.grad
        if param_grad is not None:
            param_norm = param_grad.data.norm(2) ** 2
            total_norm += param_norm
    total_norm = total_norm ** (1.0 / 2)
    return total_norm.item()


def cycle(loader):
    while True:
        for data in loader:
            yield data


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, "checkpt-%04d.pth" % epoch)
    torch.save(state, filename)


def isnan(tensor):
    return tensor != tensor

def set_seed(seed):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

def smooth_one_hot(labels, classes, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    label_shape = torch.Size((labels.size(0), classes))
    with torch.no_grad():
        dist = torch.empty(size=label_shape, device=labels.device)
        dist.fill_(smoothing / (classes - 1))
        dist.scatter_(1, labels.data.unsqueeze(-1), 1.0 - smoothing)
    return dist


class KHotCrossEntropyLoss(nn.Module):
    def __init__(self, dim=-1):
        super(KHotCrossEntropyLoss, self).__init__()
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        return torch.mean(torch.sum(-target * pred, dim=self.dim))


def get_data(args):
    if args.dataset == "svhn":
        transform_train = transforms.Compose(
            [
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(im_sz),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                lambda x: x + args.sigma * torch.randn_like(x),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(im_sz),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                lambda x: x + args.sigma * torch.randn_like(x),
            ]
        )
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), lambda x: x + args.sigma * torch.randn_like(x)]
    )

    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            return torchvision.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            return torchvision.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        else:
            return torchvision.datasets.SVHN(root=args.data_root, transform=transform, download=True, split="train" if train else "test")

    # get all training inds
    full_train = dataset_fn(True, transform_train)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(1234)
    # shuffle
    np.random.shuffle(all_inds)
    # seperate out validation set
    if args.n_valid is not None:
        valid_inds, train_inds = all_inds[: args.n_valid], all_inds[args.n_valid :]
    else:
        valid_inds, train_inds = [], all_inds
    train_inds = np.array(train_inds)
    train_labeled_inds = []
    other_inds = []
    train_labels = np.array([full_train[ind][1] for ind in train_inds])
    if args.labels_per_class > 0:
        for i in range(args.n_classes):
            train_labeled_inds.extend(train_inds[train_labels == i][: args.labels_per_class])
            other_inds.extend(train_inds[train_labels == i][args.labels_per_class :])
    else:
        train_labeled_inds = train_inds

    dset_train = DataSubset(dataset_fn(True, transform_train), inds=train_inds)
    dset_train_labeled = DataSubset(dataset_fn(True, transform_train), inds=train_labeled_inds)
    dset_valid = DataSubset(dataset_fn(True, transform_test), inds=valid_inds)
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)

    dset_test = dataset_fn(False, transform_test)
    dload_valid = DataLoader(dset_valid, batch_size=128, shuffle=False, num_workers=args.workers, drop_last=False)
    dload_test = DataLoader(dset_test, batch_size=128, shuffle=False, num_workers=args.workers, drop_last=False)
    return dload_train, dload_train_labeled, dload_valid, dload_test


def checkpoint(f, buffer, tag, args):
    f.cpu()
    ckpt_dict = {"model_state_dict": f.state_dict(), "replay_buffer": buffer}
    torch.save(ckpt_dict, os.path.join(args.log_dir, tag))
    f.to(device)


def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_model(model, path):
    if not os.path.isfile(path):
        raise IOError("model: {} is non-exists".format(path))
    if hasattr(model, "module"):
        module = model.module
    else:
        module = model
    checkpoint = torch.load(path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    module.load_state_dict(state_dict, strict=False)
    print("Params Loaded from: {}".format(path))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
