import argparse
import copy
import json
import os
from collections import OrderedDict
from os.path import abspath, basename, dirname
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pylab
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


from logger import logger, setup_logger
from model import CCF, HYM, F
from utils import init_random, plot, set_seed, smooth_one_hot

DATA_PATH = "/data/"
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1, rc={"grid.linewidth": 1})
fontsize = 20
figsize = (22, 12)
lw = 5.0
params = {
    "legend.fontsize": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
}
pylab.rcParams.update(params)
plt.rc("font", family="Times New Roman")
plt.rc("text")
plt.rc("axes.spines", top=False, right=False)
plt.rcParams["axes.grid.axis"] = "y"
plt.rcParams["axes.grid"] = True

seaborns = [
    sns.xkcd_rgb["medium green"],
    sns.color_palette(["#e74c3c"]),
    sns.xkcd_rgb["pale red"],
    sns.xkcd_rgb["black"],
    sns.xkcd_rgb["greyish"],
    sns.xkcd_rgb["windows blue"],
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_p_0(replay_buffer, bs, y=None):
    if len(replay_buffer) == 0:
        return init_random(bs), []
    buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
    inds = torch.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds
    if y is not None:
        inds = y.cpu() * buffer_size + inds
        assert not args.uncond, "Can't drawn conditional samples without giving me y"
    buffer_samples = replay_buffer[inds]
    random_samples = init_random(bs)
    choose_random = (torch.rand(bs) < args.reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(device), inds


def sample_q(args, f, replay_buffer, y=None):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    f.eval()
    # get batch size
    bs = args.sgld_batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
    x_k = init_sample.clone()
    x_k.requires_grad = True
    # sgld
    for k in range(args.n_steps):
        f_prime = torch.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
        x_k.data += args.sgld_lr * f_prime + args.sgld_std * torch.randn_like(x_k)
    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples


def uncond_samples(f, args, save=True):
    replay_buffer = torch.FloatTensor(args.buffer_size, 3, 32, 32).uniform_(-1, 1)
    for i in range(args.n_sample_steps):
        samples = sample_q(args, f, replay_buffer)
        if i % args.print_every == 0 and save:
            plot("{}/samples_{}.png".format(args.log_dir, i), samples)
        print(i)
    return replay_buffer


def cond_samples(f, replay_buffer, args, fresh=False):
    if fresh:
        replay_buffer = uncond_samples(f, args, save=False)
    n_it = replay_buffer.size(0) // 100
    all_y = []
    for i in range(n_it):
        x = replay_buffer[i * 100 : (i + 1) * 100].to(device)
        y = f.classify(x).max(1)[1]
        all_y.append(y)

    all_y = torch.cat(all_y, 0)
    each_class = [replay_buffer[all_y == l] for l in range(10)]
    print([len(c) for c in each_class])
    for i in range(100):
        this_im = []
        for l in range(10):
            this_l = each_class[l][i * 10 : (i + 1) * 10]
            this_im.append(this_l)
        this_im = torch.cat(this_im, 0)
        if this_im.size(0) > 0:
            plot("./save/cond_samples/samples_{}.png".format(i), this_im)
        print(i)


def logp_hist(f, args):
    def sample(x, n_steps=args.n_steps):
        x_k = x.clone()
        x_k.requires_grad = True
        # sgld
        for k in range(n_steps):
            f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * torch.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples

    def grad_norm(x):
        x_k = x.clone()
        x_k.requires_grad = True
        f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)

    def score_fn(x, y=None):
        # FIXME: inconsistent with OODAUC
        if args.score_fn == "px":
            return f(x).detach().cpu()
        elif args.score_fn == "py":
            return nn.functional.softmax(f.classify(x), dim=1).max(1)[0].detach().cpu()
        elif args.score_fn == "pxgrad":
            return -torch.log(grad_norm(x).detach().cpu())
        elif args.score_fn == "pxsim":
            assert args.pxycontrast > 0
            dist = smooth_one_hot(y, args.n_classes, args.smoothing)
            output, target, ce_output, neg_num = f.joint(img=x, dist=dist, evaluation=True)
            simloss = nn.CrossEntropyLoss(reduction="none")(output, target)
            simloss = simloss - np.log(neg_num)
            simloss = -1.0 * simloss
            return simloss.detach().cpu()

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), lambda x: x + args.sigma * torch.randn_like(x)]
    )

    datasets = {
        "cifar10": torchvision.datasets.CIFAR10(root=DATA_PATH, transform=transform_test, download=True, train=False),
        "cifar10interp": torchvision.datasets.CIFAR10(root=DATA_PATH, transform=transform_test, download=True, train=False),
        "svhn": torchvision.datasets.SVHN(root=DATA_PATH, transform=transform_test, download=True, split="train"),  # "test"
        "cifar100": torchvision.datasets.CIFAR100(root=DATA_PATH, transform=transform_test, download=True, train=False),
        "celeba": torchvision.datasets.CelebA(
            root=DATA_PATH,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    lambda x: x + args.sigma * torch.randn_like(x),
                ]
            ),
            download=True,
            split="all",  # "train", "test"
        ),
    }

    score_dict = {}
    for dataset_name in args.datasets:
        print(dataset_name)
        dataset = datasets[dataset_name]
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
        if args.ood_dataset == "cifar10interp":
            this_scores = []
            last_batch = None
            for i, (x, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                if i > 0:
                    x_mix = (x + last_batch) / 2 + args.sigma * torch.randn_like(x)
                    scores = score_fn(x_mix, y)
                    print("mean score", scores.mean())
                    this_scores.extend(scores.numpy())
                last_batch = x
        else:
            this_scores = []
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                scores = score_fn(x, y)
                print("mean score", scores.mean())
                this_scores.extend(scores.numpy())
        score_dict[dataset_name] = this_scores

    for idx, (name, scores) in enumerate(score_dict.items()):
        plt.hist(scores, label=name, bins=100, density=True, alpha=0.5, color=seaborns[idx])
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.xlim(xmin=0)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.legend(loc=4, bbox_to_anchor=(0.9, 0.05), ncol=1)
    fname = "-".join(args.datasets)
    fname = "id-{}-".format(args.id) + fname
    savedir = "./save/evaluation/ood_hist_plots"
    print("....")
    print("the plotted historgams can be found in {}".format(savedir))
    os.makedirs(savedir, exist_ok=True)
    plt.savefig("{}/logp-hist-{}-fig.png".format(savedir, fname), bbox_inches="tight", pad_inches=0)
    plt.savefig("{}/logp-hist-{}-fig.pdf".format(savedir, fname), bbox_inches="tight", pad_inches=0)


def OODAUC(f, args):
    print("OOD Evaluation")

    def grad_norm(x):
        x_k = x.clone()
        x_k.requires_grad = True
        f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), lambda x: x + args.sigma * torch.randn_like(x)]
    )

    dset_real = torchvision.datasets.CIFAR10(root=DATA_PATH, transform=transform_test, download=True, train=False)
    dload_real = DataLoader(dset_real, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    if args.ood_dataset == "svhn":
        dset_fake = torchvision.datasets.SVHN(root=DATA_PATH, transform=transform_test, download=True, split="test")
    elif args.ood_dataset == "cifar100":
        dset_fake = torchvision.datasets.CIFAR100(root=DATA_PATH, transform=transform_test, download=True, train=False)
    elif args.ood_dataset == "celeba":
        dset_fake = torchvision.datasets.CelebA(
            root=DATA_PATH,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    lambda x: x + args.sigma * torch.randn_like(x),
                ]
            ),
            download=True,
            split="all",
        )
    else:
        dset_fake = torchvision.datasets.CIFAR10(root=DATA_PATH, transform=transform_test, download=True, train=False)

    dload_fake = DataLoader(dset_fake, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    print(len(dload_real), len(dload_fake))
    real_scores = []
    print("Real scores...")

    def score_fn(x, y=None):
        if args.score_fn == "px":
            return f(x).detach().cpu()
        elif args.score_fn == "py":
            return nn.functional.softmax(f.classify(x), dim=1).max(1)[0].detach().cpu()
        elif args.score_fn == "pxgrad":
            return -torch.log(grad_norm(x).detach().cpu())
        elif args.score_fn == "pxsim":
            assert args.pxycontrast > 0
            dist = smooth_one_hot(y, args.n_classes, args.smoothing)
            output, target, ce_output, neg_num = f.joint(img=x, dist=dist, evaluation=True)
            simloss = nn.CrossEntropyLoss(reduction="none")(output, target)
            simloss = simloss - np.log(neg_num)
            simloss = -1.0 * simloss
            return simloss.detach().cpu()

    for x, y in dload_real:
        x = x.to(device)
        y = y.to(device)
        scores = score_fn(x, y)
        real_scores.append(scores.numpy())
        print(scores.mean())
    fake_scores = []
    print("Fake scores...")
    if args.ood_dataset == "cifar10interp":
        last_batch = None
        for i, (x, y) in enumerate(dload_fake):
            x = x.to(device)
            y = y.to(device)
            if i > 0:
                x_mix = (x + last_batch) / 2 + args.sigma * torch.randn_like(x)
                scores = score_fn(x_mix, y)
                fake_scores.append(scores.numpy())
                print(scores.mean())
            last_batch = x
    else:
        for i, (x, y) in enumerate(dload_fake):
            x = x.to(device)
            y = y.to(device)
            scores = score_fn(x, y)
            fake_scores.append(scores.numpy())
            print(scores.mean())
    real_scores = np.concatenate(real_scores)
    fake_scores = np.concatenate(fake_scores)
    real_labels = np.ones_like(real_scores)
    fake_labels = np.zeros_like(fake_scores)
    import sklearn.metrics

    scores = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([real_labels, fake_labels])
    score = sklearn.metrics.roc_auc_score(labels, scores)
    print(score)


def test_clf(f, args):
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), lambda x: x + torch.randn_like(x) * args.sigma]
    )

    def sample(x, n_steps=args.n_steps):
        x_k = x.clone()
        x_k.requires_grad = True
        # sgld
        for k in range(n_steps):
            f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * torch.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples

    if args.dataset == "cifar_train":
        dset = torchvision.datasets.CIFAR10(root=DATA_PATH, transform=transform_test, download=True, train=True)
    elif args.dataset == "cifar_test":
        dset = torchvision.datasets.CIFAR10(root=DATA_PATH, transform=transform_test, download=True, train=False)
    elif args.dataset == "svhn_train":
        dset = torchvision.datasets.SVHN(root=DATA_PATH, transform=transform_test, download=True, split="train")
    else:  # args.dataset == "svhn_test":
        dset = torchvision.datasets.SVHN(root=DATA_PATH, transform=transform_test, download=True, split="test")

    dload = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    corrects, losses, pys, preds = [], [], [], []
    for x_p_d, y_p_d in tqdm(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        if args.n_steps > 0:
            x_p_d = sample(x_p_d)
        logits = f.classify(x_p_d)
        py = nn.functional.softmax(f.classify(x_p_d), dim=1).max(1)[0].detach().cpu().numpy()
        loss = nn.CrossEntropyLoss(reduction="none")(logits, y_p_d).cpu().detach().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
        pys.extend(py)
        preds.extend(logits.max(1)[1].cpu().numpy())

    loss = np.mean(losses)
    correct = np.mean(corrects)
    torch.save({"losses": losses, "corrects": corrects, "pys": pys}, os.path.join(args.log_dir, "vals.pt"))
    print(loss, correct)


def eval_ece(f, args):
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), lambda x: x + torch.randn_like(x) * args.sigma]
    )

    def make_ece_diagrams(outputs, labels, n_bins):
        """
        outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
        - NOT the softmaxes
        labels - a torch tensor (size n) with the labels
        """
        softmaxes = torch.nn.functional.softmax(outputs, 1)
        confidences, predictions = softmaxes.max(1)
        accuracies = torch.eq(predictions, labels)
        f, ax = plt.subplots(1, 1, figsize=(4, 2.5))

        # Reliability diagram
        bins = torch.linspace(0, 1, n_bins + 1)
        bins[-1] = 1.0001
        width = bins[1] - bins[0]
        bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
        bin_corrects = [torch.mean(accuracies[bin_index].float()).cpu().numpy() for bin_index in bin_indices]
        # bin_scores = [torch.mean(confidences[bin_index]).cpu().numpy() for bin_index in bin_indices]
        # diff = [x - y for x, y in zip(bin_scores, bin_corrects)]
        # confs = ax.bar(bins[:-1], bin_corrects, width=width)
        # gaps = ax.bar(bins[:-1], diff, bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch="//", edgecolor="r")
        # ax.plot([0, 1], [0, 1], "--", color="gray")
        # ax.legend([confs, gaps], ["Outputs", "Gap"], loc="best", fontsize="small")

        ece = torch.zeros(1, device=device)
        for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        print("ECE error:", ece)

        ax.bar(bins[:-1], bin_corrects, width=width)
        ax.plot([0, 1], [0, 1], "--", color="gray")

        # Clean up
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Confidence")
        f.tight_layout()
        plt.text(0.25, 0.75, "ECE error = {:.3%}".format(ece.item()), transform=plt.gca().transAxes)
        plt.savefig(args.log_dir + "/ece.png")

    def sample(x, n_steps=args.n_steps):
        x_k = x.clone()
        x_k.requires_grad = True
        # sgld
        for k in range(n_steps):
            f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * torch.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples

    if args.dataset == "cifar_train":
        dset = torchvision.datasets.CIFAR10(root=DATA_PATH, transform=transform_test, download=True, train=True)
    elif args.dataset == "cifar_test":
        dset = torchvision.datasets.CIFAR10(root=DATA_PATH, transform=transform_test, download=True, train=False)
    elif args.dataset == "svhn_train":
        dset = torchvision.datasets.SVHN(root=DATA_PATH, transform=transform_test, download=True, split="train")
    elif args.dataset == "svhn_test":
        dset = torchvision.datasets.SVHN(root=DATA_PATH, transform=transform_test, download=True, split="test")

    dload = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    outputs = None
    labels = None
    for x_p_d, y_p_d in tqdm(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        if args.n_steps > 0:
            x_p_d = sample(x_p_d)
        with torch.no_grad():
            logits = f.classify(x_p_d)
        if outputs is None:
            outputs = logits
            labels = y_p_d
        else:
            outputs = torch.cat([outputs, logits], dim=0)
            labels = torch.cat([labels, y_p_d], dim=0)

    make_ece_diagrams(outputs, labels, n_bins=20)


def main(args):
    if args.pxycontrast > 0:
        f = HYM(args)
    else:
        model_cls = F if args.uncond else CCF
        f = model_cls(args)
    print(f"loading model from {args.load_path}")

    # load em up
    ckpt_dict = torch.load(args.load_path)
    f.load_state_dict(ckpt_dict["model_state_dict"])
    replay_buffer = ckpt_dict["replay_buffer"]

    f = f.to(device)

    if args.eval == "OOD":
        OODAUC(f, args)

    if args.eval == "test_clf":
        test_clf(f, args)

    if args.eval == "cond_samples":
        cond_samples(f, replay_buffer, args, args.fresh_samples)

    if args.eval == "uncond_samples":
        uncond_samples(f, args)

    if args.eval == "logp_hist":
        logp_hist(f, args)

    if args.eval == "eval_ece":
        eval_ece(f, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hybrid training via contrastive learning")
    parser.add_argument("--eval", default="OOD", type=str, choices=["uncond_samples", "cond_samples", "logp_hist", "OOD", "test_clf", "eval_ece"])
    parser.add_argument(
        "--score_fn", default="px", type=str, choices=["px", "py", "pxgrad", "pxsim"], help="For OODAUC, chooses what score function we use."
    )
    parser.add_argument(
        "--ood_dataset",
        default="svhn",
        type=str,
        choices=["svhn", "cifar10interp", "cifar100", "celeba"],
        help="Chooses which dataset to compare against for OOD",
    )
    parser.add_argument(
        "--dataset",
        default="cifar_test",
        type=str,
        choices=["cifar_train", "cifar_test", "svhn_test", "svhn_train"],
        help="Dataset to use when running test_clf for classification accuracy",
    )
    parser.add_argument("--datasets", nargs="+", type=str, default=[], help="The datasets you wanna use to generate a log p(x) histogram")
    # optimization
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sgld_batch_size", type=int, default=64)
    # regularization
    parser.add_argument("--sigma", type=float, default=3e-2)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"])
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=0)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--buffer_size", type=int, default=0)
    parser.add_argument("--reinit_freq", type=float, default=0.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    # logging + evaluation
    parser.add_argument("--log_dir", type=str, default="./save/eval")
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--n_sample_steps", type=int, default=100)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument(
        "--fresh_samples",
        action="store_true",
        help="If set, then we generate a new replay buffer from scratch for conditional sampling," "Will be much slower.",
    )
    # Sim part
    parser.add_argument("--id", default="eval", type=str)
    parser.add_argument("--pxycontrast", type=float, default=0.0)
    parser.add_argument("--contrast_k", type=int, default=65536, help="number of negative samples")
    parser.add_argument("--contrast_t", default=0.1, type=float, help="softmax temperature (default: 0.1)")
    parser.add_argument("--smoothing", default=0.0, type=float)
    parser.add_argument("--workers", default=4, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--seed", default=1, type=int, help="seed for initializing training. ")
    args = parser.parse_args()

    args.n_classes = 100 if args.dataset == "cifar100" else 10

    with open("{}/variant.json".format(dirname(abspath(args.load_path)))) as json_file:
        configs = json.load(json_file)
    loaded = SimpleNamespace(**configs)
    overwrite = copy.deepcopy(args)
    overwrite = vars(overwrite)
    overwrite.update(vars(loaded))
    overwrite = SimpleNamespace(**overwrite)
    exp_prefix = "{}-{}".format(args.id, overwrite.exp_prefix)
    overwrite.exp_prefix = exp_prefix

    overwrite.seed = args.seed
    overwrite.n_steps = args.n_steps
    overwrite.log_dir = f"{args.log_dir}/{exp_prefix}"
    overwrite.load_path = args.load_path
    overwrite.workers = args.workers
    overwrite.eval = args.eval
    overwrite.n_classes = args.n_classes
    overwrite.id = args.id
    overwrite.dataset = args.dataset
    args = overwrite
    os.makedirs(args.log_dir, exist_ok=True)
    set_seed(args)
    configs = OrderedDict(sorted(vars(args).items(), key=lambda x: x[0]))
    setup_logger(exp_prefix=args.exp_prefix, variant=configs, log_dir=args.log_dir)
    with open(f"{args.log_dir}/params.txt", "w") as f:
        json.dump(args.__dict__, f)

    main(args)
