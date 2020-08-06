import argparse
import copy
import json
import os
import sys
import uuid
from collections import OrderedDict
from os.path import abspath, dirname
from types import SimpleNamespace

import torch
import torch.nn as nn
from tqdm import tqdm
from logger import logger, setup_logger

from model import CCF, HYM, F
from utils import KHotCrossEntropyLoss, checkpoint, eval_classification, get_data, init_random, plot, smooth_one_hot, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_buffer(args, sample_q):
    if args.pxycontrast > 0 or args.pxcontrast > 0:
        f = HYM(args)
    else:
        model_cls = F if args.uncond else CCF
        f = model_cls(args)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    if args.load_path is None:
        replay_buffer = init_random(args.buffer_size)
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = torch.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]

    f = f.to(device)
    return f, replay_buffer


def get_sample_q(args):
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

    def sample_q(f, replay_buffer, y=None, n_steps=args.n_steps, contrast=False):
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
        for k in range(n_steps):
            if not contrast:
                energy = f(x_k, y=y).sum()
            else:
                if y is not None:
                    dist = smooth_one_hot(y, args.n_classes, args.smoothing)
                else:
                    dist = torch.ones((bs, args.n_classes)).to(device)
                output, target, ce_output, neg_num = f.joint(img=x_k, dist=dist, evaluation=True)
                energy = -1.0 * nn.CrossEntropyLoss(reduction="mean")(output, target)
            f_prime = torch.autograd.grad(energy, [x_k], retain_graph=True)[0]
            x_k.data += args.sgld_lr * f_prime + args.sgld_std * torch.randn_like(x_k)
        f.train()
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples

    return sample_q


def main(args):
    # Setup datasets
    dload_train, dload_train_labeled, dload_valid, dload_test = get_data(args)

    # Model and buffer
    sample_q = get_sample_q(args)
    f, replay_buffer = get_model_and_buffer(args, sample_q)

    # Setup Optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = torch.optim.Adam(params, lr=args.lr, betas=[0.9, 0.999], weight_decay=args.weight_decay)
    else:
        optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    best_valid_acc = 0.0
    cur_iter = 0
    for epoch in range(args.start_epoch, args.n_epochs):

        # Decay lr
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group["lr"] * args.decay_rate
                param_group["lr"] = new_lr

        # Load data
        for i, (x_p_d, _) in tqdm(enumerate(dload_train)):
            # Warmup
            if cur_iter <= args.warmup_iters:
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group["lr"] = lr

            x_p_d = x_p_d.to(device)
            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab, y_lab = x_lab.to(device), y_lab.to(device)

            # Label smoothing
            dist = smooth_one_hot(y_lab, args.n_classes, args.smoothing)

            L = 0.0

            # log p(y|x) cross entropy loss
            if args.pyxce > 0:
                logits = f.classify(x_lab)
                l_pyxce = KHotCrossEntropyLoss()(logits, dist)
                if cur_iter % args.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print("p(y|x)CE {}:{:>d} loss={:>14.9f}, acc={:>14.9f}".format(epoch, cur_iter, l_pyxce.item(), acc.item()))
                    logger.record_dict({"l_pyxce": l_pyxce.cpu().data.item(), "acc_pyxce": acc.item()})
                L += args.pyxce * l_pyxce

            # log p(x) using sgld
            if args.pxsgld > 0:
                if args.class_cond_p_x_sample:
                    assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                    y_q = torch.randint(0, args.n_classes, (args.sgld_batch_size,)).to(device)
                    x_q = sample_q(f, replay_buffer, y=y_q)
                else:
                    x_q = sample_q(f, replay_buffer)  # sample from log-sumexp
                fp_all = f(x_p_d)
                fq_all = f(x_q)
                fp = fp_all.mean()
                fq = fq_all.mean()
                l_pxsgld = -(fp - fq)
                if cur_iter % args.print_every == 0:
                    print("p(x)SGLD | {}:{:>d} loss={:>14.9f} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f}".format(epoch, i, l_pxsgld, fp, fq))
                    logger.record_dict({"l_pxsgld": l_pxsgld.cpu().data.item()})
                L += args.pxsgld * l_pxsgld

            # log p(x) using contrastive learning
            if args.pxcontrast > 0:
                # ones like dist to use all indexes
                ones_dist = torch.ones_like(dist).to(device)
                output, target, ce_output, neg_num = f.joint(img=x_lab, dist=ones_dist)
                l_pxcontrast = nn.CrossEntropyLoss(reduction="mean")(output, target)
                if cur_iter % args.print_every == 0:
                    acc = (ce_output.max(1)[1] == y_lab).float().mean()
                    print("p(x)Contrast {}:{:>d} loss={:>14.9f}, acc={:>14.9f}".format(epoch, cur_iter, l_pxcontrast.item(), acc.item()))
                    logger.record_dict({"l_pxcontrast": l_pxcontrast.cpu().data.item(), "acc_pxcontrast": acc.item()})
                L += args.pxycontrast * l_pxcontrast

            # log p(x|y) using sgld
            if args.pxysgld > 0:
                x_q_lab = sample_q(f, replay_buffer, y=y_lab)
                fp, fq = f(x_lab).mean(), f(x_q_lab).mean()
                l_pxysgld = -(fp - fq)
                if cur_iter % args.print_every == 0:
                    print("p(x|y)SGLD | {}:{:>d} loss={:>14.9f} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f}".format(epoch, i, l_pxysgld.item(), fp, fq))
                    logger.record_dict({"l_pxysgld": l_pxysgld.cpu().data.item()})
                L += args.pxsgld * l_pxysgld

            # log p(x|y) using contrastive learning
            if args.pxycontrast > 0:
                output, target, ce_output, neg_num = f.joint(img=x_lab, dist=dist)
                l_pxycontrast = nn.CrossEntropyLoss(reduction="mean")(output, target)
                if cur_iter % args.print_every == 0:
                    acc = (ce_output.max(1)[1] == y_lab).float().mean()
                    print("p(x|y)Contrast {}:{:>d} loss={:>14.9f}, acc={:>14.9f}".format(epoch, cur_iter, l_pxycontrast.item(), acc.item()))
                    logger.record_dict({"l_pxycontrast": l_pxycontrast.cpu().data.item(), "acc_pxycontrast": acc.item()})
                L += args.pxycontrast * l_pxycontrast

            # SGLD training of log q(x) may diverge
            # break here and record information to restart
            if L.abs().item() > 1e8:
                print("restart epoch: {}".format(epoch))
                print("save dir: {}".format(args.log_dir))
                print("id: {}".format(args.id))
                print("steps: {}".format(args.n_steps))
                print("seed: {}".format(args.seed))
                print("exp prefix: {}".format(args.exp_prefix))
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                print("restart epoch: {}".format(epoch))
                print("save dir: {}".format(args.log_dir))
                print("id: {}".format(args.id))
                print("steps: {}".format(args.n_steps))
                print("seed: {}".format(args.seed))
                print("exp prefix: {}".format(args.exp_prefix))
                assert False, "shit loss explode..."

            optim.zero_grad()
            L.backward()
            optim.step()
            cur_iter += 1

        if epoch % args.plot_every == 0:
            if args.plot_uncond:
                if args.class_cond_p_x_sample:
                    assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                    y_q = torch.randint(0, args.n_classes, (args.sgld_batch_size,)).to(device)
                    x_q = sample_q(f, replay_buffer, y=y_q)
                    plot("{}/x_q_{}_{:>06d}.png".format(args.log_dir, epoch, i), x_q)
                    if args.plot_contrast:
                        x_q = sample_q(f, replay_buffer, y=y_q, contrast=True)
                        plot("{}/contrast_x_q_{}_{:>06d}.png".format(args.log_dir, epoch, i), x_q)
                else:
                    x_q = sample_q(f, replay_buffer)
                    plot("{}/x_q_{}_{:>06d}.png".format(args.log_dir, epoch, i), x_q)
                    if args.plot_contrast:
                        x_q = sample_q(f, replay_buffer, contrast=True)
                        plot("{}/contrast_x_q_{}_{:>06d}.png".format(args.log_dir, epoch, i), x_q)
            if args.plot_cond:  # generate class-conditional samples
                y = torch.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                x_q_y = sample_q(f, replay_buffer, y=y)
                plot("{}/x_q_y{}_{:>06d}.png".format(args.log_dir, epoch, i), x_q_y)
                if args.plot_contrast:
                    y = torch.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                    x_q_y = sample_q(f, replay_buffer, y=y, contrast=True)
                    plot("{}/contrast_x_q_y_{}_{:>06d}.png".format(args.log_dir, epoch, i), x_q_y)

        if args.ckpt_every > 0 and epoch % args.ckpt_every == 0:
            checkpoint(f, replay_buffer, f"ckpt_{epoch}.pt", args)

        if epoch % args.eval_every == 0:
            # Validation set
            correct, val_loss = eval_classification(f, dload_valid)
            if correct > best_valid_acc:
                best_valid_acc = correct
                print("Best Valid!: {}".format(correct))
                checkpoint(f, replay_buffer, "best_valid_ckpt.pt", args)
            # Test set
            correct, test_loss = eval_classification(f, dload_test)
            print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, val_loss, correct))
            print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, test_loss, correct))
            f.train()
            logger.record_dict(
                {
                    "Epoch": epoch,
                    "Valid Loss": val_loss,
                    "Valid Acc": correct.detach().cpu().numpy(),
                    "Test Loss": test_loss,
                    "Test Acc": correct.detach().cpu().numpy(),
                    "Best Valid": best_valid_acc.detach().cpu().numpy(),
                    "Loss": L.cpu().data.item(),
                }
            )
        checkpoint(f, replay_buffer, "last_ckpt.pt", args)

        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hybrid training via contrastive learning")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100"])
    parser.add_argument("--data_root", type=str, default="/data/lhao/")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180], help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=0.3, help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    parser.add_argument("--start_epoch", type=int, default=0, help="helpful for reloading")
    parser.add_argument("--labels_per_class", type=int, default=-1, help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sgld_batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_iters", type=int, default=-1, help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--pyxce", type=float, default=0.0)
    parser.add_argument("--pxcontrast", type=float, default=0.0)
    parser.add_argument("--pxsgld", type=float, default=0.0)
    parser.add_argument("--pxycontrast", type=float, default=0.0)
    parser.add_argument("--pxysgld", type=float, default=0.0)
    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2, help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # network
    parser.add_argument(
        "--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"], help="norm to add to weights, none works fine"
    )
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20, help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument(
        "--class_cond_p_x_sample",
        action="store_true",
        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
        "Sample quality higher if set, but classification accuracy better if not.",
    )
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=0.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    parser.add_argument("--contrast_k", type=int, default=65536, help="number of negative samples")
    parser.add_argument("--contrast_t", default=0.1, type=float, help="softmax temperature (default: 0.1)")
    parser.add_argument("--smoothing", default=0.0, type=float)
    parser.add_argument("--workers", default=4, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
    # logging + evaluation
    parser.add_argument("--log_dir", type=str, default="./save/tmp")
    parser.add_argument("--id", default="testofhybridshit", type=str)
    parser.add_argument("--ckpt_every", type=int, default=-1, help="Epochs between checkpoint save")
    parser.add_argument("--plot_every", type=int, default=40, help="Epochs between plot")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--plot_uncond", choices=[0, 1], type=float, default=0)
    parser.add_argument("--plot_cond", choices=[0, 1], type=float, default=0)
    parser.add_argument("--plot_contrast", choices=[0, 1], type=float, default=0)
    parser.add_argument("--n_valid", type=int, default=5000)
    args = parser.parse_args()

    if args.load_path is not None:
        with open("{}/variant.json".format(dirname(abspath(args.load_path)))) as json_file:
            configs = json.load(json_file)
        loaded = SimpleNamespace(**configs)
        overwrite = copy.deepcopy(args)
        overwrite = vars(overwrite)
        overwrite.update(vars(loaded))
        overwrite = SimpleNamespace(**overwrite)
        exp_prefix = f"{args.id}-{overwrite.exp_prefix}"
        overwrite.exp_prefix = exp_prefix
        overwrite.log_dir = f"{dirname(dirname(abspath(args.load_path)))}/{exp_prefix}"

        # You may want to change seed, SGLD steps when restart after crashing
        # helpful to stablize log q(y|x) + log q(x) and log q(y|x) + log q(x|y) + log q(x)
        overwrite.seed = args.seed
        overwrite.n_steps = args.n_steps
        overwrite.start_epoch = args.start_epoch if args.start_epoch > 0 else overwrite.start_epoch
        overwrite.warmup_iters = args.warmup_iters
        overwrite.workers = args.workers
        args = overwrite
    else:
        exp_prefix = f"{args.id}-{uuid.uuid4().hex}"
        args.exp_prefix = exp_prefix
        args.log_dir = f"{args.log_dir}/{exp_prefix}"

    args.plot_contrast = 1 if (args.pxycontrast > 0 and args.plot_contrast) else 0
    args.n_classes = 100 if args.dataset == "cifar100" else 10
    set_seed(args)
    os.makedirs(args.log_dir, exist_ok=True)
    configs = OrderedDict(sorted(vars(args).items(), key=lambda x: x[0]))
    setup_logger(exp_prefix=args.exp_prefix, variant=configs, log_dir=args.log_dir)
    with open(f"{args.log_dir}/params.txt", "w") as f:
        json.dump(args.__dict__, f)
    sys.stdout = open(f"{args.log_dir}/log.txt", "a")

    main(args)
