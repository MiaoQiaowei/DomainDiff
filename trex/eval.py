# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import argparse
import datetime
import math
import numpy
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import multi_crop
import utils
from trex import tReX


def get_args():
    parser = argparse.ArgumentParser(description="Training t-ReX models on ImageNet-1K")

    # Model parameters
    parser.add_argument(
        "--arch",
        default="resnet50",
        type=str,
        choices=["resnet50"],
        help="Name of the architecture to train.",
    )

    # Multi-crop arguments
    parser.add_argument(
        "--mc_global_number",
        type=int,
        default=1,
        help="Number of random global crops.",
    )
    parser.add_argument(
        "--mc_global_scale",
        type=float,
        nargs="+",
        default=(0.25, 1.0),
        help="Scale range for global crops.",
    )
    parser.add_argument(
        "--mc_local_number",
        type=int,
        default=8,
        help="Number of random local crops.",
    )
    parser.add_argument(
        "--mc_local_scale",
        type=float,
        nargs="+",
        default=(0.05, 0.25),
        help="Scale range for local crops.",
    )
    # Projector head arguments
    parser.add_argument(
        "--pr_hidden_layers",
        default=3,
        type=int,
        help="Number of hidden layers in the projector head.",
    )
    parser.add_argument(
        "--pr_hidden_dim",
        default=2048,
        type=int,
        help="Number of hidden units in the hidden layers of the projector head.",
    )
    parser.add_argument(
        "--pr_bottleneck_dim",
        default=256,
        type=int,
        help="Bottleneck layer dimension in the projector head.",
    )
    parser.add_argument(
        "--pr_no_input_l2_norm",
        action="store_true",
        help="Whether to NOT use l2 normalization in the input of the projector head.",
    )

    # Training/Optimization parameters
    parser.add_argument(
        "--use_fp16",
        type=utils.bool_flag,
        default=True,
        help="Whether or not to use mixed precision for training.",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=64,
        type=int,
        help="Batch size per GPU. Total batch size is proportional to the number of GPUs.",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4,
        help="Weight decay for the SGD optimizer.",
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float,
        help="Maximum learning rate at the end of linear warmup.",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="Number of training epochs for the learning-rate-warm-up phase.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate at the end of training.",
    )

    # Supervised classification parameters
    parser.add_argument(
        "--clf_tau",
        default=0.1,
        type=float,
        help="Temperature for cosine softmax loss.",
    )

    # Memory parameters
    parser.add_argument(
        "--memory_size",
        default=8192,
        type=int,
        help="Size of the memory bank used to compute class prototypes.",
    )

    # Misc
    parser.add_argument(
        "--data_dir",
        default="/path/to/imagenet",
        type=str,
        help="Path to the ImageNet dataset containing train/ and val/ folders.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to the output folder to save logs and checkpoints.",
    )
    parser.add_argument(
        "--saveckpt_freq",
        default=20,
        type=int,
        help="Frequency of intermediate checkpointing.",
    )
    parser.add_argument(
        "--seed",
        default=22,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        default=12,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="Url used to set up distributed training.",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore this argument; No need to set it manually.",
    )

    parser.add_argument(
        "--test",
        action="store_true"
    )

    parser.add_argument(
        "--n_classes",
        default=100,
        type=int,
    )

    args = parser.parse_args()

    return args


def main(args):
    utils.init_distributed_mode(args)

    # os.makedirs(args.output_dir, exist_ok=True)
    utils.print_program_info(args, os.path.join(args.output_dir, "program_info.txt"))

    utils.fix_random_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    # ==================================================
    # Data
    # ==================================================
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    _dataset = datasets.ImageFolder(
        # os.path.join("/data/yuanjunkun/yjk/dataset/imagenet-1k", "val")
        os.path.join("/data/miaoqiaowei/data/imagenet-100", "val")
    )
    real_dict = _dataset.class_to_idx

    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_dict = val_dataset.class_to_idx
    val2real = {}
    for k,v in val_dict.items():
        val_index = v
        real_index = real_dict[k]
        val2real[val_index] = real_index

    print("=> Validation dataset:c {}".format(val_dataset))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # ==================================================
    # Model and optimizer
    # ==================================================
    print("=> creating model '{}'".format(args.arch))
    model = tReX(
        args.arch,
        not args.pr_no_input_l2_norm,
        args.pr_hidden_layers,
        args.pr_hidden_dim,
        args.pr_bottleneck_dim,
        clf_tau=args.clf_tau,
        memory_size=args.memory_size,
        n_classes=args.n_classes
    )
    with open(os.path.join(args.output_dir, "model.txt"), "w") as fp:
        fp.write("{}".format(model))

    model = model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=False
    )

    clf_loss = nn.CrossEntropyLoss().cuda()

    params_groups = utils.get_params_groups(model)
    optimizer = torch.optim.SGD(
        params_groups, lr=0, momentum=0.9
    )  # we set lr and wd in train_one_epoch

    fp16_scaler = None
    if args.use_fp16:
        # mixed precision training
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ==================================================
    # Loading previous checkpoint & initializing tensorboard
    # ==================================================

    # to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        # run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
    )
    # start_epoch = to_restore["epoch"]

    tb_dir = os.path.join(args.output_dir, f"tb-{args.rank}")
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(tb_dir, flush_secs=30)
    # ============ evaluate model ... ============
    test_stats = eval(model, clf_loss, val_loader, 0, args, val2real)

    
@torch.no_grad()
def eval(model, clf_loss, data_loader, epoch, args, val2real):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    model.eval()

    mask = []
    for i in range(args.n_classes):
        if i in val2real.values():
            mask += [1]
        else:
            mask += [0]
    mask = torch.BoolTensor(mask).float().cuda().view(1,-1)

    for it, (image, label) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):

        image = image.cuda(non_blocking=True)
        # label = label.cuda(non_blocking=True)
        # print(label)
        label = torch.LongTensor([val2real[i] for i in label.numpy()]).cuda(non_blocking=True)

        # compute output
        output = model(image)

        # output *= mask

        loss = clf_loss(output, label)
        acc1, acc5 = utils.accuracy(output, label, topk=(1, 5))

        # record logs
        metric_logger.update(loss=loss.item())
        metric_logger.update(top1=acc1.item())
        metric_logger.update(top5=acc5.item())
        # print(metric_logger)
        # exit()
    metric_logger.synchronize_between_processes()
    
    print("Averaged test stats:", metric_logger)
    return {f"test/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    args = get_args()
    main(args)
