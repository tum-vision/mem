import argparse
import copy
import os
import os.path as osp
import time

import numpy as np
import sys
sys.path.append(os.path.abspath("../../.."))
sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("../"))

import mmcv
import mmcv_custom
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed
from mmcv_custom import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

from backbone import EventDataset
from mem.mem.semantic_segmentation.backbone import mem


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models', default="/usr/wiss/klenk/code/pretraining/ss/")
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=None)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    # Here the model is not yet loaded, so we can not do the check

    with torch.no_grad():
        freeze = False
        if freeze:
            for name, param in model.backbone.named_parameters():
                if "fpn" not in name:
                    param.requires_grad = False
            print(f"\n\n**********WARNING**************\n\n: We have frozen the backbone!!!!!!!! \n\n**********\n\n**********\n\n")

    if "subset" in cfg.data.train.data_root:
        print(f"\n\nWarning**** FT only on Subset****")
        assert "subset" not in cfg.data.val.data_root

    anns = [cfg.data.train, cfg.data.val, cfg.data.test]
    if "anns19" in anns[0].ann_dir:
        assert np.all(["anns19/" in ann.ann_dir for ann in anns])
        assert np.all(["/labels19.txt" == ann.classes[-13:] for ann in anns])
        print(f"Using 19 labels")
    else:
        assert np.all(["anns19" not in ann.ann_dir for ann in anns])
        assert np.all(["anns/" in ann.ann_dir for ann in anns])
        assert np.all(["/labels.txt" == ann.classes[-11:] for ann in anns])
        print(f"Using 11 labels")

    logger.info(model)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_segmentor(
        model,
        datasets,
        cfg,
        args,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
