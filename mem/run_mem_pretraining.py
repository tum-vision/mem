# --------------------------------------------------------
# Masked Event Modelling: Self-Supervised Pretraining for Event Cameras
# 
# Based on the following code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------'
import configargparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os, sys

from pathlib import Path

from timm.models import create_model
from optim_factory import create_optimizer

from datasets import build_pretraining_dataset
from engine_for_pretraining import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils

sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("./"))

def get_args():
    parser = configargparse.ArgumentParser('Pretraining script', add_help=False)
    parser.add_argument(
        "--config",                                       
        is_config_file=True, 
        help="config file path",
    )
    parser.add_argument('--expweek', type = str, required = True, help='MONTH-DAY for better experiment managment')
    parser.add_argument('--expname', default=None, type=str, help='descriptive name')
    parser.add_argument('--batch_size', "--pt_batch_size", default=64, type=int)
    parser.add_argument('--epochs', "--pt_epochs", default=300, type=int)
    parser.add_argument('--save_ckpt_freq', "--pt_save_ckpt_freq", default=20, type=int)
    parser.add_argument("--discrete_vae_weight_path", type=str)
    parser.add_argument("--discrete_vae_type", type=str, default="event")

    # Preprocessing
    parser.add_argument('--timesurface', type=int, default=0)
    parser.add_argument('--hotpixfilter', type=int, default=1)
    parser.add_argument('--hotpix_num_stds', type=float, default=10)
    parser.add_argument('--logtrafo', type=int, default=0)
    parser.add_argument('--gammatrafo', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--normalize_events', type=int, default=0, help="normalizes max event count to be 1 (divide by max)")
    parser.add_argument('--slice_max_evs', type=int, default=30000)
    parser.add_argument('--max_random_shift_evs', type=int, default=15)
    parser.add_argument('--rand_aug', type=int, default=1)

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--pretrained', default=0, type=int, help='bool')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--masking', default="block", type=str, help='type of masking. block or random')
    parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)


    parser.add_argument('--MAE', '--mae', default=0, type=int, help='use MAE loss')
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
    parser.add_argument('--input_H', default=128, type=int, help='images input size for backbone')
    parser.add_argument('--input_W', default=128, type=int, help='images input size for backbone')
    parser.add_argument('--input_H2', default=128, type=int, help='images input size for discrete vae')
    parser.add_argument('--input_W2', default=128, type=int, help='images input size for discrete vae')

    parser.add_argument('--drop_path', "--pt_dropout", type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_pretraining', action='store_true', default=False)


    # Optimizer parametersx
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: 0.9, 0.999, use opt default)')
    parser.add_argument('--clip_grad', "--pt_grad_clip", type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', "--pt_lr", type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', "--pt_warmup_steps", type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "lanczos")')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='npy', choices=['CIFAR', 'IMNET', 'image_folder', 'npy', 'dsec_semseg'],
                        # IMNET = (train, val). image_folder = (all sub-folders)
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')
    parser.add_argument('--resize', action='store_true', default=False)
    parser.add_argument('--color_jitter', '--pt_color_jitter', type=float, default=0.2, metavar='PCT', 
                        help='Color jitter factor (default: 0.2)')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='.',
                        help='path where to tensorboard log')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # wandb settings
    parser.add_argument('--wandb', type=bool, default=True, help='Use wandb?')
    parser.add_argument('--wandb_group', default='pt', help='wandb group distributed training')

    return parser.parse_known_args()[0]


def get_model(args):
    print(f"Creating model: {args.model}")
    
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        in_chans= 2 if args.voxel==0 else args.voxel,
        img_size=(args.input_H, args.input_W),
        patch_size= (2 ** args.num_layers, 2 ** args.num_layers),
        embed_dim=args.transformer_emb,
        depth=args.transformer_depth,
        num_heads=args.transformer_heads,
        mlp_ratio=args.transformer_mlp_ratio,
        vocab_size=args.num_tokens,
    )

    if bool(args.pretrained):
        print(f"creating vit model = {args.model}")
        vit_model = create_model(
            'vit_base_patch16_224',
            pretrained=True,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )

        with torch.no_grad():
            model.patch_embed.proj.weight.copy_(vit_model.patch_embed.proj.weight)
            model.patch_embed.proj.bias.copy_(vit_model.patch_embed.proj.bias)

            for i in range(len(model.blocks)):
                model.blocks[i].norm1.weight.copy_(vit_model.blocks[i].norm1.weight)
                model.blocks[i].norm1.bias.copy_(vit_model.blocks[i].norm1.bias)
                model.blocks[i].attn.qkv.weight.copy_(vit_model.blocks[i].attn.qkv.weight)
                model.blocks[i].attn.proj.weight.copy_(vit_model.blocks[i].attn.proj.weight)
                model.blocks[i].attn.proj.bias.copy_(vit_model.blocks[i].attn.proj.bias)
                model.blocks[i].norm2.weight.copy_(vit_model.blocks[i].norm2.weight)
                model.blocks[i].norm2.bias.copy_(vit_model.blocks[i].norm2.bias)
                model.blocks[i].mlp.fc1.weight.copy_(vit_model.blocks[i].mlp.fc1.weight)
                model.blocks[i].mlp.fc1.bias.copy_(vit_model.blocks[i].mlp.fc1.bias)
                model.blocks[i].mlp.fc2.weight.copy_(vit_model.blocks[i].mlp.fc2.weight)
                model.blocks[i].mlp.fc2.bias.copy_(vit_model.blocks[i].mlp.fc2.bias)

            model.norm.weight.copy_(vit_model.norm.weight)
            model.norm.bias.copy_(vit_model.norm.bias)

    return model


def main(args):
    utils.init_distributed_mode(args)
    print("Running", f"{args.expweek}_{args.expname}")
    print(args)

    if args.MAE:
        from modeling_mae import mae_vit_base_patch16_dec512d8b
    else:
        import modeling_pretrain

    if utils.is_main_process():
        import wandb

        wandb_dir = os.path.abspath('./../')
        os.environ['WANDB_DIR'] = wandb_dir
        print(f"Logging to {wandb_dir}")

        run = wandb.init(
            dir = wandb_dir,
            group= f"{args.expweek}_{args.expname}",
            project = 'mem_pretraining',
            job_type = 'train_model',
            config = args,
            sync_tensorboard=True,
        )

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # normalize input_size
    input_size = (args.input_H, args.input_W)
    if len(input_size) == 1:
        input_size = tuple((input_size[0], input_size[0]))
    else:
        input_size = tuple(input_size[:2])

    second_input_size = (args.input_H2, args.input_W2)
    if len(second_input_size) == 1:
        second_input_size = tuple((second_input_size[0], second_input_size[0]))
    else:
        second_input_size = tuple(second_input_size[:2])

    if args.MAE:
        model = mae_vit_base_patch16_dec512d8b(norm_pix_loss=0, LOSS_ONLY_MASKED_MAE=True) # LOSS_ONLY_MASKED_MAE = True if same as MAE paper
    else:
        model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])
    print("Window size = %s" % str(args.window_size))
    args.patch_size = patch_size

    # get dataset
    dataset_train = build_pretraining_dataset(args)
    if args.disable_eval_during_pretraining:
        dataset_val = None
    else:
        dataset_val = build_pretraining_dataset(is_train=False, args=args)

    # prepare discrete vae
    if not args.MAE:
        d_vae =  utils.create_d_vae(
            weight_path=args.discrete_vae_weight_path, d_vae_type=args.discrete_vae_type,
            device=device, image_size=second_input_size)
    else:
        print("Using MAE loss")
        d_vae = None

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        print(num_tasks, sampler_rank)
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        args.log_dir = args.log_dir + args.wandb_group
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * utils.get_world_size()
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    model.to(device)

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model, d_vae, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            run=(run if args.wandb and utils.is_main_process() else None),
            args=args,
            MAE=args.MAE
        )
        
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        if data_loader_val is not None:
            print(f"before evaluate()")
            test_stats = evaluate(data_loader_val, model, d_vae, device, args, MAE=args.MAE)
            print(f"test_stats: {test_stats}")

            if log_writer is not None:
                if not args.MAE:
                    log_writer.update(test_mlm_acc=test_stats['mlm_acc'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        wandb.finish()

    utils.cleanup_distributed_mode()


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
    print("Finish")
