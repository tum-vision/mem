import math
import configargparse
import sys, os

# torch
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# dalle classes and utils
from vae import distributed_utils
from vae.vae_model import DiscreteVAE, evaluate

# helper functions from mem
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("./"))
from mem.datasets import build_dataset
import mem.utils as utils
import os

def assert_config(args):
    assert args.input_H > 10 and args.input_H < 1000
    assert args.input_W > 10 and args.input_W < 1000

    assert args.timesurface == 0 or args.timesurface == 1
    assert args.logtrafo == 0 or args.logtrafo == 1
    assert args.gammatrafo == 0 or args.gammatrafo == 1
    assert (args.logtrafo and args.gammatrafo) == 0

    assert args.hotpixfilter == 0 or args.hotpixfilter == 1
    assert args.hotpix_num_stds > 0 and args.hotpix_num_stds < 30
    assert args.gamma > 0 and args.gamma < 5

    assert args.max_random_shift_evs >= 0 and args.max_random_shift_evs < 200
    assert args.max_random_shift_evs / args.input_H < 0.15
    assert args.max_random_shift_evs / args.input_W < 0.15
    
    return

# argument parsing
def get_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "--config",  
        is_config_file=True, 
        help="config file path",
    )
    parser.add_argument('--data_path', default="/usr/stud/bonello/storage/group/dataset_mirrors/01_incoming/event-n-imagenet/", type = str, required = True,
                        help='path to your folder of images for learning the discrete VAE and its codebook')
    parser.add_argument('--output_dir', default="/usr/wiss/klenk/runs/debug/", type = str,  help='path where to save logs')
    parser.add_argument('--expweek', default="09-05", type = str,  help='MONTH-DAY for better experiment managment')
    parser.add_argument('--expname', default="test", type=str, help='descriptive name')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')
    parser.add_argument('--data_set', default='npy', choices=['CIFAR', 'IMNET', 'image_folder', 'npy', 'dsec_semseg'], 
                        type=str, help='ImageNet dataset path')
    # Preprocessing
    parser.add_argument('--timesurface', type=int, default=0)
    parser.add_argument('--hotpix_num_stds', type=float, default=10)
    parser.add_argument('--hotpixfilter', type=int, default=1)
    parser.add_argument('--logtrafo', type=int, default=0)
    parser.add_argument('--gammatrafo', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--normalize_events', type=int, default=1, help="normalizes max event count to be 1 (divide by max)")
    parser.add_argument('--slice_max_evs', type=int, default=30000)
    parser.add_argument('--max_random_shift_evs', type=int, default=15)
    parser.add_argument('--rand_aug', type=int, default=1)
    

    parser.add_argument('--disable_eval', action='store_true', default=False)
    parser.add_argument('--dist_eval', action='store_true', default=False,
                    help='Enabling distributed evaluation')
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
    parser.add_argument('--color_jitter', type=float, default=0., metavar='PCT',
                        help='Color jitter factor (default: 0.)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original' )
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
            # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
                
    
    parser.add_argument('--input_H', type = int, required =False, default = 128, help='image height')
    parser.add_argument('--input_W', type = int, required = False, default = 128, help='image width')
    parser.add_argument('--weights', type = str, required = False, default=None, help = 'File with trained weights')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--save_ckpt_freq', "--vae_save_ckpt_freq", default=5, type=int)
    parser.add_argument('--resize', action='store_true', default=False)
    parser.add_argument('--disable_wandb', action='store_true', default=False)
    parser = distributed_utils.wrap_arg_parser(parser)

    train_group = parser.add_argument_group('Training settings')
    train_group.add_argument('--epochs', '--vae_epochs', type = int, default = 20, help = 'number of epochs')
    train_group.add_argument('--start_epoch', default = 0, type=int, metavar='N', help='start epoch')
    train_group.add_argument('--batch_size', "--vae_batch_size", type = int, default = 8, help = 'batch size')
    train_group.add_argument('--lr', '--vae_lr', type = float, default = 1e-3, help = 'learning rate')
    train_group.add_argument('--lr_decay_rate', "--vae_lr_decay", type = float, default = 0.98, help = 'learning rate decay')
    train_group.add_argument('--starting_temp', type = float, default = 1., help = 'starting temperature')
    train_group.add_argument('--temp_min', type = float, default = 0.5, help = 'minimum temperature to anneal to')
    train_group.add_argument('--anneal_rate', type = float, default = 1e-6, help = 'temperature annealing rate')
    train_group.add_argument('--num_images_save', type = int, default = 4, help = 'number of images to save')

    model_group = parser.add_argument_group('Model settings')
    model_group.add_argument('--num_tokens', type = int, default = 8192, help = 'number of image tokens')
    model_group.add_argument('--num_layers', type = int, default = 3, help = 'number of layers (should be 3 or above)')
    
    model_group.add_argument('--num_resnet_blocks', '--vae_num_resnet_blocks', type = int, default = 2, help = 'number of residual net blocks')
    model_group.add_argument('--loss', '--vae_loss', type=str, default='mse', choices=['mse', 'smooth_l1', 'cosine'])
    model_group.add_argument('--emb_dim', type = int, default = 512, help = 'embedding dimension == codebook dimension')
    model_group.add_argument('--hidden_dim', '--vae_hidden_dim', type = int, default = 256, help = 'hidden dimension')
    model_group.add_argument('--kl_loss_weight', "--vae_kl_loss_weight", type = float, default = 0., help = 'KL loss weight')
    model_group.add_argument('--clip', "--vae_grad_clip", type = float, default = 1e-3, help = 'Gradient clipping')
    model_group.add_argument('--straight_through', '--vae_straight_through', type=int, default=0)

    return parser.parse_known_args()[0]

def main(args):
    # initialize distributed backend
    distr_backend = distributed_utils.set_backend_from_args(args)
    distr_backend.initialize()

    using_deepspeed = \
        distributed_utils.using_backend(distributed_utils.DeepSpeedBackend)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
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
            batch_size=int(args.batch_size / torch.cuda.device_count()),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    if distr_backend.is_root_worker():
        print(f"Image size = {args.input_H}x{args.input_W}")

    vae_params, weights, loaded_start_epoch, loaded_optimizer = utils.auto_load_vae_model(args)
        
    if loaded_start_epoch is not None:
        args.start_epoch = loaded_start_epoch

    if vae_params is None:
        vae_params = dict(
            input_H = args.input_H,
            input_W = args.input_W,
            num_layers = args.num_layers,
            num_tokens = args.num_tokens,
            codebook_dim = args.emb_dim,
            hidden_dim = args.hidden_dim,
            num_resnet_blocks = args.num_resnet_blocks
        )

    vae = DiscreteVAE(
        **vae_params,
        loss = args.loss,
        kl_div_loss_weight = args.kl_loss_weight,
        straight_through = args.straight_through,
    )

    if weights is not None:
        print("Loading checkpoint from epoch %d" % args.start_epoch)
        vae.load_state_dict(weights)

    if not using_deepspeed:
        vae = vae.cuda()

    assert len(dataset_train) > 0, 'folder does not contain any images'
    if distr_backend.is_root_worker():
        print(f'{len(dataset_train)} images found for training')

    # optimizer
    opt = Adam(vae.parameters(), lr = args.lr)
    sched = ExponentialLR(optimizer = opt, gamma = args.lr_decay_rate)


    # CHANGE print num of parameters
    pytorch_total_params = sum(p.numel() for p in vae.parameters())
    pytorch_train_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    if distr_backend.is_root_worker():
        print("Number of parameters in vae:", pytorch_total_params)
        print("Number of trainable parameters in vae:", pytorch_train_params)

    if distr_backend.is_root_worker():
        # weights & biases experiment tracking
        import wandb

        wandb_dir = os.path.abspath('./../')
        os.environ['WANDB_DIR'] = wandb_dir
        print(f"Logging to {wandb_dir}")

        run = wandb.init(
            project = 'dalle_train_vae',
            dir = wandb_dir,
            group = f"{args.expweek}_{args.expname}",
            job_type = 'train_model',
            config = args,
            mode = 'disabled' if args.disable_wandb else 'online',
        )

    # distribute
    distr_backend.check_batch_size(args.batch_size)
    deepspeed_config = {'train_batch_size': args.batch_size}

    (distr_vae, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
        args=args,
        model=vae,
        optimizer=opt,
        model_parameters=vae.parameters(),
        training_data=dataset_train if using_deepspeed else data_loader_train,
        lr_scheduler=sched if not using_deepspeed else None,
        config_params=deepspeed_config,
    )

    using_deepspeed_sched = False
    # Prefer scheduler in `deepspeed_config`.
    if distr_sched is None:
        distr_sched = sched
    elif using_deepspeed:
        # We are using a DeepSpeed LR scheduler and want to let DeepSpeed
        # handle its scheduling.
        using_deepspeed_sched = True

    def save_model(path, epoch):
        save_obj = {
            'hparams': vae_params,
            'epoch': epoch,
            'optimizer': distr_opt.state_dict(),
            'args': args,
        }

        if not distr_backend.is_root_worker():
            return

        save_obj = {
            **save_obj,
            'weights': vae.state_dict()
        }

        print(f'Saving model to {path}')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_obj, path)

    # starting temperature
    global_step = 0
    temp = args.starting_temp

    if distr_backend.is_root_worker():
        print(vae)

    if distr_backend.is_root_worker():
        print(f"Start training for {args.epochs} epochs")
        print(f"Start epoch: {args.start_epoch}")
        print(f"batch size: {args.batch_size}")
        print(args)
    for epoch in range(args.start_epoch, args.epochs):
        vae.train(True)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 10

        for i, (images, _) in enumerate(metric_logger.log_every(distr_dl, print_freq, header)):
            images = images.cuda()
            logs = {}

            loss, recons = distr_vae(
                images,
                return_loss = True,
                return_recons = True,
                temp = temp 
            )

            if using_deepspeed:
                # Gradients are automatically zeroed after the step
                distr_vae.backward(loss)
                if i % 10 == 0:
                    logs = {**logs, 'grad_norm': distr_vae.get_grad_norm()}

                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(distr_vae.parameters(), args.clip)
                distr_vae.step()
            else:
                distr_opt.zero_grad()
                loss.backward()

                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(distr_vae.parameters(), args.clip)
                distr_opt.step()

            torch.cuda.synchronize()


            if i % 10000 == 0:
                if distr_backend.is_root_worker():
                    k = args.num_images_save

                # temperature anneal
                temp = max(temp * math.exp(-args.anneal_rate * global_step), args.temp_min)

                # lr decay

                # Do not advance schedulers from `deepspeed_config`.
                if not using_deepspeed_sched:
                    distr_sched.step()

            # Collective loss, averaged
            avg_loss = distr_backend.average_all(loss)

            if distr_backend.is_root_worker():
                if i % 1000 == 0:
                    lr = distr_sched.get_last_lr()[0]

                    logs = {
                        **logs,
                        'epoch': epoch,
                        'iter': i,
                        'loss': avg_loss.item(),
                        'lr': lr,
                    }

                wandb.log(logs)
            
            lr = distr_sched.get_last_lr()[0]
            metric_logger.update(loss=avg_loss.item())
            metric_logger.update(lr=lr)

            global_step += 1
            del loss

        if (epoch + 1) % 25 == 0:
            test_stats = evaluate(data_loader_val, vae, 'cuda')

            if distr_backend.is_root_worker():
                wandb.log({
                    'test_loss': test_stats["loss"],
                    'codebook_usage': float(test_stats["codebook_indices"]) / args.num_tokens,
                    'epoch': epoch
                })

        if distr_backend.is_root_worker() and ((epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            save_model(os.path.join(args.output_dir, f'checkpoint-{epoch}.pt'), epoch)

        metric_logger.synchronize_between_processes()

    if distr_backend.is_root_worker():
        # save final vae and cleanup
        save_model(os.path.join(args.output_dir, f'checkpoint-final.pt'), args.epochs-1)
        wandb.save('./vae-final.pt')

        wandb.finish()


if __name__ == '__main__':
    opts = get_args()
    assert_config(opts)
    main(opts)
