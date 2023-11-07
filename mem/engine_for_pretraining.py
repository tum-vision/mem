# --------------------------------------------------------
# Masked Event Modelling: Self-Supervised Pretraining for Event Cameras
# 
# Based on the following code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------'
import math
from math import sqrt
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import os
import numpy as np

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

import utils
import wandb

MASK_RATIO = 0.5

def plot_mae(gt, pred, mask, dumppath, counter=0, normalize=False):
    if counter == 0:
        print(f"Writing masks to {dumppath}")
    if normalize:
        dumppath = os.path.join(dumppath, "norm")
    os.makedirs(dumppath, exist_ok=True)

    preds = pred.detach().cpu().numpy().transpose((0, 2, 3, 1)) # (B, H, W, 3)
    gts = gt.detach().cpu().numpy().transpose((0, 2, 3, 1)) # (B, H, W, 3)

    #mask = mask - 1
    #mask = -mask
    masks = mask.detach().unsqueeze(1).cpu().numpy().transpose((0, 2, 3, 1)) # (B, H, W, 1)
    masks = (255*masks).astype(np.uint8)

    for i in range(len(preds)):
        # GT
        plt.figure()
        plt.imshow(gts[i, ...])
        plt.axis('off')
        plt.show()
        path = os.path.join(dumppath, f"{counter:06d}_GT.png")
        plt.savefig(path)
        plt.close()
        
        # Pred
        pred = (255*(preds[i, ...] - preds[i, ...].min()) / (preds[i, ...].max() - preds[i, ...].min())).astype(np.int)
        plt.figure()
        plt.imshow(pred)
        plt.axis('off')
        plt.show()
        path = os.path.join(dumppath, f"{counter:06d}_pred.png")
        plt.savefig(path)
        plt.close()

        # Overlay
        plt.figure()
        plt.imshow(pred)
        plt.imshow(masks[i, ...], alpha=0.5, cmap="gray")
        plt.axis('off')
        plt.show()
        path = os.path.join(dumppath, f"{counter:06d}_overlay.png")
        plt.savefig(path)
        plt.close()
        counter += 1

    return counter



def plot_masks_and_images(images_sample, reconstruction, masks, dumppath, targets, counter=0, normalize=False):
    if counter == 0:
        print(f"Writing masks to {dumppath}")
    if normalize:
        dumppath = os.path.join(dumppath, "norm")
    os.makedirs(dumppath, exist_ok=True)
    pltimgs = images_sample.detach().cpu().numpy().transpose((0, 2, 3, 1))
    recs = reconstruction.detach().cpu().numpy().transpose((0, 2, 3, 1))  # (B, H, W, 3)
    masks = masks.cpu().numpy().transpose((0, 2, 3, 1)) # (B, H, W, 1)
    masks = masks.astype(np.uint8) * 255
    for samp_i in range(len(reconstruction)):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(40, 10))
        recs[samp_i,:,:,1] *= 0
        if normalize:
            recs[samp_i,:,:,0] = (recs[samp_i,:,:,0]-recs[samp_i,:,:,0].min())/(recs[samp_i,:,:,0].max()-recs[samp_i,:,:,0].min())
            recs[samp_i,:,:,2] = (recs[samp_i,:,:,2]-recs[samp_i,:,:,2].min())/(recs[samp_i,:,:,2].max()-recs[samp_i,:,:,2].min())
        axs[0].imshow(pltimgs[samp_i, :, :])
        axs[0].imshow(masks[samp_i, :, :], alpha=0.5, cmap="gray")
        axs[1].imshow(recs[samp_i, :, :])
        axs[2].imshow(pltimgs[samp_i, :, :])
        axs[0].set_title("masked gt")
        axs[1].set_title("reconstruction")
        axs[2].set_title("gt")
        plt.show()
        plt.savefig(os.path.join(dumppath, f"{targets[samp_i].item()}_{counter}.png"))
        plt.close()
        counter += 1  
    return counter


def train_one_epoch(model: torch.nn.Module, d_vae: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, run=None, args=None, plotting=False, MAE=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    counter = 0
    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        logs = {}

        samples, images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)  # (B, C, 224, 224)
        samples = samples.to(device, non_blocking=True) # (B, C, 224, 224)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True) # (B, numMasksX, numMasksY) = (B, 14, 14)

        with torch.no_grad():
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool) # (B, 14*14)
            if MAE:
                labels = images.flatten()
            else:
                input_ids = d_vae.get_codebook_indices(images).flatten(1)   # (B, 14*14)
                labels = input_ids[bool_masked_pos]  # (numMasked)

        with torch.cuda.amp.autocast(): # TODO: debug loss value / check loss range
            if MAE:
                loss, pred, mask = model(samples, mask_ratio=MASK_RATIO)
            else:
                outputs = model(samples, bool_masked_pos=bool_masked_pos, return_all_tokens=False) # (numMasked, 8192)
                loss = nn.CrossEntropyLoss()(input=outputs, target=labels)
                # outputs.dtype, outputs.shape => (torch.float16, torch.Size([381, 150528]))
                # labels.dtype, labels.shape => (torch.int64, torch.Size([381]))

        loss_value = loss.item()

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if not MAE and ((run != None and step % 10000 == 0) or grad_norm > 6) and utils.is_main_process():
            outputs_max = torch.argmax(outputs, 1)

            with torch.no_grad():
                outputs_max = torch.argmax(outputs, 1)
                k = min(4, input_ids.shape[0])
                reconstructed_input = d_vae.decode(input_ids[:k])
                codes = input_ids[:k]
                
                code_size = codes[bool_masked_pos[:k]].shape[0]
                codes[bool_masked_pos[:k]] = outputs_max[:code_size]
                reconstruction = d_vae.decode(codes)

                # show where image was reconstructed
                bool_masked_pos_large = torch.reshape(bool_masked_pos[:k], (k,1,int(d_vae.input_size[0] / (2 ** d_vae.num_layers)), int(d_vae.input_size[1] / (2 ** d_vae.num_layers))))
                bool_masked_pos_large = bool_masked_pos_large.repeat_interleave(2 ** d_vae.num_layers, dim=1+1)                 # TODO: make general
                bool_masked_pos_large = bool_masked_pos_large.repeat_interleave(2 ** d_vae.num_layers, dim=2+1)                 # TODO: make general

                images_sample = images[:k]
                images_sample, reconstruction, outputs_max, reconstructed_input, bool_masked_pos_large = map(lambda t: t.detach().cpu(), (images_sample, reconstruction, outputs_max, reconstructed_input, bool_masked_pos_large))

                if plotting:
                    dumppath = os.path.join("/usr/stud/bonello/storage/user/event-pretraining/figures/visualize_reconstructions/renders", args.expweek, args.expname, f"epoch_{epoch}")
                    counter = plot_masks_and_images(images_sample, reconstruction, bool_masked_pos_large, dumppath, counter=counter) 

                images_sample, reconstruction, reconstructed_input = map(lambda t: make_grid(t.float(), nrow = int(sqrt(k)), normalize = False, range = (-1, 1)), (images_sample, reconstruction, reconstructed_input))
                bool_masked_pos_large = make_grid(bool_masked_pos_large, nrow=2)

                logs = {
                    **logs,
                    'sample images':        wandb.Image(images_sample, caption = 'original images'),
                    'reconstructed_input':  wandb.Image(reconstructed_input, caption = 'reconstructed_input'),
                    'reconstructions':      wandb.Image(reconstruction, caption = 'reconstruction', masks={"prediction": {"mask_data": bool_masked_pos_large[0].numpy(), "class_lables": {0: "input", 1:"guess"}}}),
                    'reconstructed_codebook_indices':     wandb.Histogram(outputs_max),
                }

        elif MAE:
            with torch.no_grad():
                B, C, H, W = images.shape
                k = 4

                images_sample = images[:k, ...]
                reconstruction = pred[:k, ...]
                images_sample, reconstruction = map(lambda t: t.detach().cpu(), (images_sample, reconstruction))
                images_sample, reconstruction = map(lambda t: make_grid(t.float(), nrow = int(sqrt(k)), normalize = False, range = (-1, 1)), (images_sample, reconstruction))

                logs = {
                    **logs,
                    'sample images':        wandb.Image(images_sample, caption = 'original images'),
                    'reconstructions':      wandb.Image(reconstruction, caption = 'reconstruction'),
                }
    
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("INFO:", "samples", samples.shape, samples)
            print("INFO:", "bool_masked_pos", bool_masked_pos.shape, bool_masked_pos)
            print("INFO:", "images", images.shape, images)
            print("INFO:", "outputs", outputs.shape, outputs)
            print("INFO:", "lables", labels.shape, labels)
            print("INFO:", "loss", loss)
            print("INFO:", "loss_value", loss_value)
            sys.exit(1)

        torch.cuda.synchronize()

        if not MAE:
            mlm_acc = (outputs.max(-1)[1] == labels).float().mean().item()      
            metric_logger.update(mlm_acc=mlm_acc)
            if log_writer is not None:
                log_writer.update(mlm_acc=mlm_acc, head="loss")
        else: 
            mlm_acc = 0

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if (step % 100 == 0 or grad_norm > 6) and utils.is_main_process():
            logs = {
                    **logs,
                    'epoch': epoch,
                    'loss': loss_value,
                    'loss_scale': loss_scale_value,
                    'lr': max_lr,
                    'min_lr': min_lr,
                    'grad_norm': grad_norm,
                    'mlm_acc' : mlm_acc,
                }
            
            wandb.log(logs)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, d_vae, device, args, plotting=False, MAE=False):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    print(f"model made eval")

    counter = 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples, images, bool_masked_pos = batch[0]
        images = images.to(device, non_blocking=True)   # (B*1.5, 3, H, W)
        samples = samples.to(device, non_blocking=True) # (B*1.5, 3, H, W)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True) # (B*1.5, Px=14, Py) where int(d_vae.input_size[0] / (2 ** d_vae.num_layers))
        mask_cop = bool_masked_pos.clone().detach()

        with torch.no_grad():
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool) # (B, 14*14), 14 == Wp == Hp
            if MAE:
                labels = images.flatten()
            else:
                input_ids = d_vae.get_codebook_indices(images).flatten(1)   # (B, 14*14)
                labels = input_ids[bool_masked_pos]  # (numMasked)

        with torch.cuda.amp.autocast():
            if MAE:
                loss, pred, mask  = model(samples, mask_ratio=MASK_RATIO)
            else:
                outputs = model(samples, bool_masked_pos=bool_masked_pos, return_all_tokens=False) # (numMasked, 8192)
                loss = nn.CrossEntropyLoss()(input=outputs, target=labels)
                # outputs.dtype, outputs.shape => (torch.float16, torch.Size([381, 150528]))
                # labels.dtype, labels.shape => (torch.int64, torch.Size([381]))

        if plotting and args.MAE:
            dumppath = os.path.join(".", args.expweek, args.expname, f"epoch_eval")

            B = mask.shape[0]
            bool_masked_pos_large = torch.reshape(mask, (B, 14, 14))
            bool_masked_pos_large = bool_masked_pos_large.repeat_interleave(16, dim=1)
            bool_masked_pos_large = bool_masked_pos_large.repeat_interleave(16, dim=2) # (6, 224, 224)
            counter = plot_mae(samples, pred, bool_masked_pos_large, dumppath, counter=counter)
        if plotting and not args.MAE:
            with torch.no_grad():
                k = input_ids.shape[0]
                codes = input_ids[:k]
                
                code_size = codes[bool_masked_pos[:k]].shape[0]
                outputs_max = torch.argmax(outputs, 1)
                codes[bool_masked_pos[:k]] = outputs_max[:code_size]
                reconstruction = d_vae.decode(codes)

            mask_cop = mask_cop.unsqueeze(1) # (B*1.5, 1, 14, 14)
            mask_cop = mask_cop.repeat_interleave(2 ** d_vae.num_layers, dim=1+1)
            mask_cop = mask_cop.repeat_interleave(2 ** d_vae.num_layers, dim=1+2) # (B*1.5, 1, H, W)

            dumppath = os.path.join("/usr/stud/bonello/storage/user/event-pretraining/figures/visualize_reconstructions/m", args.expweek, args.expname, f"epoch_eval")
            counter = plot_masks_and_images(images, reconstruction, mask_cop, dumppath, targets=batch[1], counter=counter) 

        metric_logger.update(loss=loss.item())

        if not MAE:
            mlm_acc = (outputs.max(-1)[1] == labels).float().mean().item()
        else:
            mlm_acc = 0
        metric_logger.meters['mlm_acc'].update(mlm_acc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if not MAE:
        print('* mlm_acc {mlm_acc.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(mlm_acc=metric_logger.mlm_acc, losses=metric_logger.loss))
    else:
        print(' loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))   

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
