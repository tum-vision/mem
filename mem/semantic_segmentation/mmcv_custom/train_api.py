import random
import warnings


import copy
import timm
from timm.models import create_model as create_model_timm
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner

import sys
from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger
try:
    import apex
except:
    print('apex is not installed')
from scipy import interpolate

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def assert_tmpModelTimm_equal_semsegModel(tmp_model, model):
    # tmp_model: created from timm-library, same as run_finetune_class.py script
    # model: created from SemSeg build_segmentor() function => check mem.py EvBEiT()
    assert model.backbone.use_rel_pos_bias == False
    assert model.backbone.use_abs_pos_emb == True

    for i in range(len(tmp_model.blocks)):
        assert torch.abs((tmp_model.blocks[i].norm1.weight - model.backbone.blocks[i].norm1.weight).sum()-0) < 1e-6
        assert torch.abs((tmp_model.blocks[i].norm1.bias - model.backbone.blocks[i].norm1.bias).sum()-0) < 1e-6

        assert torch.abs((tmp_model.blocks[i].attn.qkv.weight - model.backbone.blocks[i].attn.qkv.weight).sum()-0) < 1e-6
        # assert torch.abs((tmp_model.blocks[i].attn.qkv.bias - model.backbone.blocks[i].attn.qkv.bias).sum()-0) < 1e-6

        assert ((tmp_model.blocks[i].attn.attn_drop.p - model.backbone.blocks[i].attn.attn_drop.p)-0) < 1e-6
        
        assert torch.abs((tmp_model.blocks[i].attn.proj.weight - model.backbone.blocks[i].attn.proj.weight).sum()-0) < 1e-6
        assert torch.abs((tmp_model.blocks[i].attn.proj.bias - model.backbone.blocks[i].attn.proj.bias).sum()-0) < 1e-6

        assert torch.abs((tmp_model.blocks[i].attn.proj.weight - model.backbone.blocks[i].attn.proj.weight).sum()-0) < 1e-6
        assert torch.abs((tmp_model.blocks[i].attn.proj.bias - model.backbone.blocks[i].attn.proj.bias).sum()-0) < 1e-6

        assert ((tmp_model.blocks[i].attn.attn_drop.p - model.backbone.blocks[i].attn.attn_drop.p)-0) < 1e-6    
        if isinstance(tmp_model.blocks[i].drop_path, torch.nn.Identity):
            assert isinstance(model.backbone.blocks[i].drop_path, torch.nn.Identity)
        # else:
            #assert ((float(str(tmp_model.blocks[i].drop_path)[11:-2]) - float(str(model.backbone.blocks[i].drop_path)[11:-2]))-0) < 1e-6    

        assert torch.abs((tmp_model.blocks[i].norm2.weight - model.backbone.blocks[i].norm2.weight).sum()-0) < 1e-6
        assert torch.abs((tmp_model.blocks[i].norm2.bias - model.backbone.blocks[i].norm2.bias).sum()-0) < 1e-6       

        assert torch.abs((tmp_model.blocks[i].mlp.fc1.weight - model.backbone.blocks[i].mlp.fc1.weight).sum()-0) < 1e-6
        assert torch.abs((tmp_model.blocks[i].mlp.fc1.bias - model.backbone.blocks[i].mlp.fc1.bias).sum()-0) < 1e-6   
        assert torch.abs((tmp_model.blocks[i].mlp.fc2.weight - model.backbone.blocks[i].mlp.fc2.weight).sum()-0) < 1e-6
        assert torch.abs((tmp_model.blocks[i].mlp.fc2.bias - model.backbone.blocks[i].mlp.fc2.bias).sum()-0) < 1e-6     

        assert ((tmp_model.blocks[i].mlp.drop.p - model.backbone.blocks[i].mlp.drop.p)-0) < 1e-6           

    print(f"blocks(tmp_model) == blocks(model)")


def interpolate_pos_embed(model, pos_tokens):
    
    embedding_size = pos_tokens.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_tokens.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_tokens[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_tokens[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        
    return new_pos_embed

def copy_tmpModelTimm_to_semsegBackbone(SRC_model, DST_model):
    # SRC_model: created from timm-library, same as run_finetune_class.py script
    # DST_model: created from SemSeg build_segmentor() function => check mem.py EvBEiT()

    with torch.no_grad():
        all_keys_copied = []
        SRC_state_dict = copy.deepcopy(SRC_model.state_dict())
        all_keys_dst_model = DST_model.state_dict().keys()
        N_tokens_DST = DST_model.backbone.patch_embed.num_patches
        N_tokens_SRC = SRC_model.patch_embed.num_patches
        for src_key, src_v in SRC_model.state_dict().items():
            if "pos_embed" in src_key and N_tokens_SRC != N_tokens_DST:
                new_pos_embed = interpolate_pos_embed(DST_model.backbone, src_v)

                # now copy the new pos_embed
                SRC_state_dict.pop(src_key)
                src_key = f"backbone.{src_key}"
                SRC_state_dict.update({src_key: new_pos_embed})
                print(f"Copying {src_key} to DST_model")
                all_keys_copied.append(src_key)
            if f"backbone.{src_key}" in all_keys_dst_model:
                SRC_state_dict.pop(src_key)
                src_key = f"backbone.{src_key}"
                SRC_state_dict.update({src_key: src_v})
                print(f"Copying {src_key} to DST_model")
                all_keys_copied.append(src_key)
            else:
                SRC_state_dict.pop(src_key)
                print(f"(not copying {src_key} to DST_model)")

    with torch.no_grad():
        DST_model.load_state_dict(SRC_state_dict, strict=False)

    for k in all_keys_copied:
        if k not in all_keys_dst_model:
            print(f"Did not initialize {k} in DST_model")

    print(f"copied blocks(SRC_model) => blocks(model)")
    
def copy_vit_blocks(src_model, dst_model):
    for i in range(len(dst_model.backbone.blocks)):
       dst_model.backbone.blocks[i].attn.proj.weight.copy_(src_model.blocks[i].attn.proj.weight)
       dst_model.backbone.blocks[i].mlp.fc2.weight.copy_(src_model.blocks[i].mlp.fc2.weight)


def get_model_string(cfg):
    ps = cfg.model["backbone"]["patch_size"]
    emb = cfg.model["backbone"]["embed_dim"]
    depth = cfg.model["backbone"]["depth"]
    heads = cfg.model["backbone"]["num_heads"]
    mlp = cfg.model["backbone"]["mlp_ratio"]
    model_string = f"classification_patch{ps}_224x224_emb{emb}_depth{depth}_heads{heads}_mlp{mlp}"
    return model_string


def load_state_dict_FT(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

def load_ckpt_same_as_class_finetune(model, args, cfg):
    args.finetune = cfg.load_from
    print(f"args.finetune={args.finetune}")
    args.model_key = "model|module"
    args.model_prefix = ''
    print(f"Load ckpt_same_as_ft {args.finetune}")

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if model.use_rel_pos_bias and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
            print("Expand the shared relative position embedding to each transformer block. ")
            num_layers = model.get_num_layers()
            rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
            for i in range(num_layers):
                checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

            checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

            if "relative_position_bias_table" in key:
                rel_pos_bias = checkpoint_model[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = model.state_dict()[key].size()
                dst_patch_shape = model.patch_embed.patch_shape
                # print(dst_patch_shape)
                # if dst_patch_shape[0] != dst_patch_shape[1]:
                #     raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                print(dst_patch_shape, src_size, dst_size)

                if src_size != dst_size:
                    print("Position interpolate for %s from %dx%d to %dx%d" % (
                        key, src_size, src_size, dst_size, dst_size))
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                    checkpoint_model[key] = new_rel_pos_bias

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        load_state_dict_FT(model, checkpoint_model, prefix=args.model_prefix)
        # model.load_state_dict(checkpoint_model, strict=False)
    
    return model


def load_checkpoint_npz(filename, cfg, args, model):
    print(f"loading 1k checkpint from {filename}")
    assert "B_16-i1k" in filename and ".npz" in filename # TODO: find 21k.npz and add it here

    if cfg.VIT == "base":
        model_string =  f"vit_base_patch16_224"
    elif cfg.VIT == "small":
        model_string =  "vit_small_patch16_224"
    elif cfg.VIT == "tiny":
        model_string =  "vit_tiny_patch16_224"
    else:
        print(f"{cfg.VIT} not implemented")

    print(f"Creating ViT {model_string} with timm fct for loading 1k-PT model (same as class_finetune.py)")
    tmp_model_timm = timm.create_model(
        model_string,
        pretrained=False, # is always False for this fct
        num_classes=None, # TODO: this does not matter I guess
        drop_rate=0,
        drop_path_rate=cfg.model["backbone"]["drop_path_rate"],
        attn_drop_rate=0,
        drop_block_rate=None,
        qkv_bias=True,
    )
    timm.models.load_checkpoint(tmp_model_timm, filename)
    copy_tmpModelTimm_to_semsegBackbone(tmp_model_timm, model)
    assert_tmpModelTimm_equal_semsegModel(tmp_model_timm, model)

    return model


def load_checkpoint_21k(cfg, args, model):
    print(f"loading 21k checkpint from {cfg.load_from}")
    assert "PT21k" == cfg.load_from

    if cfg.VIT == "base":
        model_string =  f"vit_base_patch16_224"
    elif cfg.VIT == "small":
        model_string =  f"vit_small_patch16_224"
    elif cfg.VIT == "tiny":
        model_string =  f"vit_tiny_patch16_224"
    else:
        print(f"{cfg.VIT} not implemented")

    tmp_model_timm = timm.create_model(
        model_string,
        pretrained=True,
        num_classes=0,
        drop_rate=0,
        drop_path_rate=cfg.model["backbone"]["drop_path_rate"],
        attn_drop_rate=0,
        drop_block_rate=None,
        qkv_bias=True,
    )
    copy_tmpModelTimm_to_semsegBackbone(tmp_model_timm, model)
    assert_tmpModelTimm_equal_semsegModel(tmp_model_timm, model)

    return model


def train_segmentor(model,
                    dataset,
                    cfg,
                    args,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    if cfg.load_from is not None and ".npz" in cfg.load_from[-4:]:
        model = load_checkpoint_npz(cfg.load_from, cfg, args, model)
    elif "PT21k" == cfg.load_from:
        model = load_checkpoint_21k(cfg, args, model)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True, 
            persistent_workers=False) for ds in dataset
    ]

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # use apex fp16 optimizer
    if cfg.optimizer_config.get("type", None) and cfg.optimizer_config["type"] == "DistOptimizerHook":
        if cfg.optimizer_config.get("use_fp16", False):
            model, optimizer = apex.amp.initialize(
                model.cuda(), optimizer, opt_level="O1")
            for m in model.modules():
                if hasattr(m, "fp16_enabled"):
                    m.fp16_enabled = True
                    print(f"\n\n\n\n  ********** Set fp_16 = True!!  ********** \n\n\n\n")

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks # TODO: checkpoint_config
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate: # this is true for us
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = 'IterBasedRunner' not in cfg.runner['type']
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # Load pretrained weights before DDP and optimizer-init stuff
    if cfg.resume_from: # this loads the whole model (incl. heads) & optimizer state
        runner.resume(cfg.resume_from)
        print(f"Resuming from {cfg.resume_from}")
    elif cfg.load_from is not None and distributed is False and ".npz" not in cfg.load_from[-4:] and "PT21k" != cfg.load_from and "-tmp.pth" not in cfg.load_from:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        state_dict = torch.load(cfg.load_from)["model"]

        block_idxs = []
        for k, v in state_dict.items():
            if "block" in k or "cls_token" in k or "patch_emb" in k:
                k = 'backbone.' + k   # add prefix
            new_state_dict[k] = v

        new_path = cfg.load_from[:-4]+"-tmp.pth"
        torch.save(new_state_dict, new_path)
        runner.load_checkpoint(new_path)
        print(f"Loaded pretrained model from {cfg.load_from}")
        sys.exit()
    elif cfg.load_from is not None and ".npz" not in cfg.load_from[-4:] and "PT21k" != cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
        print(f"Loaded pretrained model from {cfg.load_from}")

    # Assert that we construct&load model same as classification finetune
    if cfg.load_from is not None and distributed is False and ".npz" not in cfg.load_from[-4:] and "PT21k" != cfg.load_from:
        model_string = get_model_string(cfg)
        print(f"Creating ViT {model_string} with timm fct (same as class_finetune.py)")
        tmp_model_timm = create_model_timm(
            model_string,
            pretrained=False, # is always False for this fct
            num_classes=None, # TODO: this does not matter I guess
            drop_rate=0,
            drop_path_rate=cfg.model["backbone"]["drop_path_rate"],
            attn_drop_rate=0,
            drop_block_rate=None,
            use_mean_pooling=True,
            init_scale=0.001,
            use_rel_pos_bias=not cfg.model["backbone"]["use_abs_pos_emb"],
            use_abs_pos_emb=cfg.model["backbone"]["use_abs_pos_emb"],
            init_values=cfg.model["backbone"]["init_values"],
        )
        tmp_model_timm = load_ckpt_same_as_class_finetune(tmp_model_timm, args, cfg)
        tmp_model_timm = tmp_model_timm.to("cuda")
        assert_tmpModelTimm_equal_semsegModel(tmp_model_timm, model.module)

    print(f"\nPrinting Model")
    m = model.module
    for name, param in m.named_parameters():
        print(f"{name}, {param.sum():03f}, {torch.mean(param):03f}, {param.abs().sum():03f}, {torch.median(param):03f}, {param.requires_grad}")
    print(f"*** Done Printing Model\n")

    runner.run(data_loaders, cfg.workflow)
