import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import os
import glob
import logging
import warnings

warnings.filterwarnings('ignore')

import shutil
import argparse
from tqdm import tqdm
import time

from datasets import get_transforms, get_datasets
from utils import BCE_SSIM_LOSS, AverageMeter, LR_Manage, setup_logging, setup_seed, StructureMeasure
from config import cfg

from models import build_model, load_model, load_checkpoint
from spikingjelly.clock_driven import functional


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', default='', help='path to the pretrained checkpoint')

    parser.add_argument('--exp_name', default='exp', help='exp name')
    parser.add_argument('--log_dir', default='logs/', help='log_dir path')
    parser.add_argument('--gpu', type=str, default='8', help='gpu id')

    parser.add_argument('--step', default=False, action='store_true', help='steps')
    parser.add_argument('--clip', default=False, action='store_true', help='calculate loss for every step')

    parser.add_argument('--seed', default=1024, type=int)
    return parser.parse_args()


def main():
    log_dir = args.log_dir

    summary_writer = SummaryWriter(log_dir=log_dir)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if cuda:
        torch.backends.cudnn.benchmark = True
        current_device = torch.cuda.current_device()
        logger.info("Running on " + torch.cuda.get_device_name(current_device))
    else:
        logger.info("Running on CPU")

    model = build_model(args)
    model.to(device)
    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

    data_mode = cfg.DATA.DATA_MODE
    data_transforms = get_transforms(input_size=cfg.DATA.SIZE, image_mode=True)

    read_clip = cfg.DATA.CLIP_LEN > 1
    train_dataset = get_datasets(name_list=[cfg.DATA.DATA_NAME],
                                 split_list=["train"],
                                 config_path=cfg.DATA.DATASET_CONFIG,
                                 root=cfg.DATA.ROOT,
                                 training=True,
                                 transforms=data_transforms['train'],
                                 read_clip=read_clip,
                                 random_reverse_clip=False,
                                 clip_len=cfg.DATA.CLIP_LEN,
                                 data_mode=data_mode,
                                 prepare=cfg.DATA.PREPARE,
                                 interval=cfg.DATA.INTERVAL)

    val_dataset = get_datasets(name_list=[cfg.DATA.DATA_NAME],
                               split_list=["val"],
                               config_path=cfg.DATA.DATASET_CONFIG,
                               root=cfg.DATA.ROOT,
                               training=True,
                               transforms=data_transforms['val'],
                               read_clip=read_clip,
                               random_reverse_clip=False,
                               clip_len=cfg.DATA.CLIP_LEN,
                               data_mode=data_mode,
                               prepare=cfg.DATA.PREPARE)

    train_dataloader = data.DataLoader(dataset=train_dataset,
                                       batch_size=cfg.TRAIN.BATCH_SIZE,
                                       num_workers=cfg.TRAIN.NUM_WORKER,
                                       shuffle=True,
                                       drop_last=True,
                                       pin_memory=True)
    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        #  batch_size=1,
        num_workers=cfg.TRAIN.NUM_WORKER,
        shuffle=False,
        pin_memory=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    params = [params for name, params in model.named_parameters() if params.requires_grad]
    optimizer = torch.optim.AdamW([{
        'params': params,
        'lr': cfg.TRAIN.LR,
        'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
        'names': "params"
    }])

    start_epoch = 0
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            logger.info('Loading state dict from: {0}'.format(args.checkpoint))
            model, optimizer, start_epoch = load_checkpoint(model=model,
                                                            model_file=args.checkpoint,
                                                            optimizer=optimizer)
        else:
            raise ValueError("Cannot find model file at {}".format(args.checkpoint))

    train(args, model, train_dataloader, val_dataloader, optimizer, device, summary_writer, start_epoch)


def train(args, model, train_dataloader, val_dataloader, optimizer, device, summary_writer, start_epoch=0):
    loss_meter = {'train': AverageMeter(), 'val': AverageMeter()}
    mae_meter = {'train': AverageMeter(), 'val': AverageMeter()}
    smean_meter = {'train': AverageMeter(), 'val': AverageMeter()}

    bce_ssim_loss = BCE_SSIM_LOSS()
    best_smeasure = 0.0
    best_epoch = 0

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    tblog_iter = {'train': len(train_dataloader) // 10, 'val': len(val_dataloader)}
    log_iter = {'train': len(train_dataloader) * start_epoch, 'val': len(val_dataloader) * start_epoch}
    all_iter = {'train': len(train_dataloader) * cfg.TRAIN.EPOCH, 'val': len(val_dataloader) * cfg.TRAIN.EPOCH}
    max_itr = all_iter['train']

    lr_manage = LR_Manage(base_lr=cfg.TRAIN.LR, max_itr=max_itr, min_lr=cfg.TRAIN.LR / 10)
    lr = 0

    for epoch in range(start_epoch, cfg.TRAIN.EPOCH):
        phases = ['train', 'val']
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            logger.info("Phase: {} Epoch: {} | {}".format(phase, epoch + 1, cfg.TRAIN.EPOCH))
            pbar = tqdm(dataloaders[phase])
            for data in pbar:
                if log_iter[phase] % tblog_iter[phase] == 0:
                    loss_meter[phase].reset()
                    mae_meter[phase].reset()
                    if phase == 'train':
                        smean_meter[phase].reset()

                log_iter[phase] += 1

                if isinstance(data, list):
                    images = torch.stack([item['image'].to(device) for item in data], dim=0)
                    labels = torch.stack([item['label'].to(device) for item in data], dim=0)
                else:
                    images = torch.stack([data['image'].to(device)], dim=0)
                    labels = torch.stack([data['label'].to(device)], dim=0)

                functional.reset_net(model)
                states = None
                preds = list()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    preds, _, states, features = model(images, states)
                    preds = F.sigmoid(preds)
                    loss = bce_ssim_loss(preds, labels)
                    if phase == 'train':
                        lr = lr_manage.adjust_learning_rate(optimizer, log_iter[phase])
                        torch.autograd.backward(loss)
                        optimizer.step()

                loss_meter[phase].update(loss.item())

                for i, (label_, pred_) in enumerate(zip(labels, preds)):
                    for j, (label, pred) in enumerate(zip(label_.detach().cpu(), pred_.detach().cpu())):
                        label_idx = label[0, :, :].numpy()
                        step = pred.shape[0]
                        for k in range(step):
                            pred_idx = pred[k, :, :].numpy()
                            # if phase == 'val':
                            smean_meter[phase].update(
                                StructureMeasure(pred_idx.astype(np.float32),
                                                 (label_idx >= 0.5).astype(np.bool_)).item())
                            mae_meter[phase].update(np.abs(pred_idx - label_idx).mean().item())

                desc = f'[LR:{lr:.7f} | {phase}_Loss:{loss_meter[phase].avg:.4f} | {phase}_MAE:{mae_meter[phase].avg:.4f} | S-measure:{smean_meter[phase].avg:.4f}]'
                pbar.desc = desc
                if log_iter[phase] % tblog_iter[phase] == 0:
                    summary_writer.add_scalar(f'{phase}_Loss', loss_meter[phase].avg,
                                              log_iter[phase] // tblog_iter[phase])
                    summary_writer.add_scalar(f'{phase}_MAE', mae_meter[phase].avg,
                                              log_iter[phase] // tblog_iter[phase])
                    summary_writer.add_scalar(f'{phase}_S', smean_meter[phase].avg,
                                              log_iter[phase] // tblog_iter[phase])
                    logger.info(
                        f'Phase: {phase} Epoch: [{epoch + 1} | {cfg.TRAIN.EPOCH}] | Iter: [{log_iter[phase]} | {all_iter[phase]}] | '
                        + desc)

            logger.info('{} Loss: {:.4f}'.format(phase, loss_meter[phase].avg))
            logger.info('{} MAE: {:.4f}'.format(phase, mae_meter[phase].avg))

            # save current best epoch
            if phase == 'val':
                epoch_smeasure = smean_meter[phase].avg
                smean_meter[phase].reset()
                logger.info('{} S-measure: {:.4f}'.format(phase, epoch_smeasure))
                if epoch_smeasure > best_smeasure:
                    best_smeasure = epoch_smeasure
                    best_epoch = epoch
                    model_path = os.path.join(args.checkpoint_dir, "checkpoint_best.pth")
                    torch.save(
                        {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'train_loss': loss_meter['train'].avg,
                            'val_loss': loss_meter['val'].avg,
                            'train_mae': mae_meter['train'].avg,
                            'val_mae': mae_meter['val'].avg,
                            'S-measure': epoch_smeasure,
                            'optimizer': optimizer.state_dict()
                        }, model_path)
                    logger.info("Saving current best model at: {}".format(model_path))

        if (epoch + 1) % cfg.TRAIN.SKIP_SAVE == 0:
            # save model
            model_path = os.path.join(args.checkpoint_dir, "checkpoint_{}.pth".format(epoch + 1))
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'train_loss': loss_meter['train'].avg,
                    'val_loss': loss_meter['val'].avg,
                    'train_mae': mae_meter['train'].avg,
                    'val_mae': mae_meter['val'].avg,
                    'S-measure': epoch_smeasure,
                    'optimizer': optimizer.state_dict()
                }, model_path)
            logger.info("Backup model at: {}".format(model_path))

    logger.info('Best S-measure: {} at epoch {}'.format(best_smeasure, best_epoch + 1))


if __name__ == "__main__":
    args = parse_args()

    # Set seed
    setup_seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Logs
    prefix = args.exp_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '-%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path)

    scripts_to_save = ['train.py', 'config.py', 'inference.py']
    scripts_to_save += list(glob.glob(os.path.join('models', '*.py')))
    scripts_to_save += list(glob.glob(os.path.join('datasets', '*.py')))
    scripts_to_save += list(glob.glob(os.path.join('utils', '*.py')))

    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))

    logger = logging.getLogger(__name__)
    logger.info('Config: {}'.format(cfg))
    logger.info('Arguments: {}'.format(args))
    logger.info('Experiment: {}'.format(args.exp_name))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main()
