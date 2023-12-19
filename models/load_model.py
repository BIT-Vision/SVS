import time
import torch
import logging


def load_checkpoint(model, model_file, optimizer):
    logger = logging.getLogger(__name__)
    t_start = time.time()
    if isinstance(model_file, str):
        device = torch.device('cpu')
        checkpoint = torch.load(model_file, map_location=device)
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
    else:
        state_dict = model_file
    t_ioend = time.time()

    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(state_dict, strict=False)
    epoch = checkpoint['epoch']

    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logger.info("Load model, Time usage: IO: {}, initialize parameters: {}".format(t_ioend - t_start, t_end - t_ioend))
    logger.info("Start train at epoch {}".format(epoch))
    return model, optimizer, epoch


def load_model(model, model_file):
    logger = logging.getLogger(__name__)
    t_start = time.time()
    if isinstance(model_file, str):
        device = torch.device('cpu')
        checkpoint = torch.load(model_file, map_location=device)
        # print(checkpoint.keys())
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = model_file
    t_ioend = time.time()

    epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else 0
    model.load_state_dict(state_dict, strict=False)

    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logger.info("Load model, Time usage: IO: {}, initialize parameters: {}".format(t_ioend - t_start, t_end - t_ioend))
    logger.info("Current epoch is {}".format(epoch))

    return model